#!/usr/bin/env python3
"""Archived orbital-element JPDA baseline and reusable exact local marginals.

The standalone script compares against the original 13-dimensional simulator
using the same generated detection stream. Its track initiation, update, and
acceptance policies differ from that pipeline, so its archived percentages
are a system-level comparison, not a controlled association effect. The
paired Rust-stream studies in ``experiments/rust_online`` provide a common
candidate pool and lifecycle instead.

``jpda_marginals_exact`` computes feasible one-to-one event marginals by
dynamic programming over detection subsets. The online study calls it on
local ambiguity clusters and logs a one-to-one MAP fallback beyond the
configured exact-solver cap. The original script retains its historical
settings for reproduction.
"""

import sys
import time
import json
import argparse
import itertools
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbital_physics as op
import data_generator as gen

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = str(REPO_ROOT / "data" / "leo_catalog.csv")
OUTPUT_PATH = str(REPO_ROOT / "outputs" / "jpda_results.json")

# --- Must match ground_station_tracking_simulation.py exactly ---
ACCELERATION_FACTOR = 300
SIMULATION_DURATION_SECONDS = 60
PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
MEASUREMENT_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}
SEED = 11
UPDATE_CONFIRMATIONS_REQUIRED = 2   # tentative -> confirmed track, mirrors the GNN pipeline

# --- JPDA-specific tuning ---
GATE_CHI2 = 9.21          # chi-square 99% for 2 effectively-independent gated dims (a, cross-track angle)
PD = 0.98                 # probability of detection given a track is illuminated this pass
CLUTTER_DENSITY = 1e-6    # spatial density of false alarms in normalized measurement space (Poisson clutter)
JPDA_MAX_CLUSTER_ENUM = 8 # max #tracks (or #detections) in a cluster before falling back to greedy
STATE_DIM = 5              # a, e, i, raan, argp

# Per-element measurement variance (diagonal R), matching MEASUREMENT_NOISE**2
R_DIAG = np.array([
    MEASUREMENT_NOISE["a_km"] ** 2,
    MEASUREMENT_NOISE["e"] ** 2,
    MEASUREMENT_NOISE["i_deg"] ** 2,
    MEASUREMENT_NOISE["angle_deg"] ** 2,
    MEASUREMENT_NOISE["angle_deg"] ** 2,
])
# Per-element process variance (diagonal Q), matching PROCESS_NOISE**2
Q_DIAG = np.array([
    PROCESS_NOISE["a_km"] ** 2,
    PROCESS_NOISE["e"] ** 2,
    PROCESS_NOISE["i_deg"] ** 2,
    PROCESS_NOISE["angle_deg"] ** 2,
    PROCESS_NOISE["angle_deg"] ** 2,
])

# Feature scale for Mahalanobis gating -- angles need wraparound-aware
# residuals just like identity_embedding.py's sin/cos handling, so the
# raw-element residual for RAAN/argp is wrapped to [-180, 180] before
# scaling. This keeps JPDA's gating physically honest rather than
# penalizing a true near-0/360 match.
def _wrap_deg(delta):
    return (delta + 180.0) % 360.0 - 180.0


def state_vec(a, e, i, raan, argp):
    return np.array([a, e, i, raan, argp], dtype=float)


def residual(meas, pred):
    """Element-wise residual, angle-wrapped for raan/argp (indices 3, 4)."""
    d = meas - pred
    d[3] = _wrap_deg(d[3])
    d[4] = _wrap_deg(d[4])
    return d


class Track:
    __slots__ = ["id", "true_name", "x", "P", "epoch", "bc", "obj_type",
                 "size", "rcs", "confirmed", "pending_name", "confidence",
                 "last_seen", "retired"]

    def __init__(self, tid, x, P, epoch, bc, obj_type, size, rcs, true_name=None):
        self.id = tid
        self.true_name = true_name       # bookkeeping only, ground truth
        self.x = x                       # (5,) state: a, e, i, raan, argp
        self.P = P                       # (5,5) covariance
        self.epoch = epoch
        self.bc = bc
        self.obj_type = obj_type
        self.size = size
        self.rcs = rcs
        self.confirmed = True
        self.pending_name = None
        self.confidence = 1
        self.last_seen = epoch
        self.retired = False

    def predict(self, real_now):
        dt = real_now - self.epoch
        if dt <= 0:
            return self.x.copy(), self.P.copy()
        a, e, i, raan, argp = self.x
        new_a, new_e, new_i, new_raan, new_argp = op.propagate_catalog_analytic(
            np.array([a]), np.array([e]), np.array([i]),
            np.array([raan]), np.array([argp]), np.array([self.bc]), dt,
        )
        x_pred = state_vec(new_a[0], new_e[0], new_i[0], new_raan[0], new_argp[0])
        # Linear-Gaussian process covariance growth: additive Q scaled by
        # elapsed time (consistent with PROCESS_NOISE being a per-detection
        # stddev in the sim -- here treated as a per-second rate for a
        # proper KF covariance, which is the standard JPDA/KF convention
        # the sim's own EMA-update approach does not use).
        P_pred = self.P + np.diag(Q_DIAG) * max(dt, 1.0) / max(SIMULATION_DURATION_SECONDS, 1.0) * 60.0
        return x_pred, P_pred


def mahalanobis_sq(res, S):
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
    return float(res @ Sinv @ res)


def gaussian_likelihood(res, S):
    d = len(res)
    try:
        det = np.linalg.det(S)
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        det = max(np.linalg.det(S), 1e-300)
        Sinv = np.linalg.pinv(S)
    det = max(det, 1e-300)
    m2 = res @ Sinv @ res
    norm = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
    return norm * np.exp(-0.5 * m2)


def enumerate_joint_events(track_ids, det_ids, feasible):
    """Yield every feasible joint association event for a cluster: each
    dict maps track_id -> det_id or None (missed detection this pass).
    `feasible[(t, d)]` must be True for gated pairs. Standard JPDA
    constraint: each detection used by at most one track."""
    det_options = {t: [None] + [d for d in det_ids if feasible.get((t, d), False)] for t in track_ids}
    for combo in itertools.product(*[det_options[t] for t in track_ids]):
        used = [d for d in combo if d is not None]
        if len(used) != len(set(used)):
            continue  # a detection can't serve two tracks in one event
        yield dict(zip(track_ids, combo))


def jpda_marginals_exact(match_weights, miss_weight=0.08, clutter_weight=0.02):
    """Exact local JPDA marginals by detection-mask dynamic programming.

    ``match_weights[t, d]`` is zero for an infeasible gated edge and otherwise
    includes detection probability times its measurement likelihood. Each
    missed track receives ``miss_weight``; each unused detection receives
    ``clutter_weight``. The common clutter factor cancels on normalization,
    leaving O(n_tracks * n_detections * 2**n_detections) work instead of
    materializing every joint event. This is intended for small clusters.
    """
    weights = np.asarray(match_weights, dtype=float)
    if weights.ndim != 2 or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("match_weights must be a finite nonnegative matrix")
    if miss_weight <= 0 or clutter_weight <= 0:
        raise ValueError("miss and clutter weights must be positive")
    n_tracks, n_dets = weights.shape
    if n_dets > 20:
        raise ValueError("exact JPDA detection-mask DP is limited to 20 detections")
    if n_tracks == 0 or n_dets == 0:
        return np.zeros_like(weights)
    scaled = weights / clutter_weight
    states = 1 << n_dets
    forward = np.zeros((n_tracks + 1, states), dtype=float)
    forward[0, 0] = 1.
    for track in range(n_tracks):
        for mask in range(states):
            mass = forward[track, mask]
            if mass == 0: continue
            forward[track + 1, mask] += mass * miss_weight
            for det in range(n_dets):
                if not mask & (1 << det) and scaled[track, det] > 0:
                    forward[track + 1, mask | (1 << det)] += mass * scaled[track, det]
    backward = np.ones((n_tracks + 1, states), dtype=float)
    for track in range(n_tracks - 1, -1, -1):
        for mask in range(states):
            total = miss_weight * backward[track + 1, mask]
            for det in range(n_dets):
                if not mask & (1 << det) and scaled[track, det] > 0:
                    total += scaled[track, det] * backward[track + 1, mask | (1 << det)]
            backward[track, mask] = total
    normalizer = backward[0, 0]
    if normalizer <= 0 or not np.isfinite(normalizer):
        raise FloatingPointError("JPDA normalizer is not finite and positive")
    beta = np.zeros_like(weights)
    for track in range(n_tracks):
        for mask in range(states):
            mass = forward[track, mask]
            if mass == 0: continue
            for det in range(n_dets):
                if not mask & (1 << det) and scaled[track, det] > 0:
                    beta[track, det] += (mass * scaled[track, det]
                                         * backward[track + 1, mask | (1 << det)])
    return beta / normalizer


def jpda_cluster_update(tracks_in_cluster, dets_in_cluster, det_vecs, det_true_names,
                         real_now, degraded_fallback_counter):
    """
    Runs full JPDA (or documented greedy fallback if the cluster is too
    large to enumerate) on one connected cluster of {tracks, detections}.
    Returns: list of (track, assigned_det_idx_or_None, beta_assigned,
    combined_residual_or_None) plus a set of detection indices (LOCAL to
    dets_in_cluster) that were used by at least one track's top
    association (for reporting "which detection resolved which track").
    """
    n_t, n_d = len(tracks_in_cluster), len(dets_in_cluster)

    preds = []
    for tr in tracks_in_cluster:
        x_pred, P_pred = tr.predict(real_now)
        S = P_pred + np.diag(R_DIAG)
        preds.append((x_pred, P_pred, S))

    feasible = {}
    lik = {}
    for ti, tr in enumerate(tracks_in_cluster):
        x_pred, P_pred, S = preds[ti]
        for di, dv in enumerate(dets_in_cluster):
            res = residual(dv, x_pred)
            m2 = mahalanobis_sq(res, S)
            if m2 <= GATE_CHI2:
                feasible[(ti, di)] = True
                lik[(ti, di)] = gaussian_likelihood(res, S)

    track_ids = list(range(n_t))
    det_ids = list(range(n_d))

    use_full_jpda = max(n_t, n_d) <= JPDA_MAX_CLUSTER_ENUM
    beta = np.zeros((n_t, n_d + 1))  # beta[:, -1] = probability of "no detection"

    if use_full_jpda:
        total_weight = 0.0
        weighted_beta = np.zeros((n_t, n_d + 1))
        for event in enumerate_joint_events(track_ids, det_ids, feasible):
            w = 1.0
            for ti in track_ids:
                d_assigned = event[ti]
                if d_assigned is None:
                    w *= (1.0 - PD)
                else:
                    w *= PD * lik[(ti, d_assigned)]
            n_false = n_d - sum(1 for v in event.values() if v is not None)
            w *= (CLUTTER_DENSITY ** n_false)
            if w <= 0:
                continue
            total_weight += w
            for ti in track_ids:
                d_assigned = event[ti]
                col = d_assigned if d_assigned is not None else n_d
                weighted_beta[ti, col] += w
        if total_weight > 0:
            beta = weighted_beta / total_weight
        else:
            beta[:, -1] = 1.0  # degenerate: nothing gated, everyone misses
    else:
        # Documented degraded mode: cluster too large to enumerate exactly.
        # Falls back to greedy highest-likelihood assignment (still
        # respects the gate; still one detection per track) rather than
        # silently pretending exact JPDA ran. Counted separately in metrics.
        degraded_fallback_counter[0] += 1
        pairs = sorted(lik.items(), key=lambda kv: -kv[1])
        used_d, used_t = set(), set()
        for (ti, di), likval in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            beta[ti, di] = 1.0
        for ti in track_ids:
            if ti not in used_t:
                beta[ti, -1] = 1.0

    results = []
    for ti, tr in enumerate(tracks_in_cluster):
        x_pred, P_pred, S = preds[ti]
        b = beta[ti]
        assoc_probs = b[:n_d]
        p_miss = b[n_d]
        best_d = int(np.argmax(assoc_probs)) if n_d > 0 else None
        best_p = float(assoc_probs[best_d]) if best_d is not None else 0.0

        # Combined innovation across all gated detections (standard JPDA
        # update, not "just take the argmax" -- ambiguity shows up as
        # covariance inflation, exactly the property that should make
        # JPDA behave conservatively/safely under ambiguity rather than
        # confidently wrong).
        combined_res = np.zeros(STATE_DIM)
        for di, dv in enumerate(dets_in_cluster):
            if assoc_probs[di] > 0:
                combined_res += assoc_probs[di] * residual(dv, x_pred)

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = P_pred @ Sinv

        x_upd = x_pred + K @ combined_res
        # JPDA covariance update (Bar-Shalom): standard KF term plus a
        # spread-of-innovations term reflecting genuine association
        # ambiguity -- this is the mechanism by which JPDA becomes
        # conservative rather than overconfident when multiple detections
        # are plausible, in contrast to a hard nearest-neighbor gate.
        P_c = P_pred - K @ S @ K.T
        spread = np.zeros((STATE_DIM, STATE_DIM))
        for di, dv in enumerate(dets_in_cluster):
            if assoc_probs[di] > 0:
                r = residual(dv, x_pred)
                diff = (K @ r) - (K @ combined_res)
                spread += assoc_probs[di] * np.outer(diff, diff)
        P_upd = p_miss * P_pred + (1 - p_miss) * P_c + spread

        results.append({
            "track": tr, "best_det_local": best_d, "best_prob": best_p,
            "x_upd": x_upd, "P_upd": P_upd, "p_miss": p_miss,
            "assoc_probs": assoc_probs,
        })
    return results


def _write_checkpoint(path, catalog_path, duration, seed, total_detections,
                       known_count, unknown_count, correct_match_count, mismatch_count,
                       update_committed_count, update_withheld_count, catalog_complete_n,
                       total_true_population, n_tracks, degraded, cluster_sizes,
                       lookup_times, elapsed_s, sim_time_reached):
    """Write partial progress to disk so a long run (e.g. a 60s-duration
    run on a large, dense catalog) is safe to interrupt and inspect, and
    so this sandbox's per-call wall-clock limit doesn't lose completed
    work. Overwrites the same file each time -- not a log, a snapshot."""
    total_accepted = correct_match_count + mismatch_count
    snapshot = {
        "algorithm": "JPDA", "partial": True,
        "sim_time_reached_s": sim_time_reached,
        "config": {"catalog_path": catalog_path, "n_objects": total_true_population,
                   "duration_s": duration, "seed": seed},
        "total_detections": total_detections,
        "known_count": known_count, "unknown_count": unknown_count,
        "correct_matches": correct_match_count, "mismatches": mismatch_count,
        "correct_matches_pct": 100.0 * correct_match_count / total_accepted if total_accepted else 0.0,
        "catalog_completeness_pct": 100.0 * catalog_complete_n / total_true_population if total_true_population else 0.0,
        "catalog_complete_n": catalog_complete_n, "catalog_total_n": total_true_population,
        "final_catalog_size": n_tracks, "degraded_fallback_clusters": degraded,
        "n_clusters_processed": len(cluster_sizes),
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "max_cluster_size": int(np.max(cluster_sizes)) if cluster_sizes else 0,
        "lookup_us_mean": float(np.mean(lookup_times)) if lookup_times else 0.0,
        "real_wall_time_s_so_far": elapsed_s,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2)
    Path(tmp).replace(path)


def run_jpda(catalog_path=CATALOG_PATH, duration=SIMULATION_DURATION_SECONDS,
             seed=SEED, verbose=True, log_every=2000, max_wall_seconds=None,
             checkpoint_path=None, checkpoint_every=10000):
    catalog_df = pd.read_csv(catalog_path)
    rng = np.random.default_rng(seed)

    truth_a = catalog_df["a"].to_numpy(dtype=float).copy()
    truth_e = catalog_df["e"].to_numpy(dtype=float).copy()
    truth_i = catalog_df["i"].to_numpy(dtype=float).copy()
    truth_raan = catalog_df["Omega"].to_numpy(dtype=float).copy()
    truth_argp = catalog_df["omega"].to_numpy(dtype=float).copy()
    truth_bc = catalog_df["bc_m2_per_kg"].to_numpy(dtype=float).copy()
    truth_epoch = np.zeros(len(catalog_df))

    names = catalog_df["name"].tolist()
    types = catalog_df["type"].tolist()
    sizes = catalog_df["size_meters"].to_numpy(dtype=float)
    rcs_vals = catalog_df["rcs"].to_numpy(dtype=float)

    real_periods = 2 * np.pi * np.sqrt(truth_a ** 3 / op.MU_EARTH)
    sim_periods = real_periods / ACCELERATION_FACTOR

    import heapq
    heap = []
    for idx in range(len(catalog_df)):
        phase = rng.uniform(0, sim_periods[idx])
        heapq.heappush(heap, (phase, idx))

    # NOTE: matches ground_station_tracking_simulation.py's scheduler
    # exactly -- every object is unconditionally rescheduled for its next
    # orbital pass after being processed (see that file's
    # `heapq.heappush(heap, (sim_time + sim_periods[idx], idx))` at the
    # end of its main loop). Without this, each object is only ever
    # detected once across the whole run and JPDA (which depends on
    # repeat detections to associate against an existing track) would
    # never get a fair chance to re-acquire anything -- that would make
    # the KNOWN-rate comparison meaningless, not favorable to either side.

    tracks = []          # confirmed + tentative tracks
    tid_counter = 0
    true_name_to_track = {}   # ground-truth bookkeeping only, for scoring
    pending_confirm = {}      # track.id -> (true_name_seen_last_pass, count)

    # --- Sorted-by-semi-major-axis coarse index over ACTIVE tracks, so
    #     gating a new detection is O(log n + k) instead of O(n). This is
    #     purely an implementation-level scaling aid (the same role
    #     leo_ann_index.py's ANN index plays for the GNN pipeline) and
    #     does not change JPDA's actual gate/association math -- every
    #     candidate returned by this index still has to pass the real
    #     chi-square Mahalanobis gate below. Rebuilt periodically rather
    #     than after every single insertion, same rebuild-cadence
    #     rationale as leo_ann_index.py (staying live-updated on every
    #     insert would be the actual bottleneck at this scale).
    INDEX_REBUILD_EVERY = 250
    sorted_a = []            # sorted list of (a_value, track_list_index)
    since_rebuild = [0]

    def rebuild_index():
        pairs = [(tr.x[0], k) for k, tr in enumerate(tracks) if not tr.retired]
        pairs.sort()
        sorted_a[:] = pairs
        since_rebuild[0] = 0

    def coarse_candidates(meas_a_val, window=5.0):
        if not sorted_a:
            return []
        keys = [p[0] for p in sorted_a]
        lo = bisect_left(keys, meas_a_val - window)
        hi = bisect_right(keys, meas_a_val + window)
        return [tracks[sorted_a[k][1]] for k in range(lo, hi) if not tracks[sorted_a[k][1]].retired]

    # --- metrics, deliberately named to match eval_results.json's
    #     tracking_simulation block so the two runs are directly comparable ---
    total_detections = 0
    known_count = 0
    unknown_count = 0
    correct_match_count = 0
    mismatch_count = 0
    update_committed_count = 0
    update_withheld_count = 0
    lookup_times = []
    degraded_fallback_counter = [0]
    cluster_sizes = []
    total_associations_considered = 0  # sum over dets of (#tracks gated to it) -- proxy for combinatorial cost

    # --- "shell" breakdown: if any catalog object names contain "_shell"
    #     (dense_shell_catalog_generator.py's naming convention), track
    #     metrics for those separately from the sparse background
    #     population. This is the number that actually matters for a
    #     density-regime test -- averaging shell performance in with a
    #     mostly-sparse background dilutes exactly the effect the test
    #     exists to measure. No-op (all zero) on a catalog with no shell
    #     objects, e.g. the original leo_catalog.csv.
    shell_total = 0
    shell_known = 0
    shell_correct = 0
    shell_mismatch = 0
    shell_lookup_times = []
    shell_cluster_sizes = []

    real_start = time.perf_counter()
    wall_limit_hit = False

    while heap:
        if max_wall_seconds is not None and (time.perf_counter() - real_start) > max_wall_seconds:
            wall_limit_hit = True
            if verbose:
                print(f"[wall-clock budget of {max_wall_seconds}s reached -- stopping early, "
                      f"partial results only]")
            break

        sim_time, idx = heapq.heappop(heap)
        if sim_time > duration:
            break
        real_now = sim_time * ACCELERATION_FACTOR

        dt_truth = real_now - truth_epoch[idx]
        new_a, new_e, new_i, new_raan, new_argp = op.propagate_elements(
            np.array([truth_a[idx]]), np.array([truth_e[idx]]), np.array([truth_i[idx]]),
            np.array([truth_raan[idx]]), np.array([truth_argp[idx]]), np.array([truth_bc[idx]]),
            dt_truth, process_noise=PROCESS_NOISE, rng=rng,
        )
        truth_a[idx], truth_e[idx], truth_i[idx] = new_a[0], new_e[0], new_i[0]
        truth_raan[idx], truth_argp[idx] = new_raan[0], new_argp[0]
        truth_epoch[idx] = real_now

        meas_a = truth_a[idx] + rng.normal(0, MEASUREMENT_NOISE["a_km"])
        meas_e = max(0.0, truth_e[idx] + rng.normal(0, MEASUREMENT_NOISE["e"]))
        meas_i = truth_i[idx] + rng.normal(0, MEASUREMENT_NOISE["i_deg"])
        meas_raan = (truth_raan[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360
        meas_argp = (truth_argp[idx] + rng.normal(0, MEASUREMENT_NOISE["angle_deg"])) % 360
        meas_vec = state_vec(meas_a, meas_e, meas_i, meas_raan, meas_argp)
        name = names[idx]

        t0 = time.perf_counter()

        if since_rebuild[0] >= INDEX_REBUILD_EVERY or not sorted_a:
            rebuild_index()

        # --- Gate this single detection against every active track's
        #     predicted state (JPDA gating step) ---
        # Coarse semi-major-axis window pre-filter via the sorted index
        # above, then the real chi-square Mahalanobis gate on that
        # candidate subset -- this is a scaling aid only (equivalent in
        # spirit to the ANN candidate pool the GNN pipeline uses in
        # leo_ann_index.py) and does NOT change which tracks pass the
        # actual gate.
        gated = []
        candidates = coarse_candidates(meas_a, window=5.0)
        for tr in candidates:
            if abs(tr.x[2] - meas_i) >= 2.0:
                continue
            x_pred, P_pred = tr.predict(real_now)
            S = P_pred + np.diag(R_DIAG)
            res = residual(meas_vec, x_pred)
            m2 = mahalanobis_sq(res, S)
            if m2 <= GATE_CHI2:
                gated.append(tr)
        total_associations_considered += len(gated)

        if gated:
            # Build the local cluster: this detection + every track gated
            # to it (a single-detection cluster degenerates to standard
            # PDA, which is the correct, well-known special case of JPDA
            # when only one measurement is in play).
            cluster_sizes.append(len(gated))
            results = jpda_cluster_update(
                gated, [meas_vec], [meas_vec], [name], real_now, degraded_fallback_counter,
            )
            # Pick the track with the highest marginal association prob
            # to this (the only) detection -- if it beats p_miss, accept.
            best = max(results, key=lambda r: r["best_prob"])
            accepted = best["best_prob"] > best["p_miss"] and best["best_prob"] > 0
        else:
            accepted = False
            best = None

        total_detections += 1
        is_shell = "_shell" in name
        if is_shell:
            shell_total += 1
            if gated:
                shell_cluster_sizes.append(len(gated))

        if accepted:
            tr = best["track"]
            known_count += 1
            matched_correctly = (tr.true_name == name)
            if matched_correctly:
                correct_match_count += 1
            else:
                mismatch_count += 1
            if is_shell:
                shell_known += 1
                if matched_correctly:
                    shell_correct += 1
                else:
                    shell_mismatch += 1

            # Confirmation gate mirrors the GNN pipeline's
            # UPDATE_CONFIRMATIONS_REQUIRED so track-initiation policy
            # isn't an asymmetry between the two systems.
            if UPDATE_CONFIRMATIONS_REQUIRED <= 1:
                commit = True
            else:
                prior = pending_confirm.get(tr.id)
                if prior == name:
                    commit = True
                    pending_confirm.pop(tr.id, None)
                else:
                    pending_confirm[tr.id] = name
                    commit = False

            if commit:
                tr.x = best["x_upd"]
                tr.P = best["P_upd"]
                tr.epoch = real_now
                tr.confidence += 1
                update_committed_count += 1
                since_rebuild[0] += 1
            else:
                update_withheld_count += 1
            tr.last_seen = real_now
        else:
            unknown_count += 1
            # Spawn a new tentative track for this unmatched detection --
            # standard JPDA/MHT track-initiation behavior.
            if name not in true_name_to_track:
                bc = op.ballistic_coefficient(sizes[idx], types[idx])
                P0 = np.diag(R_DIAG) * 4.0  # generous initial uncertainty
                tr = Track(tid_counter, meas_vec.copy(), P0, real_now, bc,
                           types[idx], sizes[idx], rcs_vals[idx], true_name=name)
                tid_counter += 1
                tracks.append(tr)
                true_name_to_track[name] = tr
                since_rebuild[0] += 1
            # else: this true object already has a track but this
            # detection failed to gate to it (a genuine JPDA miss) --
            # scored as unknown/unmatched for this pass, no new track
            # spawned for an already-tracked object (mirrors the GNN
            # pipeline's existing_slot-correction intent without granting
            # JPDA a ground-truth shortcut it wouldn't have operationally).

        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        lookup_times.append(elapsed_us)
        if is_shell:
            shell_lookup_times.append(elapsed_us)

        if verbose and total_detections % log_every == 0:
            print(f"[T+{sim_time:7.2f}s sim] processed {total_detections} detections "
                  f"({len(tracks)} tracks, {degraded_fallback_counter[0]} degraded-fallback clusters)")

        if checkpoint_path is not None and total_detections % checkpoint_every == 0:
            _write_checkpoint(
                checkpoint_path, catalog_path, duration, seed, total_detections,
                known_count, unknown_count, correct_match_count, mismatch_count,
                update_committed_count, update_withheld_count, len(true_name_to_track),
                total_true_population=len(catalog_df), n_tracks=len(tracks),
                degraded=degraded_fallback_counter[0], cluster_sizes=cluster_sizes,
                lookup_times=lookup_times, elapsed_s=time.perf_counter() - real_start,
                sim_time_reached=sim_time,
            )

        heapq.heappush(heap, (sim_time + sim_periods[idx], idx))

    real_elapsed = time.perf_counter() - real_start

    total_true_population = len(catalog_df)
    catalog_complete_n = len(true_name_to_track)
    catalog_completeness_pct = 100.0 * catalog_complete_n / total_true_population if total_true_population else 0.0

    total_accepted = correct_match_count + mismatch_count
    correct_matches_pct = 100.0 * correct_match_count / total_accepted if total_accepted else 0.0
    mismatches_pct = 100.0 * mismatch_count / total_accepted if total_accepted else 0.0

    results = {
        "algorithm": "JPDA",
        "partial": wall_limit_hit,
        "sim_time_reached_s": sim_time if wall_limit_hit else duration,
        "config": {
            "catalog_path": catalog_path, "n_objects": total_true_population,
            "duration_s": duration, "seed": seed, "gate_chi2": GATE_CHI2,
            "pd": PD, "clutter_density": CLUTTER_DENSITY,
            "max_cluster_enum": JPDA_MAX_CLUSTER_ENUM,
            "update_confirmations_required": UPDATE_CONFIRMATIONS_REQUIRED,
        },
        "total_detections": total_detections,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "real_wall_time_s": real_elapsed,
        "update_committed": update_committed_count,
        "update_withheld": update_withheld_count,
        "correct_matches": correct_match_count,
        "correct_matches_pct": correct_matches_pct,
        "mismatches": mismatch_count,
        "mismatches_pct": mismatches_pct,
        "catalog_completeness_pct": catalog_completeness_pct,
        "catalog_complete_n": catalog_complete_n,
        "catalog_total_n": total_true_population,
        "final_catalog_size": len(tracks),
        "degraded_fallback_clusters": degraded_fallback_counter[0],
        "n_clusters_processed": len(cluster_sizes),
        "mean_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "max_cluster_size": int(np.max(cluster_sizes)) if cluster_sizes else 0,
        "total_associations_considered": total_associations_considered,
        "lookup_us_mean": float(np.mean(lookup_times)) if lookup_times else 0.0,
        "lookup_us_p50": float(np.percentile(lookup_times, 50)) if lookup_times else 0.0,
        "lookup_us_p95": float(np.percentile(lookup_times, 95)) if lookup_times else 0.0,
        "lookup_us_p99": float(np.percentile(lookup_times, 99)) if lookup_times else 0.0,
        "lookup_us_max": float(np.max(lookup_times)) if lookup_times else 0.0,
        "elapsed_s": real_elapsed,
        "shell_breakdown": {
            "shell_total_detections": shell_total,
            "shell_known_count": shell_known,
            "shell_correct_matches": shell_correct,
            "shell_mismatches": shell_mismatch,
            "shell_correct_matches_pct": 100.0 * shell_correct / (shell_correct + shell_mismatch) if (shell_correct + shell_mismatch) else 0.0,
            "shell_mismatches_pct": 100.0 * shell_mismatch / (shell_correct + shell_mismatch) if (shell_correct + shell_mismatch) else 0.0,
            "shell_mean_cluster_size": float(np.mean(shell_cluster_sizes)) if shell_cluster_sizes else 0.0,
            "shell_max_cluster_size": int(np.max(shell_cluster_sizes)) if shell_cluster_sizes else 0,
            "shell_lookup_us_mean": float(np.mean(shell_lookup_times)) if shell_lookup_times else 0.0,
            "shell_lookup_us_p50": float(np.percentile(shell_lookup_times, 50)) if shell_lookup_times else 0.0,
            "shell_lookup_us_p95": float(np.percentile(shell_lookup_times, 95)) if shell_lookup_times else 0.0,
            "shell_lookup_us_p99": float(np.percentile(shell_lookup_times, 99)) if shell_lookup_times else 0.0,
            "shell_lookup_us_max": float(np.max(shell_lookup_times)) if shell_lookup_times else 0.0,
        },
    }

    print("-" * 70)
    print(f"JPDA baseline complete.")
    print(f"Total detections     : {total_detections}")
    print(f"KNOWN                 : {known_count} ({100*known_count/total_detections:.1f}%)")
    print(f"NOT KNOWN (novel)     : {unknown_count} ({100*unknown_count/total_detections:.1f}%)")
    print(f"Real wall time        : {real_elapsed:.2f}s")
    print(f"Update commits        : {update_committed_count}  withheld: {update_withheld_count}")
    if total_accepted:
        print(f"Correct matches       : {correct_match_count} ({correct_matches_pct:.1f}%)")
        print(f"Mismatches            : {mismatch_count} ({mismatches_pct:.1f}%)")
    print(f"Catalog completeness   : {catalog_completeness_pct:.1f}% ({catalog_complete_n}/{total_true_population})")
    print(f"Final catalog size    : {len(tracks)}")
    print(f"Degraded-fallback clusters (exceeded {JPDA_MAX_CLUSTER_ENUM}-way exact enum): "
          f"{degraded_fallback_counter[0]} / {len(cluster_sizes)} clusters")
    print(f"Mean/max cluster size  : {results['mean_cluster_size']:.2f} / {results['max_cluster_size']}")
    print(f"Lookup latency (us)    : mean={results['lookup_us_mean']:.2f} "
          f"p50={results['lookup_us_p50']:.2f} p95={results['lookup_us_p95']:.2f} "
          f"p99={results['lookup_us_p99']:.2f} max={results['lookup_us_max']:.2f}")
    if shell_total:
        sb = results["shell_breakdown"]
        print("-" * 70)
        print(f"SHELL-ONLY BREAKDOWN ({shell_total} detections from '_shell' catalog objects)")
        print(f"Shell correct matches  : {sb['shell_correct_matches']} ({sb['shell_correct_matches_pct']:.1f}%)")
        print(f"Shell mismatches       : {sb['shell_mismatches']} ({sb['shell_mismatches_pct']:.1f}%)")
        print(f"Shell mean/max cluster : {sb['shell_mean_cluster_size']:.2f} / {sb['shell_max_cluster_size']}")
        print(f"Shell lookup (us)      : mean={sb['shell_lookup_us_mean']:.2f} "
              f"p50={sb['shell_lookup_us_p50']:.2f} p95={sb['shell_lookup_us_p95']:.2f} "
              f"p99={sb['shell_lookup_us_p99']:.2f} max={sb['shell_lookup_us_max']:.2f}")
        print("(compare these numbers to the whole-catalog numbers above -- this is the "
              "signal that matters for a density-regime test; the whole-catalog average "
              "is diluted by the sparse background population)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the JPDA baseline tracker.")
    parser.add_argument("--catalog", type=str, default=CATALOG_PATH)
    parser.add_argument("--duration", type=float, default=SIMULATION_DURATION_SECONDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=None,
                         help="Stop early after this many real seconds and write "
                              "whatever partial results exist (results['partial']=True). "
                              "Useful on constrained/sandboxed environments; not needed "
                              "on a normal machine for the default 20k/60s config.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to write a periodically-overwritten progress "
                              "snapshot to, so a long run can be inspected or safely "
                              "killed without losing all progress.")
    parser.add_argument("--checkpoint-every", type=int, default=10000,
                         help="Write the checkpoint file every N detections.")
    args = parser.parse_args()

    results = run_jpda(catalog_path=args.catalog, duration=args.duration,
                        seed=args.seed, verbose=not args.quiet,
                        max_wall_seconds=args.max_wall_seconds,
                        checkpoint_path=args.checkpoint,
                        checkpoint_every=args.checkpoint_every)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    if results.get("partial"):
        print(f"\nWrote PARTIAL JPDA results (wall-clock budget hit at sim T+"
              f"{results['sim_time_reached_s']:.2f}s of {args.duration}s) to {args.output}")
    else:
        print(f"\nWrote JPDA results to {args.output}")


if __name__ == "__main__":
    main()
