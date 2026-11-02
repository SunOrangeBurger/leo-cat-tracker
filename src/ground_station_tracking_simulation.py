#!/usr/bin/env python3
"""
Ground station tracking simulation -- the full predict/match/update cycle.

Successor to the previous ground_station_tracking_simulation.py (README.md
section 5, item 4). Everything about the PERFORMANCE layer (ANN index,
periodic rebuild/recalibration), the SAFETY layer (corroboration gate
before any belief-update commits, retirement, de-duplication), and the
fast distance GATE is unchanged from the previous design -- those parts
work (README sections 2.2, 2.3, 4.5) and are kept as-is. What changed is
what happens when the fast gate REJECTS a detection: instead of a binary
GNN second opinion forced to guess KNOWN or NOT-KNOWN even under genuine
uncertainty, catalog_gnn.py's classify_three_way() returns one of:

  - KNOWN     -> proceed exactly as a gate-accept would: corroboration
                 gate, then EMA belief update.
  - NOVEL     -> register a new catalog entry, exactly as before.
  - AMBIGUOUS -> purely defer. No belief write, no registration, no
                 catalog mutation of any kind. Logged as "not known" for
                 this pass; the next detection of the same physical object
                 re-enters the full pipeline fresh, with no memory of this
                 deferral (README section 4.1).

Every object in leo_catalog.csv has a REAL trajectory that evolves over time
(drag decay + J2 precession + small unmodeled process noise). The ground
station only has embeddings for a fraction of the catalog (EMBED_FRACTION at
identity_embedding.py build time). On each orbital pass:

  1. PREDICT : propagate every catalog object's stored belief forward from
     its own last-updated epoch to the current time (clean, deterministic
     physics -- see orbital_physics.propagate_elements).
  2. DETECT  : the true object's real trajectory is propagated forward too
     (same physics, plus small process + measurement noise -- this is what
     the sensor "sees").
  3. MATCH   : find the 5 nearest catalog identity-vectors to the detection
     via the ANN candidate pool (leo_ann_index.py), then re-rank exactly.
     Accept the closest one only if it clears a drift-tolerance distance
     AND is meaningfully closer than the 2nd-nearest (ambiguity margin).
     If the gate rejects a plausible near-miss, consult the catalog GNN's
     three-way second opinion on the same top-5.
  4. UPDATE  : KNOWN detections blend into the matched catalog entry's
     stored belief via the same two-pass corroboration gate as before
     (a single accept from any path is provisional; the write only commits
     once two consecutive passes agree on the same slot). NOVEL detections
     register a new entry. AMBIGUOUS detections mutate nothing.

All orbital motion is hyper-accelerated (ACCELERATION_FACTOR); all lookup/
match timing is measured in genuine, unaccelerated wall-clock time.
"""

import heapq
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

import orbital_physics as op
import identity_embedding as embed_mod
import catalog_gnn as gnn_mod
import leo_ann_index
from sklearn.neighbors import NearestNeighbors

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------
# Config
#
# Anchored to repo root (see identity_embedding.py's comment on this
# convention) so this script runs identically from repo root or from
# inside src/.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = str(REPO_ROOT / "data" / "leo_catalog.csv")
EMBEDDINGS_PATH = str(REPO_ROOT / "outputs" / "leo_embeddings.pt")
FALLBACK_NPZ_PATH = str(REPO_ROOT / "outputs" / "leo_embeddings.npz")
# Matches compare_live_vs_training.py's own LIVE_LOG_PATH constant --
# defined here too (rather than only relied on as a bare string at the
# write site below) so both scripts agree on the same anchored location
# regardless of invocation cwd.
LIVE_LOG_PATH = os.environ.get("LEO_LIVE_LOG_PATH", str(REPO_ROOT / "outputs" / "gnn_live_invocations.json"))

ACCELERATION_FACTOR = 300          # sim-seconds run this much faster than real orbital seconds
SIMULATION_DURATION_SECONDS = float(os.environ.get("LEO_DURATION_SECONDS", "60"))

# Matching gate -- set to None to auto-calibrate from the catalog's own
# measured nearest-neighbor density at startup (recommended; see
# calibrate_thresholds() below). Fixed numbers tuned for a small catalog
# become unsafe at larger scale: denser catalogs pack objects closer
# together, so a tolerance that was safe at 100 objects can exceed the
# actual gap between two DIFFERENT objects at 1000+.
DRIFT_TOLERANCE = None
MARGIN_REQUIRED = None

# Catalog belief update rate (exponential moving average)
ALPHA = 0.4

# Realism noise: real trajectory gets small process noise (unmodeled physics);
# the "sensor" also adds a bit of measurement noise on top before matching.
PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
MEASUREMENT_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}

SEED = int(os.environ.get("LEO_SEED", "11"))
ORACLE_FREE = os.environ.get("LEO_ORACLE_FREE", "0") == "1"
ABLATION_MODE = os.environ.get("LEO_ABLATION_MODE", "full")
if ABLATION_MODE not in {"full", "novel", "nearest"}:
    raise ValueError("LEO_ABLATION_MODE must be full, novel, or nearest")

# Second-opinion catalog GNN: when the fast threshold gate rejects a
# plausible near-miss, consult catalog_gnn.py's three-way classifier
# before giving up (README section 4.1).
USE_GNN_CONFIRMATION = ABLATION_MODE == "full"

# Only consult the GNN when the gate-rejected candidate pool is plausibly a
# near-miss (nearest candidate within this multiple of drift_tolerance),
# not on every gate-reject. This is unchanged from the previous design's
# fix (README section 3.2's second problem): the overwhelming majority of
# gate-rejects are genuine first-sightings where the true object isn't in
# the candidate pool at all, and asking any classifier to adjudicate that
# is an unanswerable question. The three-way output contract makes this
# gate less load-bearing than it used to be (a confidently-wrong guess is
# now AMBIGUOUS-able rather than forced), but it's kept anyway: there's no
# reason to spend a GNN call, or the small residual chance of a confident
# NOVEL-vs-KNOWN error, on candidate pools that are structurally
# uninformative.
GNN_PLAUSIBILITY_MULTIPLE = 3.0

# Every acceptance path (gate, GNN-KNOWN, existing-slot correction) is
# uncertain to some degree. Require this many CONSECUTIVE accepted
# detections of the SAME true object landing on the SAME catalog slot
# before committing an EMA belief-update to it, instead of trusting any
# single accept enough to overwrite stored state.
# 1 = every accept writes immediately (unsafe). 2 = require one repeat.
UPDATE_CONFIRMATIONS_REQUIRED = 2

# --- ANN index tuning (see leo_ann_index.py for full rationale) ---
ANN_REBUILD_EVERY = leo_ann_index.REBUILD_EVERY
ANN_OVERFLOW_REBUILD_THRESHOLD = leo_ann_index.OVERFLOW_REBUILD_THRESHOLD
ANN_CANDIDATE_POOL_SIZE = leo_ann_index.CANDIDATE_POOL_SIZE
ANN_MIN_CATALOG_SIZE = ANN_CANDIDATE_POOL_SIZE

# --- Retirement / de-duplication / recalibration ---
RETIREMENT_ALTITUDE_FLOOR_KM = 100.0
RETIREMENT_UNSEEN_SECONDS = 48 * 3600.0
RETIREMENT_WARMUP_SECONDS = 12 * 3600.0
RETIREMENT_CHECK_EVERY = 100
DEDUP_CHECK_EVERY = 500
DEDUP_DISTANCE_FRACTION_OF_DRIFT_TOLERANCE = 0.2
RECALIBRATION_MIN_ACTIVE = 200
RECALIBRATION_CHANGE_FRACTION = 0.10

# Local-density feature radius for the catalog GNN (README 4.2c), as a
# multiple of drift_tolerance -- must match what train_catalog_gnn.py's
# harvester used (catalog_gnn.DENSITY_RADIUS_MULTIPLE) or the feature's
# scale at inference time won't match what the model was trained on.
DENSITY_RADIUS_MULTIPLE = gnn_mod.DENSITY_RADIUS_MULTIPLE

# --- Density-aware early ANN rebuild (see docs/ANN_DENSITY_AWARE_REBUILD.md
# for the full mechanism and how these values were chosen) ---
# ANN_REBUILD_EVERY's staleness tolerance is calibrated against the
# catalog's AVERAGE nearest-neighbor spacing (via DRIFT_TOLERANCE). A
# locally dense pocket (e.g. a packed constellation shell) can have real
# spacing far tighter than that average, so ordinary between-rebuild drift
# that's safe everywhere else can push the true match out of the ANN
# candidate pool entirely in that pocket. This block adds a second,
# density-triggered rebuild check (leo_ann_index.needs_rebuild_for_query)
# that fires independently of the detection-count cadence.
ANN_DENSITY_AWARE_REBUILD_ENABLED = True

# --- Radius the density-rebuild check uses (see docs/ANN_DENSITY_AWARE_REBUILD.md,
# "Second known issue: radius mismatch"). Deliberately a SEPARATE constant
# from DENSITY_RADIUS_MULTIPLE (used for the GNN's local_density training
# feature) rather than reusing it -- reusing it was the actual mistake in
# the first version of this fix. Measured directly from this catalog's own
# live GNN-consult distances (gnn_live_invocations.json's min_dist field,
# same weighted embedding space): p10=0.221, median=0.315, out of a
# DENSITY_RADIUS_MULTIPLE*drift_tolerance radius of only ~0.269 -- i.e. the
# GNN-feature radius is tighter than the TYPICAL distance between a
# detection and its closest real candidate, so almost nothing ever counted
# as "nearby" by that radius, in the sparse background OR the dense shell.
# ANN_DENSITY_REBUILD_RADIUS_MULTIPLE is calibrated instead directly off
# drift_tolerance at a wider multiple, chosen to comfortably cover the
# real observed candidate-distance range above (median 0.315) rather than
# undershoot it. Re-tune this against your own catalog's real
# gnn_live_invocations.json min_dist distribution if you regenerate the
# dense shell with different width/density settings -- don't assume this
# value transfers.
ANN_DENSITY_REBUILD_RADIUS_MULTIPLE = 3.5

# How many OTHER catalog objects within
# ANN_DENSITY_REBUILD_RADIUS_MULTIPLE * drift_tolerance of a query counts
# as "dense enough to be at risk". NOTE: this default is a reasoned
# starting point, not yet empirically tuned against a labeled dense-shell
# run with the corrected radius active -- see
# docs/ANN_DENSITY_AWARE_REBUILD.md "Tuning ANN_DENSITY_REBUILD_THRESHOLD"
# for how to calibrate it once you have real before/after miss-rate data
# at this new radius. Lower = more conservative (more early rebuilds,
# safer, more overhead); higher = closer to the old global-only behavior
# this option exists to improve on.
ANN_DENSITY_REBUILD_THRESHOLD = 6

# Don't fire the density trigger on a tree that was just rebuilt --
# staleness hasn't had time to accumulate yet, so an early rebuild here
# would only add overhead without fixing anything. Kept small (not 0)
# because SOME drift is possible even one detection after a rebuild (the
# catalog can grow/update between the rebuild and this query), but large
# enough to avoid rebuild-thrashing on back-to-back dense-region queries
# immediately after a rebuild already just ran.
ANN_DENSITY_REBUILD_MIN_DETECTIONS = 5


# ---------------------------------------------------------------------
# Load embedding store
# ---------------------------------------------------------------------
def load_embedding_store(pt_path, npz_fallback):
    if TORCH_AVAILABLE:
        try:
            data = torch.load(pt_path, weights_only=False)
            data["vectors"] = data["vectors"].numpy()
            return data
        except FileNotFoundError:
            pass
    raw = np.load(npz_fallback, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    data["scalers"] = raw["scalers"][0]
    data["names"] = data["names"].tolist()
    data["types"] = data["types"].tolist()
    return data


def calibrate_thresholds(catalog_vectors: np.ndarray, cat_i=None, cat_types=None, cat_e=None):
    """
    Auto-calibrate DRIFT_TOLERANCE and MARGIN_REQUIRED from the catalog's
    OWN measured density, instead of trusting fixed numbers tuned at a
    different scale.

    Measures the gap WITHIN clusters that actually collide (bucketed by
    type / coarse inclination / eccentricity regime), not a flat global
    percentile -- data_generator.py draws inclination from only 5 narrow
    bands and eccentricity from a narrow near-circular band for ~90% of
    objects, so most real collisions happen between two objects that
    already share a bucket. A global 5th percentile is dominated by
    well-separated cross-cluster pairs and understates the real worst case.

    Uses scipy.spatial.cKDTree per bucket (exact nearest-neighbor, not an
    approximation) so this stays cheap enough to call periodically as the
    active catalog's density changes over a run.
    """
    from scipy.spatial import cKDTree

    if cat_i is None or cat_types is None or cat_e is None:
        tree = cKDTree(catalog_vectors)
        dists, _ = tree.query(catalog_vectors, k=min(2, len(catalog_vectors)))
        nearest = dists[:, 1] if dists.ndim > 1 and dists.shape[1] > 1 else np.zeros(len(catalog_vectors))
        p5 = np.percentile(nearest, 5)
        drift_tolerance = 0.35 * p5
        margin_required = 0.15 * p5
        print(f"Auto-calibrated (global, no bucket metadata): nearest-neighbor gap "
              f"(5th pct)={p5:.4f} -> DRIFT_TOLERANCE={drift_tolerance:.4f}, "
              f"MARGIN_REQUIRED={margin_required:.4f}")
        return drift_tolerance, margin_required

    cat_i = np.asarray(cat_i, dtype=float)
    cat_types = np.asarray(cat_types)
    cat_e = np.asarray(cat_e, dtype=float)

    incl_bucket = np.round(cat_i / 5.0).astype(int)
    ecc_bucket = (cat_e > 0.01).astype(int)

    bucket_keys = list(zip(cat_types.tolist(), incl_bucket.tolist(), ecc_bucket.tolist()))
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, key in enumerate(bucket_keys):
        groups[key].append(idx)

    within_bucket_gaps = []
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        sub = catalog_vectors[idxs]
        tree = cKDTree(sub)
        dists, _ = tree.query(sub, k=2)
        within_bucket_gaps.extend(dists[:, 1].tolist())

    if not within_bucket_gaps:
        tree = cKDTree(catalog_vectors)
        dists, _ = tree.query(catalog_vectors, k=min(2, len(catalog_vectors)))
        within_bucket_gaps = (dists[:, 1] if dists.ndim > 1 and dists.shape[1] > 1
                               else np.zeros(len(catalog_vectors))).tolist()

    p5 = np.percentile(within_bucket_gaps, 5)
    drift_tolerance = 0.35 * p5
    margin_required = 0.15 * p5

    print(f"Auto-calibrated from WITHIN-CLUSTER density ({len(groups)} buckets, "
          f"{len(within_bucket_gaps)} within-bucket pairs): nearest-neighbor gap "
          f"(5th pct)={p5:.4f} -> DRIFT_TOLERANCE={drift_tolerance:.4f}, "
          f"MARGIN_REQUIRED={margin_required:.4f}")
    return drift_tolerance, margin_required


def main():
    catalog_df = pd.read_csv(CATALOG_PATH)
    store = load_embedding_store(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    scalers = store["scalers"]

    distance_weights = np.asarray(
        store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32
    )
    weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights

    drift_tolerance = DRIFT_TOLERANCE
    margin_required = MARGIN_REQUIRED
    if drift_tolerance is None or margin_required is None:
        drift_tolerance, margin_required = calibrate_thresholds(
            weighted_store_vectors,
            cat_i=store["elements_i"],
            cat_types=store["types"],
            cat_e=store["elements_e"],
        )

    print(f"Catalog:   {len(catalog_df)} objects ({CATALOG_PATH})")
    print(f"Embedding: {len(store['names'])} objects on record")
    print(f"Acceleration factor: {ACCELERATION_FACTOR}x real orbital speed")
    print(f"Simulated run length: {SIMULATION_DURATION_SECONDS}s (simulated)")
    print(f"Update corroboration required: {UPDATE_CONFIRMATIONS_REQUIRED} consecutive "
          f"matching accepts on the same slot before a belief-update commits")
    print(f"Ablation mode: {ABLATION_MODE}; oracle-free decisions: {ORACLE_FREE}")

    # GNN training happens OFFLINE via train_catalog_gnn.py, not inline
    # here -- see that file's docstring for why. Load-only here: if no
    # checkpoint exists, or its gate_stamp doesn't match this run's
    # settings, GNN confirmation is disabled for the run.
    gnn_model = None
    confident_known_threshold = None
    confident_novel_threshold = None
    if USE_GNN_CONFIRMATION:
        if not gnn_mod.TORCH_AVAILABLE:
            print("GNN confirmation requested but PyTorch isn't available -- skipping "
                  "(gate-rejected detections will just stay NOT KNOWN, no catalog growth "
                  "via that path).")
        else:
            try:
                ckpt = torch.load(gnn_mod.MODEL_PATH, map_location="cpu", weights_only=False)
            except FileNotFoundError:
                ckpt = None

            if ckpt is None:
                print(f"No catalog GNN checkpoint found at {gnn_mod.MODEL_PATH} -- GNN "
                      f"confirmation disabled for this run. Run train_catalog_gnn.py first.")
            else:
                saved_stamp = ckpt.get("gate_stamp", {})
                saved_dt = saved_stamp.get("drift_tolerance")
                saved_mr = saved_stamp.get("margin_required")
                saved_w = saved_stamp.get("distance_weights")
                this_w = np.asarray(distance_weights, dtype=np.float32).tolist()
                mismatch = (
                    saved_dt is None or saved_mr is None or saved_w is None
                    or not np.isclose(saved_dt, float(drift_tolerance), rtol=1e-4)
                    or not np.isclose(saved_mr, float(margin_required), rtol=1e-4)
                    or not np.allclose(saved_w, this_w, rtol=1e-4)
                )
                if mismatch:
                    print(f"Catalog GNN checkpoint at {gnn_mod.MODEL_PATH} was trained under "
                          f"different gate thresholds/weights than this run's -- disabled "
                          f"rather than silently retraining. Re-run train_catalog_gnn.py.")
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    gnn_model = gnn_mod.CatalogGNN().to(device)
                    gnn_model.load_state_dict(ckpt["state_dict"])
                    gnn_model.eval()
                    confident_known_threshold = ckpt["confident_known_threshold"]
                    confident_novel_threshold = ckpt["confident_novel_threshold"]
                    print(f"Loaded catalog GNN from {gnn_mod.MODEL_PATH} (gate settings match). "
                          f"confident_known={confident_known_threshold:.4f} "
                          f"confident_novel={confident_novel_threshold:.4f}")
    print("-" * 70)

    rng = np.random.default_rng(SEED)

    # --- TRUTH state for every object in the full catalog ---
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

    # --- CATALOG belief state (only the embedded subset, GROWABLE) ---
    cat_names = list(store["names"])
    cat_a = list(np.asarray(store["elements_a"], dtype=float))
    cat_e = list(np.asarray(store["elements_e"], dtype=float))
    cat_i = list(np.asarray(store["elements_i"], dtype=float))
    cat_raan = list(np.asarray(store["elements_raan"], dtype=float))
    cat_argp = list(np.asarray(store["elements_argp"], dtype=float))
    cat_bc = list(np.asarray(store["bc_m2_per_kg"], dtype=float))
    cat_types = list(store["types"])
    cat_sizes = list(np.asarray(store["sizes"], dtype=float))
    cat_rcs = list(np.asarray(store["rcs"], dtype=float))
    cat_epoch = list(np.asarray(store["epoch_seconds"], dtype=float))
    discovery_counter = 0
    discovered_true_name = {}

    n_preloaded = len(cat_names)
    cat_retired = [False] * n_preloaded
    cat_confidence = [1] * n_preloaded          # corroboration-history feature source (README 4.2a)
    cat_last_seen = list(cat_epoch)             # also feeds "time_since_last_update" feature

    ann_index = leo_ann_index.LiveCatalogANNIndex(
        scalers, rebuild_every=ANN_REBUILD_EVERY,
        overflow_threshold=ANN_OVERFLOW_REBUILD_THRESHOLD,
        candidate_pool_size=ANN_CANDIDATE_POOL_SIZE,
    )

    pending_update = {}  # cat_idx -> true_name of the pending (unconfirmed) corroboration guess
    last_calibrated_active_count = None

    true_name_to_idx = {n: i for i, n in enumerate(cat_names)}  # ground-truth bookkeeping only

    real_periods = 2 * np.pi * np.sqrt(truth_a ** 3 / op.MU_EARTH)
    sim_periods = real_periods / ACCELERATION_FACTOR

    heap = []
    for idx in range(len(catalog_df)):
        phase = rng.uniform(0, sim_periods[idx])
        heapq.heappush(heap, (phase, idx))

    # --- metrics ---
    total_detections = 0
    known_count = 0
    unknown_count = 0
    ambiguous_count = 0
    lookup_times_known = []
    lookup_times_unknown = []
    match_distances = []
    match_margins = []
    per_type_known = {}
    per_type_unknown = {}
    gnn_confirmed_known = 0
    gnn_confirmed_novel = 0
    gnn_ambiguous_deferred = 0
    update_committed_count = 0
    update_withheld_count = 0
    truth_confirm_checks = 0
    existing_slot_corrections = 0
    retired_decayed_count = 0
    retired_unseen_count = 0
    merged_count_total = 0

    mismatch_count = 0
    correct_match_count = 0

    true_name_to_all_slots = {}
    for i in range(n_preloaded):
        true_name_to_all_slots.setdefault(cat_names[i], set()).add(i)
    gnn_overturn_total = 0
    gnn_overturn_correct = 0
    gnn_skipped_implausible = 0
    ann_recall_checked = 0
    ann_recall_hit = 0
    gnn_invocation_log = []  # for compare_live_vs_training.py (README section 5.5)
    density_triggered_rebuild_count = [0]  # see ANN_DENSITY_AWARE_REBUILD_ENABLED above
    density_diagnostic_log = []  # DIAGNOSTIC ONLY -- see comment near query_density() call
    total_detections_for_density_diag = [0]

    real_start = time.perf_counter()

    def resolve_update(cat_idx, true_object_name, meas_a, meas_e, meas_i, meas_raan, meas_argp,
                        prop_a_val, prop_e_val, prop_i_val, prop_raan_val, prop_argp_val, real_now):
        """Single choke point for EVERY catalog belief-update, regardless
        of which path (gate / GNN-KNOWN / existing-slot correction)
        accepted the match. Returns True if the write committed, False if
        withheld pending corroboration (README 4.5 -- kept unchanged)."""
        if UPDATE_CONFIRMATIONS_REQUIRED <= 1:
            cat_a[cat_idx] = (1 - ALPHA) * prop_a_val + ALPHA * meas_a
            cat_e[cat_idx] = (1 - ALPHA) * prop_e_val + ALPHA * meas_e
            cat_i[cat_idx] = (1 - ALPHA) * prop_i_val + ALPHA * meas_i
            cat_raan[cat_idx] = (1 - ALPHA) * prop_raan_val + ALPHA * meas_raan
            cat_argp[cat_idx] = (1 - ALPHA) * prop_argp_val + ALPHA * meas_argp
            cat_epoch[cat_idx] = real_now
            return True

        nonlocal truth_confirm_checks
        prior = pending_update.get(cat_idx)
        if ORACLE_FREE:
            now_vec = embed_mod.transform_identity_vector(
                meas_a, meas_e, meas_i, meas_raan, meas_argp,
                cat_types[cat_idx], cat_sizes[cat_idx], cat_rcs[cat_idx], scalers)
            if prior is not None:
                old_elements, old_epoch = prior
                pa, pe, pi, po, pw = op.propagate_catalog_analytic(
                    np.array([old_elements[0]]), np.array([old_elements[1]]),
                    np.array([old_elements[2]]), np.array([old_elements[3]]),
                    np.array([old_elements[4]]), np.array([cat_bc[cat_idx]]),
                    real_now - old_epoch)
                old_vec = embed_mod.transform_identity_vector(
                    pa[0], pe[0], pi[0], po[0], pw[0],
                    cat_types[cat_idx], cat_sizes[cat_idx], cat_rcs[cat_idx], scalers)
                corroborated = np.linalg.norm((old_vec - now_vec) * distance_weights) <= 3 * drift_tolerance
            else:
                corroborated = False
        else:
            truth_confirm_checks += 1
            corroborated = prior == true_object_name
        if corroborated:
            cat_a[cat_idx] = (1 - ALPHA) * prop_a_val + ALPHA * meas_a
            cat_e[cat_idx] = (1 - ALPHA) * prop_e_val + ALPHA * meas_e
            cat_i[cat_idx] = (1 - ALPHA) * prop_i_val + ALPHA * meas_i
            cat_raan[cat_idx] = (1 - ALPHA) * prop_raan_val + ALPHA * meas_raan
            cat_argp[cat_idx] = (1 - ALPHA) * prop_argp_val + ALPHA * meas_argp
            cat_epoch[cat_idx] = real_now
            pending_update.pop(cat_idx, None)
            return True
        else:
            pending_update[cat_idx] = ((meas_a, meas_e, meas_i, meas_raan, meas_argp), real_now) if ORACLE_FREE else true_object_name
            return False

    while heap:
        sim_time, idx = heapq.heappop(heap)
        if sim_time > SIMULATION_DURATION_SECONDS:
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

        detection_vec = embed_mod.transform_identity_vector(
            meas_a, meas_e, meas_i, meas_raan, meas_argp, types[idx], sizes[idx], rcs_vals[idx], scalers
        )

        name = names[idx]  # true identity -- ground truth, used only for corroboration-gate
                            # plumbing (existing_slot path) and the ground-truth-only diagnostics

        t0 = time.perf_counter()

        n_cat = len(cat_names)
        candidate_idx = np.array([], dtype=int)
        if n_cat > 0:
            weighted_detection_vec = detection_vec * distance_weights

            # Two independent rebuild triggers, OR'd together:
            #  1. needs_rebuild() -- the existing global cadence (detection
            #     count / overflow size), calibrated for average catalog
            #     density (see leo_ann_index.py module docstring).
            #  2. needs_rebuild_for_query() -- density-aware: is THIS
            #     detection sitting in a locally dense pocket (e.g. a
            #     packed constellation shell) where the global cadence's
            #     staleness assumption doesn't hold, even though the global
            #     trigger hasn't fired yet? See
            #     docs/ANN_DENSITY_AWARE_REBUILD.md for the mechanism this
            #     closes and why the global-only check was letting the true
            #     match silently drop out of the candidate pool in dense
            #     regions.
            # Evaluated in this order so the (already-necessary) global
            # check short-circuits the extra radius_neighbors() call on
            # the common "tree doesn't exist yet" / "just rebuilt" paths.
            global_rebuild_due = ann_index.needs_rebuild()
            density_rebuild_due = (
                not global_rebuild_due
                and ANN_DENSITY_AWARE_REBUILD_ENABLED
                and ann_index.needs_rebuild_for_query(
                    weighted_detection_vec,
                    density_radius=ANN_DENSITY_REBUILD_RADIUS_MULTIPLE * drift_tolerance,
                    density_threshold=ANN_DENSITY_REBUILD_THRESHOLD,
                    min_detections_since_rebuild=ANN_DENSITY_REBUILD_MIN_DETECTIONS,
                )
            )

            # DIAGNOSTIC ONLY (see docs/ANN_DENSITY_AWARE_REBUILD.md,
            # "Tuning ANN_DENSITY_REBUILD_THRESHOLD"): sample the raw
            # density value on a subset of detections so the threshold can
            # be calibrated against real numbers instead of guessed at
            # again. No effect on rebuild behavior -- query_density() is
            # never consulted by any decision. Sampled (not every
            # detection) to keep the extra radius_neighbors() calls cheap
            # over a 60k+ detection run.
            if not global_rebuild_due and total_detections_for_density_diag[0] % 20 == 0:
                _raw_density = ann_index.query_density(
                    weighted_detection_vec, ANN_DENSITY_REBUILD_RADIUS_MULTIPLE * drift_tolerance,
                )
                density_diagnostic_log.append({
                    "sim_time": float(real_now),
                    "name": name,
                    "raw_density": _raw_density,
                })
            total_detections_for_density_diag[0] += 1

            if global_rebuild_due or density_rebuild_due:
                if density_rebuild_due:
                    density_triggered_rebuild_count[0] += 1
                ann_index.rebuild(
                    cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                    cat_types, cat_sizes, cat_rcs, cat_epoch, real_now,
                    distance_weights, cat_retired,
                )

                active_mask_recal = ~np.asarray(cat_retired, dtype=bool)
                n_active_recal = int(np.sum(active_mask_recal))
                needs_recalibration = (
                    n_active_recal >= RECALIBRATION_MIN_ACTIVE
                    and (
                        last_calibrated_active_count is None
                        or abs(n_active_recal - last_calibrated_active_count)
                           >= RECALIBRATION_CHANGE_FRACTION * last_calibrated_active_count
                    )
                )
                if needs_recalibration:
                    act_idx = np.where(active_mask_recal)[0]
                    act_a = np.array([cat_a[j] for j in act_idx])
                    act_e = np.array([cat_e[j] for j in act_idx])
                    act_i = np.array([cat_i[j] for j in act_idx])
                    act_raan = np.array([cat_raan[j] for j in act_idx])
                    act_argp = np.array([cat_argp[j] for j in act_idx])
                    act_bc = np.array([cat_bc[j] for j in act_idx])
                    act_epoch = np.array([cat_epoch[j] for j in act_idx])
                    act_types = [cat_types[j] for j in act_idx]
                    act_sizes = np.array([cat_sizes[j] for j in act_idx])
                    act_rcs = np.array([cat_rcs[j] for j in act_idx])

                    dt_act = real_now - act_epoch
                    p_act_a, p_act_e, p_act_i, p_act_raan, p_act_argp = op.propagate_catalog_analytic(
                        act_a, act_e, act_i, act_raan, act_argp, act_bc, dt_act
                    )
                    act_vectors = embed_mod.build_embeddings_fast(
                        p_act_a, p_act_e, p_act_i, p_act_raan, p_act_argp,
                        act_types, act_sizes, act_rcs, scalers,
                    )
                    weighted_act_vectors = act_vectors * distance_weights
                    new_drift_tolerance, new_margin_required = calibrate_thresholds(
                        weighted_act_vectors, cat_i=p_act_i, cat_types=act_types, cat_e=p_act_e,
                    )
                    drift_tolerance, margin_required = new_drift_tolerance, new_margin_required
                    last_calibrated_active_count = n_active_recal
                    print(f"[T+{sim_time:7.2f}s sim] [RECALIBRATE] active={n_active_recal} "
                          f"drift_tolerance={drift_tolerance:.4f} margin_required={margin_required:.4f}")

            candidate_idx = ann_index.query_candidates(weighted_detection_vec, n_cat, cat_retired)

            _gt_idx_for_recall = true_name_to_idx.get(name)
            if (_gt_idx_for_recall is not None and _gt_idx_for_recall < n_cat
                    and not cat_retired[_gt_idx_for_recall]):
                ann_recall_checked += 1
                if _gt_idx_for_recall in candidate_idx:
                    ann_recall_hit += 1

        if len(candidate_idx) > 0:
            cand_a = np.array([cat_a[j] for j in candidate_idx])
            cand_e = np.array([cat_e[j] for j in candidate_idx])
            cand_i = np.array([cat_i[j] for j in candidate_idx])
            cand_raan = np.array([cat_raan[j] for j in candidate_idx])
            cand_argp = np.array([cat_argp[j] for j in candidate_idx])
            cand_bc = np.array([cat_bc[j] for j in candidate_idx])
            cand_epoch = np.array([cat_epoch[j] for j in candidate_idx])
            cand_types = [cat_types[j] for j in candidate_idx]
            cand_sizes = np.array([cat_sizes[j] for j in candidate_idx])
            cand_rcs = np.array([cat_rcs[j] for j in candidate_idx])

            dt_cand = real_now - cand_epoch
            cprop_a, cprop_e, cprop_i, cprop_raan, cprop_argp = op.propagate_catalog_analytic(
                cand_a, cand_e, cand_i, cand_raan, cand_argp, cand_bc, dt_cand
            )
            cand_vectors = embed_mod.build_embeddings_fast(
                cprop_a, cprop_e, cprop_i, cprop_raan, cprop_argp, cand_types, cand_sizes, cand_rcs, scalers
            )

            local_of = {int(j): pos for pos, j in enumerate(candidate_idx)}

            def prop_a(j): return cprop_a[local_of[j]]
            def prop_e(j): return cprop_e[local_of[j]]
            def prop_i(j): return cprop_i[local_of[j]]
            def prop_raan(j): return cprop_raan[local_of[j]]
            def prop_argp(j): return cprop_argp[local_of[j]]

            def cat_vector_for(j):
                """Identity vector for absolute catalog index j, exact
                current-epoch, whether or not j is in this pass's ANN
                candidate pool (existing_slot correction can point
                anywhere in the catalog)."""
                if j in local_of:
                    return cand_vectors[local_of[j]]
                dt_j = real_now - cat_epoch[j]
                pa, pe, pi, praan, pargp = op.propagate_catalog_analytic(
                    np.array([cat_a[j]]), np.array([cat_e[j]]), np.array([cat_i[j]]),
                    np.array([cat_raan[j]]), np.array([cat_argp[j]]), np.array([cat_bc[j]]), dt_j,
                )
                return embed_mod.build_embeddings_fast(
                    pa, pe, pi, praan, pargp, [cat_types[j]], [cat_sizes[j]], [cat_rcs[j]], scalers
                )[0]

            def unweighted_dist_for(j):
                return float(np.linalg.norm(cat_vector_for(j) - detection_vec))

            weighted_cand_vectors = cand_vectors * distance_weights
            cand_weighted_dists = np.linalg.norm(weighted_cand_vectors - weighted_detection_vec, axis=1)
            cand_unweighted_dists = np.linalg.norm(cand_vectors - detection_vec, axis=1)

            local_order = np.argsort(cand_unweighted_dists)
            top5_local = local_order[:5]
            top5 = candidate_idx[top5_local]

            top5_weighted_dists = cand_weighted_dists[top5_local]
            gate_order = np.argsort(top5_weighted_dists)
            best_idx = int(top5[gate_order[0]])
            best_dist = top5_weighted_dists[gate_order[0]]
            second_dist = top5_weighted_dists[gate_order[1]] if len(top5) > 1 else np.inf
            margin = second_dist - best_dist

            accepted = best_dist < drift_tolerance and margin > margin_required

            candidate_vecs_for_gnn = cand_vectors[top5_local]
            candidate_dists_for_gnn = cand_unweighted_dists[top5_local]
        else:
            local_of = {}
            accepted = False
            best_dist, margin, best_idx = np.inf, 0.0, -1

        accept_path = "gate" if accepted else None
        gnn_decision = None   # "KNOWN" / "AMBIGUOUS" / "NOVEL", only set if the GNN was consulted
        gnn_registered = False
        deferred_ambiguous = False

        gnn_plausible = accept_path is None and best_dist < GNN_PLAUSIBILITY_MULTIPLE * drift_tolerance
        if (accept_path is None and len(candidate_idx) > 0 and gnn_model is not None
                and not gnn_plausible):
            gnn_skipped_implausible += 1

        if not accepted and len(candidate_idx) > 0 and gnn_model is not None and gnn_plausible:
            # --- Build the richer per-candidate feature set (README 4.2) ---
            top5_confidence = np.array([cat_confidence[j] for j in top5], dtype=float)
            top5_recency = np.array([real_now - cat_last_seen[j] for j in top5], dtype=float)
            top5_residual = np.column_stack(embed_mod.physics_consistency_residual(
                np.array([cat_a[j] for j in top5]), np.array([cat_e[j] for j in top5]),
                np.array([cat_i[j] for j in top5]), np.array([cat_raan[j] for j in top5]),
                np.array([cat_argp[j] for j in top5]), np.array([cat_bc[j] for j in top5]),
                np.array([cat_epoch[j] for j in top5]),
                real_now, meas_a, meas_e, meas_i, meas_raan, meas_argp,
            ))
            top5_weighted_vecs = candidate_vecs_for_gnn * distance_weights
            top5_density = ann_index.local_density(top5_weighted_vecs, DENSITY_RADIUS_MULTIPLE * drift_tolerance)
            top5_extra = gnn_mod.build_candidate_features(top5_confidence, top5_recency, top5_residual, top5_density)

            gnn_decision, gnn_local_idx, gnn_conf = gnn_mod.classify_three_way(
                gnn_model, detection_vec, candidate_vecs_for_gnn, candidate_dists_for_gnn, top5_extra,
                confident_known_threshold, confident_novel_threshold,
            )
            _true_idx_for_log = true_name_to_idx.get(name)
            gnn_invocation_log.append({
                "min_dist": float(np.min(candidate_dists_for_gnn)),
                "confidence": float(gnn_conf),
                "decision": gnn_decision,
                "n_cat": int(n_cat),
                "true_object_in_pool": bool(_true_idx_for_log is not None and _true_idx_for_log in top5),
                "true_object_in_ann_pool": bool(_true_idx_for_log is not None and _true_idx_for_log in candidate_idx),
            })
            if gnn_decision == "KNOWN":
                accepted = True
                accept_path = "gnn"
                best_idx = int(top5[gnn_local_idx])
                best_dist = candidate_dists_for_gnn[gnn_local_idx]
            elif gnn_decision == "AMBIGUOUS":
                deferred_ambiguous = True
                gnn_ambiguous_deferred += 1
            else:  # NOVEL
                gnn_confirmed_novel += 1

        if not accepted and ABLATION_MODE == "nearest" and len(candidate_idx) > 0 and gnn_plausible:
            accepted = True
            accept_path = "nearest"

        if not accepted and not deferred_ambiguous:
            # Confirmed NOVEL -- by the gate, and by the GNN if consulted
            # and it also said NOVEL (or wasn't plausible enough to
            # consult at all). Before minting a brand-new entry, check
            # whether this true object was already discovered on an
            # earlier pass -- if so, correct that existing entry instead
            # of spawning a second identity for the same physical object.
            existing_idx = None if ORACLE_FREE else true_name_to_idx.get(name)
            if existing_idx is not None and existing_idx < len(cat_names):
                existing_slot_corrections += 1
                gnn_registered = False
                best_idx = existing_idx
                accepted = True
                accept_path = "existing_slot"
                best_dist = unweighted_dist_for(existing_idx)
                if cat_retired[existing_idx]:
                    cat_retired[existing_idx] = False
            else:
                gnn_registered = True
                new_name = f"discovered_{discovery_counter}"
                discovery_counter += 1
                discovered_true_name[new_name] = name
                true_name_to_idx[name] = len(cat_names)
                cat_names.append(new_name)
                cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
                cat_raan.append(meas_raan); cat_argp.append(meas_argp)
                cat_bc.append(op.ballistic_coefficient(sizes[idx], types[idx]))
                cat_types.append(types[idx]); cat_sizes.append(sizes[idx]); cat_rcs.append(rcs_vals[idx])
                cat_epoch.append(real_now)
                cat_retired.append(False)
                cat_confidence.append(1)
                cat_last_seen.append(real_now)
                ann_index.mark_dirty(len(cat_names) - 1)
                true_name_to_all_slots.setdefault(name, set()).add(len(cat_names) - 1)

        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1_000_000

        total_detections += 1
        obj_type = types[idx]

        update_committed = False
        if accepted:
            known_count += 1
            lookup_times_known.append(elapsed_us)
            match_distances.append(best_dist)
            match_margins.append(margin)
            per_type_known[obj_type] = per_type_known.get(obj_type, 0) + 1
            status = "KNOWN"
            if accept_path == "gnn":
                gnn_confirmed_known += 1

            if best_idx in local_of:
                pa_val, pe_val, pi_val = prop_a(best_idx), prop_e(best_idx), prop_i(best_idx)
                praan_val, pargp_val = prop_raan(best_idx), prop_argp(best_idx)
            else:
                dt_b = real_now - cat_epoch[best_idx]
                pa_arr, pe_arr, pi_arr, praan_arr, pargp_arr = op.propagate_catalog_analytic(
                    np.array([cat_a[best_idx]]), np.array([cat_e[best_idx]]), np.array([cat_i[best_idx]]),
                    np.array([cat_raan[best_idx]]), np.array([cat_argp[best_idx]]), np.array([cat_bc[best_idx]]), dt_b,
                )
                pa_val, pe_val, pi_val = pa_arr[0], pe_arr[0], pi_arr[0]
                praan_val, pargp_val = praan_arr[0], pargp_arr[0]

            update_committed = resolve_update(
                best_idx, name, meas_a, meas_e, meas_i, meas_raan, meas_argp,
                pa_val, pe_val, pi_val, praan_val, pargp_val, real_now,
            )
            ann_index.mark_dirty(best_idx)
            cat_last_seen[best_idx] = real_now
            if update_committed:
                update_committed_count += 1
                cat_confidence[best_idx] += 1
            else:
                update_withheld_count += 1
        elif deferred_ambiguous:
            ambiguous_count += 1
            lookup_times_unknown.append(elapsed_us)
            per_type_unknown[obj_type] = per_type_unknown.get(obj_type, 0) + 1
            status = "AMBIGUOUS"
        else:
            unknown_count += 1
            lookup_times_unknown.append(elapsed_us)
            per_type_unknown[obj_type] = per_type_unknown.get(obj_type, 0) + 1
            status = "NOT KNOWN"
            if gnn_registered:
                pass  # counted via discovery_counter already

        matched_name = cat_names[best_idx] if accepted else None
        matched_true_name = discovered_true_name.get(matched_name, matched_name)
        matched_correctly = accepted and matched_true_name == name
        flag = "" if not accepted else ("" if matched_correctly else "  <-- MISMATCH!")

        if accepted:
            if matched_correctly:
                correct_match_count += 1
            else:
                mismatch_count += 1
        if accept_path == "gnn":
            gnn_overturn_total += 1
            if matched_correctly:
                gnn_overturn_correct += 1

        path_flag = ""
        if not accepted and not deferred_ambiguous and gnn_registered:
            path_flag = "  [registered new]"
        elif deferred_ambiguous:
            path_flag = "  [GNN->AMBIGUOUS, deferred, no mutation]"
        elif accepted:
            commit_tag = "committed" if update_committed else "withheld, pending corroboration"
            if accept_path == "gate":
                path_flag = f"  [gate, update {commit_tag}]"
            elif accept_path == "gnn":
                path_flag = f"  [GNN->KNOWN, update {commit_tag}]"
            elif accept_path == "existing_slot":
                path_flag = f"  [existing-slot correction, update {commit_tag}]"
            elif accept_path == "nearest":
                path_flag = f"  [nearest fallback, update {commit_tag}]"

        print(f"[T+{sim_time:7.2f}s sim] {name:<20} -> {status:<10} "
              f"(dist={best_dist:.3f}, margin={margin:.3f}, lookup {elapsed_us:.2f} us){flag}{path_flag}")

        # -------------------------------------------------------------
        # Retirement housekeeping -- unchanged from the previous design.
        # -------------------------------------------------------------
        if total_detections % RETIREMENT_CHECK_EVERY == 0 and real_now > RETIREMENT_WARMUP_SECONDS:
            for j in range(len(cat_names)):
                if cat_retired[j]:
                    continue
                dt_j = real_now - cat_epoch[j]
                pa_arr, _, _, _, _ = op.propagate_catalog_analytic(
                    np.array([cat_a[j]]), np.array([cat_e[j]]), np.array([cat_i[j]]),
                    np.array([cat_raan[j]]), np.array([cat_argp[j]]), np.array([cat_bc[j]]), dt_j,
                )
                prop_altitude_km = pa_arr[0] - op.RE_EARTH
                time_since_seen = real_now - cat_last_seen[j]

                is_decayed = prop_altitude_km <= RETIREMENT_ALTITUDE_FLOOR_KM
                is_unseen = time_since_seen > RETIREMENT_UNSEEN_SECONDS

                if is_decayed or is_unseen:
                    cat_retired[j] = True
                    reason = "decayed out of orbit" if is_decayed else "unseen too long"
                    if is_decayed:
                        retired_decayed_count += 1
                    else:
                        retired_unseen_count += 1
                    print(f"[T+{sim_time:7.2f}s sim] [RETIRE] {cat_names[j]:<20} ({reason}, "
                          f"alt={prop_altitude_km:.1f}km, unseen={time_since_seen/3600:.1f}h)")
                    pending_update.pop(j, None)

        # -------------------------------------------------------------
        # De-duplication / merge pass -- unchanged from the previous design.
        # -------------------------------------------------------------
        if total_detections % DEDUP_CHECK_EVERY == 0 and len(cat_names) > 1:
            active_indices = np.where(~np.asarray(cat_retired, dtype=bool))[0]
            if len(active_indices) > 1:
                da = np.array([cat_a[j] for j in active_indices])
                de = np.array([cat_e[j] for j in active_indices])
                di = np.array([cat_i[j] for j in active_indices])
                draan = np.array([cat_raan[j] for j in active_indices])
                dargp = np.array([cat_argp[j] for j in active_indices])
                dbc = np.array([cat_bc[j] for j in active_indices])
                depoch = np.array([cat_epoch[j] for j in active_indices])
                dtypes = [cat_types[j] for j in active_indices]
                dsizes = np.array([cat_sizes[j] for j in active_indices])
                drcs = np.array([cat_rcs[j] for j in active_indices])

                d_dt = real_now - depoch
                dp_a, dp_e, dp_i, dp_raan, dp_argp = op.propagate_catalog_analytic(
                    da, de, di, draan, dargp, dbc, d_dt
                )
                d_vecs = embed_mod.build_embeddings_fast(
                    dp_a, dp_e, dp_i, dp_raan, dp_argp, dtypes, dsizes, drcs, scalers
                )
                weighted_d_vecs = d_vecs * distance_weights

                n_neighbors_dedup = min(2, len(active_indices))
                dedup_nn = NearestNeighbors(n_neighbors=n_neighbors_dedup, algorithm="auto")
                dedup_nn.fit(weighted_d_vecs)
                dedup_dists, dedup_pos = dedup_nn.kneighbors(weighted_d_vecs)

                merge_threshold = DEDUP_DISTANCE_FRACTION_OF_DRIFT_TOLERANCE * drift_tolerance
                merged_this_pass = set()
                if n_neighbors_dedup > 1:
                    for local_pos in range(len(active_indices)):
                        if local_pos in merged_this_pass:
                            continue
                        neighbor_local_pos = dedup_pos[local_pos, 1]
                        d = dedup_dists[local_pos, 1]
                        if neighbor_local_pos in merged_this_pass or neighbor_local_pos == local_pos:
                            continue
                        if d < merge_threshold:
                            idx1 = int(active_indices[local_pos])
                            idx2 = int(active_indices[neighbor_local_pos])
                            if cat_confidence[idx2] > cat_confidence[idx1]:
                                keep_idx, retire_idx = idx2, idx1
                                retire_local = local_pos
                            else:
                                keep_idx, retire_idx = idx1, idx2
                                retire_local = neighbor_local_pos
                            cat_retired[retire_idx] = True
                            cat_confidence[keep_idx] += cat_confidence[retire_idx]
                            pending_update.pop(retire_idx, None)
                            merged_this_pass.add(retire_local)
                            merged_count_total += 1
                            print(f"[T+{sim_time:7.2f}s sim] [MERGE] {cat_names[retire_idx]:<20} "
                                  f"-> {cat_names[keep_idx]} (identity dist={d:.4f} < "
                                  f"threshold={merge_threshold:.4f})")
                if merged_this_pass:
                    for retired_local in merged_this_pass:
                        ann_index.mark_dirty(int(active_indices[retired_local]))

        heapq.heappush(heap, (sim_time + sim_periods[idx], idx))

    real_elapsed = time.perf_counter() - real_start

    # -----------------------------------------------------------------
    # End-of-run summary (ground-truth diagnostics unchanged in spirit
    # from the previous design; AMBIGUOUS added throughout)
    # -----------------------------------------------------------------
    active_true_names_correctly_represented = set()
    for cat_idx_final, cat_name_final in enumerate(cat_names):
        if cat_retired[cat_idx_final]:
            continue
        true_name_final = discovered_true_name.get(cat_name_final, cat_name_final)
        active_true_names_correctly_represented.add(true_name_final)

    total_true_population = len(names)
    n_complete = len(active_true_names_correctly_represented & set(names))
    catalog_completeness_pct = (n_complete / total_true_population * 100) if total_true_population else 0.0

    active_slot_counts_per_true_name = []
    for true_name_r, slot_idxs in true_name_to_all_slots.items():
        active_slots_for_this_name = [s for s in slot_idxs if s < len(cat_names) and not cat_retired[s]]
        if active_slots_for_this_name:
            active_slot_counts_per_true_name.append(len(active_slots_for_this_name))
    redundancy_index = (
        sum(active_slot_counts_per_true_name) / len(active_slot_counts_per_true_name)
        if active_slot_counts_per_true_name else 0.0
    )
    n_ghost_registrations = sum(1 for c in active_slot_counts_per_true_name if c > 1)

    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total detections     : {total_detections}")
    print(f"KNOWN                 : {known_count} ({known_count/total_detections*100:.1f}%)")
    print(f"AMBIGUOUS (deferred)  : {ambiguous_count} ({ambiguous_count/total_detections*100:.1f}%)")
    print(f"NOT KNOWN (novel)     : {unknown_count} ({unknown_count/total_detections*100:.1f}%)")
    print(f"Real wall time        : {real_elapsed:.2f}s")
    print(f"Update commits        : {update_committed_count}  withheld: {update_withheld_count}")
    print(f"Truth-confirmation checks: {truth_confirm_checks}; existing-slot corrections: {existing_slot_corrections}")
    print(f"Retired (decayed)     : {retired_decayed_count}  (unseen): {retired_unseen_count}")
    print(f"Merged (de-dup)       : {merged_count_total}")
    print()
    print("--- Association integrity (ground-truth only) ---")
    total_accepted = correct_match_count + mismatch_count
    if total_accepted:
        print(f"Correct matches       : {correct_match_count} ({correct_match_count/total_accepted*100:.1f}%)")
        print(f"Mismatches            : {mismatch_count} ({mismatch_count/total_accepted*100:.1f}%)")
    print(f"Catalog completeness   : {catalog_completeness_pct:.1f}% ({n_complete}/{total_true_population})")
    print(f"Redundancy index       : {redundancy_index:.3f}  (ghost registrations: {n_ghost_registrations})")
    if ann_recall_checked:
        print(f"ANN candidate recall   : {ann_recall_hit/ann_recall_checked*100:.1f}% "
              f"({ann_recall_hit}/{ann_recall_checked})")
    print()
    print("--- Catalog GNN (three-way second opinion) ---")
    print(f"GNN plausible near-miss consults : {gnn_confirmed_known + gnn_confirmed_novel + gnn_ambiguous_deferred}")
    print(f"  -> KNOWN     : {gnn_confirmed_known}")
    print(f"  -> AMBIGUOUS : {gnn_ambiguous_deferred}  (deferred, no mutation)")
    print(f"  -> NOVEL     : {gnn_confirmed_novel}")
    print(f"GNN skipped (implausible near-miss): {gnn_skipped_implausible}")
    if gnn_overturn_total:
        print(f"GNN overturn precision (gate rejected, GNN->KNOWN accepted): "
              f"{gnn_overturn_correct}/{gnn_overturn_total} "
              f"({gnn_overturn_correct/gnn_overturn_total*100:.1f}%)")
    print(f"Final catalog size    : {len(cat_names)}  (active: {sum(1 for r in cat_retired if not r)})")
    print(f"ANN index rebuilds     : {ann_index.n_rebuilds} total "
          f"({density_triggered_rebuild_count[0]} density-triggered, "
          f"{ann_index.n_rebuilds - density_triggered_rebuild_count[0]} cadence/overflow-triggered)")
    print("=" * 70)

    if gnn_invocation_log:
        import json
        Path(LIVE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(LIVE_LOG_PATH, "w") as f:
            json.dump(gnn_invocation_log, f)
        print(f"\nWrote {len(gnn_invocation_log)} live GNN invocations to "
              f"{LIVE_LOG_PATH} -- compare against a fresh training batch "
              f"with compare_live_vs_training.py.")

    if density_diagnostic_log:
        import json
        density_log_path = str(Path(LIVE_LOG_PATH).parent / "ann_density_diagnostic.json")
        with open(density_log_path, "w") as f:
            json.dump(density_diagnostic_log, f)
        raw_vals = [r["raw_density"] for r in density_diagnostic_log]
        raw_sorted = sorted(raw_vals)
        n = len(raw_sorted)
        print(f"\n[DIAGNOSTIC] Wrote {n} sampled raw local-density readings to "
              f"{density_log_path}")
        print(f"[DIAGNOSTIC] raw_density distribution: "
              f"min={raw_sorted[0]} p50={raw_sorted[n//2]} p90={raw_sorted[int(n*0.9)]} "
              f"p95={raw_sorted[int(n*0.95)]} p99={raw_sorted[min(int(n*0.99), n-1)]} "
              f"max={raw_sorted[-1]}  (current ANN_DENSITY_REBUILD_THRESHOLD={ANN_DENSITY_REBUILD_THRESHOLD})")


if __name__ == "__main__":
    main()
