#!/usr/bin/env python3
"""
Successor to gnn_matcher.py (see README.md section 5, item 2).

WHY THIS EXISTS -- the measured failure this file is built to fix
---------------------------------------------------------------------------
The previous GNN second-opinion matcher (gnn_matcher.py) measured
0.18%-0.94% live overturn precision across four rounds of increasingly
aggressive fixes, despite one checkpoint measuring ~90%+ precision on its
own held-out validation split. The root cause (README sections 3.2-3.4),
confirmed with real percentile data:

  - 99.6% of GNN invocations were genuine first-sightings (true object not
    in the candidate pool at all) -- a question no classifier can answer
    correctly by picking one of the 5 candidates.
  - Even restricted to genuine near-misses, KNOWN examples' min-distance
    distribution (p50=0.02x drift_tolerance, p95=2.23x) and NOVEL examples'
    min-distance distribution (p1=1.37x, p50=3.00x) overlap across almost
    their entire range. No threshold on distance/margin alone separates
    them -- this is a property of the catalog's clustered geometry
    (data_generator.py's 5 inclination clusters + narrow eccentricity band
    + lockstep J2 precession within a cluster), not a tuning problem.

This module fixes the actual bottleneck (the feature set + output
contract + training distribution), not the symptom:
  1. Three-way output (KNOWN / AMBIGUOUS / NOVEL) instead of a forced
     binary guess -- see calibrate_confidence_thresholds() and
     classify_three_way() below (README 4.1).
  2. Per-candidate features that are genuinely orthogonal to identity-
     vector distance: corroboration history, physics-consistency residual,
     local catalog density (README 4.2) -- built with
     identity_embedding.physics_consistency_residual() and
     LiveCatalogANNIndex.local_density().
  3. Training data harvested from an actual simulated live loop (growing
     catalog, periodic ANN staleness, repeated not-yet-registered lookups)
     instead of i.i.d. draws against a static full-size catalog (README
     4.3) -- see harvest_live_loop_examples() below.
  4. Asymmetric loss (confident-wrong-KNOWN costs more than confident-
     wrong-NOVEL, matching the real cost asymmetry: a false KNOWN corrupts
     catalog state and compounds; a false NOVEL is a bounded, self-
     correcting redundant registration) -- see asymmetric_bce_loss().
"""

import numpy as np

import orbital_physics as op
import identity_embedding as embed_mod
import leo_ann_index

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------
# Model / graph shape
# ---------------------------------------------------------------------
IDENTITY_DIM = 13
N_CANDIDATES = 5
N_NODES = 1 + N_CANDIDATES     # 1 query + 5 candidates

# Per-candidate extra features beyond the raw identity vector (README 4.2):
#   is_query (1) + log1p(confidence) (1) + log1p(recency_hours) (1)
#   + physics residual, per-element, heuristically scaled (5)
#   + log1p(local_density) (1)
EXTRA_FEAT_DIM = 1 + 1 + 1 + 5 + 1
NODE_FEAT_DIM = IDENTITY_DIM + EXTRA_FEAT_DIM   # 22
HIDDEN = 32

MODEL_PATH = "outputs/catalog_gnn.pt"

# Heuristic normalization constants for the physics-residual features, so
# each residual element lands roughly O(1)-O(10) instead of wildly
# different scales (km vs. degrees vs. unitless eccentricity). These are
# domain priors (same spirit as identity_embedding's cold-start scalers),
# not fitted -- they only need to put the five residual axes on comparable
# footing for the network, not be exact.
RESIDUAL_SCALE = {
    "a_km": 1.0,        # km
    "e": 0.01,           # unitless
    "i_deg": 1.0,         # degrees
    "raan_deg": 1.0,       # degrees
    "argp_deg": 1.0,        # degrees
}

# Local-density query radius, as a multiple of drift_tolerance -- "how
# crowded is this candidate's neighborhood at roughly the gate's own
# acceptance scale" (README 4.2c).
DENSITY_RADIUS_MULTIPLE = 2.0


# ---------------------------------------------------------------------
# Model: same two-round fully-connected message-passing design as the
# previous gnn_matcher.GNNMatcher, just re-sized for the richer node
# features (README keeps the graph-matching approach; the fix is the
# features/data/output contract, not the architecture family).
# ---------------------------------------------------------------------
if TORCH_AVAILABLE:
    class CatalogGNN(nn.Module):
        """Two rounds of message passing over the fully-connected 6-node
        graph, then an edge-scoring head over the 5 query-candidate edges.
        Batched: node_feats (B,6,NODE_FEAT_DIM), edge_dist (B,6,6) ->
        logits (B,5). Each output logit is a calibrated-in-training
        probability that that specific candidate is the true match --
        classify_three_way() below turns the 5 logits into a KNOWN /
        AMBIGUOUS / NOVEL decision via two thresholds, not a single cutoff
        (README 4.1, 4.4)."""

        def __init__(self, feat_dim=NODE_FEAT_DIM, hidden=HIDDEN):
            super().__init__()
            self.msg1 = nn.Linear(feat_dim + 1, hidden)
            self.upd1 = nn.Linear(feat_dim + hidden, hidden)
            self.msg2 = nn.Linear(hidden + 1, hidden)
            self.upd2 = nn.Linear(hidden + hidden, hidden)
            self.score = nn.Linear(hidden * 2 + 1, 1)

        def _message_pass(self, h, edge_dist, msg_layer, upd_layer):
            B, N, D = h.shape
            h_j = h.unsqueeze(1).expand(B, N, N, D)
            edge_feat = edge_dist.unsqueeze(-1)
            msg_in = torch.cat([h_j, edge_feat], dim=-1)
            msg = torch.tanh(msg_layer(msg_in))
            diag_mask = torch.eye(N, device=h.device).view(1, N, N, 1)
            msg = msg * (1 - diag_mask)
            agg = msg.sum(dim=2) / (N - 1)
            upd_in = torch.cat([h, agg], dim=-1)
            return torch.tanh(upd_layer(upd_in))

        def forward(self, node_feats, edge_dist):
            h = self._message_pass(node_feats, edge_dist, self.msg1, self.upd1)
            h = self._message_pass(h, edge_dist, self.msg2, self.upd2)

            h_query = h[:, 0:1, :].expand(-1, N_CANDIDATES, -1)
            h_cand = h[:, 1:N_NODES, :]
            dist_qc = edge_dist[:, 0, 1:N_NODES].unsqueeze(-1)
            score_in = torch.cat([h_query, h_cand, dist_qc], dim=-1)
            logits = self.score(score_in).squeeze(-1)
            return logits


# ---------------------------------------------------------------------
# Per-candidate feature assembly (README 4.2) + graph construction
# ---------------------------------------------------------------------
def build_candidate_features(cand_confidence, cand_recency_seconds, cand_residual, cand_density):
    """
    Assemble the EXTRA_FEAT_DIM (7)-wide feature block for N candidates:
        [log1p(confidence), log1p(recency_hours),
         resid_a/scale, resid_e/scale, resid_i/scale, resid_raan/scale, resid_argp/scale,
         log1p(density)]
    cand_residual is (N, 5): (delta_a, delta_e, delta_i, delta_raan, delta_argp),
    as returned by identity_embedding.physics_consistency_residual().
    """
    cand_confidence = np.asarray(cand_confidence, dtype=float)
    cand_recency_seconds = np.asarray(cand_recency_seconds, dtype=float)
    cand_residual = np.asarray(cand_residual, dtype=float)
    cand_density = np.asarray(cand_density, dtype=float)

    log_conf = np.log1p(np.maximum(cand_confidence, 0.0))
    log_recency = np.log1p(np.maximum(cand_recency_seconds, 0.0) / 3600.0)

    scale = np.array([RESIDUAL_SCALE["a_km"], RESIDUAL_SCALE["e"], RESIDUAL_SCALE["i_deg"],
                       RESIDUAL_SCALE["raan_deg"], RESIDUAL_SCALE["argp_deg"]])
    resid_scaled = cand_residual / scale[None, :]

    log_density = np.log1p(np.maximum(cand_density, 0.0))

    return np.column_stack([log_conf, log_recency, resid_scaled, log_density]).astype(np.float32)


def build_graph(query_vec, candidate_vecs, query_candidate_dists, candidate_extra_feats):
    """
    query_vec: (13,), candidate_vecs: (5,13), query_candidate_dists: (5,)
    (raw, unweighted identity distances -- same space used for the
    edge_dist message-passing signal as the previous design),
    candidate_extra_feats: (5, EXTRA_FEAT_DIM-1) from build_candidate_features
    (the is_query flag is added here, not by the caller).

    -> node_feats (6, NODE_FEAT_DIM), edge_dist (6,6)
    """
    is_query_col = np.zeros((N_NODES, 1), dtype=np.float32)
    is_query_col[0, 0] = 1.0

    identity_block = np.vstack([query_vec[None, :], candidate_vecs]).astype(np.float32)

    query_extra = np.zeros((1, candidate_extra_feats.shape[1]), dtype=np.float32)
    extra_block = np.vstack([query_extra, candidate_extra_feats]).astype(np.float32)

    node_feats = np.concatenate([identity_block, is_query_col, extra_block], axis=1).astype(np.float32)

    edge_dist = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    edge_dist[0, 1:] = query_candidate_dists
    edge_dist[1:, 0] = query_candidate_dists
    diffs = candidate_vecs[:, None, :] - candidate_vecs[None, :, :]
    edge_dist[1:, 1:] = np.linalg.norm(diffs, axis=-1)
    return node_feats, edge_dist


# ---------------------------------------------------------------------
# Live-loop training data harvesting (README 4.3, section 3.4's "root
# cause, stated precisely" / section 3.2's third problem)
# ---------------------------------------------------------------------
DEFAULT_PROCESS_NOISE = {"a_km": 0.05, "e": 0.00005, "i_deg": 0.002, "angle_deg": 0.01}
DEFAULT_MEASUREMENT_NOISE = {"a_km": 0.02, "e": 0.00002, "i_deg": 0.001, "angle_deg": 0.005}
DEFAULT_EMA_ALPHA = 0.4

# Probability a genuinely-novel (not-yet-registered) sighting gets
# registered as a new catalog slot on any given detection where it was
# gate-rejected and consulted. Deliberately < 1 so the harvester reproduces
# the real system's "same not-yet-registered object seen repeatedly before
# it's confirmed" structure (README 4.3), rather than registering on the
# very first sighting every time.
DEFAULT_REGISTER_PROB = 0.35

TRAINING_LOGIC_VERSION = 1  # bump whenever harvest_live_loop_examples' sampling/labeling changes


def harvest_live_loop_examples(
    truth_df, scalers, distance_weights, drift_tolerance, margin_required,
    n_examples, seed=0, embed_fraction=0.5, plausibility_multiple=3.0,
    ema_alpha=DEFAULT_EMA_ALPHA, process_noise=None, measurement_noise=None,
    register_prob=DEFAULT_REGISTER_PROB, rebuild_every=200,
    overflow_threshold=100, candidate_pool_size=40,
    detection_interval_seconds=(30.0, 600.0), max_events=None,
    min_known_examples=None,
):
    """
    Replay a lightweight version of ground_station_tracking_simulation.py's
    actual matching loop -- same growing catalog, same periodically-stale
    ANN index, same "gate-reject is the moment the classifier gets
    consulted" structure -- and harvest a (candidate_set, label,
    per-candidate feature) example at every point the live system would
    actually invoke the classifier (a gate-reject whose nearest candidate
    is plausibly a near-miss, mirroring GNN_PLAUSIBILITY_MULTIPLE in the
    tracking sim).

    IMPORTANT ASYMMETRY (matches the measured live distribution documented
    in this module's docstring, section 3.2: ~99.6% of real GNN
    invocations are genuine first-sightings): once an object is registered
    and corroborated even once, this system's fast gate is extremely
    reliable for it -- a plausible gate-REJECT of an object that IS
    already in the candidate pool ("known-in-pool", the harder, rarer, and
    more informative class for calibrating the confident-KNOWN threshold)
    only happens when a genuine near-collision with a DIFFERENT catalog
    object fails the ambiguity margin, which is a real but empirically
    rare event even at full catalog scale. A pure `len(examples) >=
    n_examples` stopping rule would let the much more common
    genuinely-novel class fill the whole quota long before enough
    known-in-pool examples accumulate to calibrate
    confident_novel_threshold against (see
    calibrate_confidence_thresholds) -- so this function separately
    targets `min_known_examples` known-in-pool examples (keeps simulating,
    without accumulating unbounded extra novel examples once the novel
    quota is already full) and stops when EITHER both targets are met OR
    `max_events` is exhausted (in which case it prints a warning rather
    than silently returning a class-starved batch).

    Deliberate simplifications relative to the full tracking sim (these
    change catalog bookkeeping over very long runs, not the shape of the
    classifier's actual input distribution, so they don't undermine the
    point of this harvester -- see README 4.3):
      - No retirement / de-duplication housekeeping.
      - Belief updates commit in a single pass (no two-pass corroboration
        gate) -- corroboration HISTORY is still tracked and fed to the
        model as a feature either way.
      - Detection scheduling is i.i.d. random-interval sampling rather than
        an orbital-period heap -- irrelevant to what the classifier sees
        (a candidate pool + labels), only to visit ordering.

    Returns a list of dicts, one per harvested example (known-in-pool
    examples first, then novel):
        {
          "query_vec": (13,) float32,
          "candidate_vecs": (5,13) float32,
          "candidate_dists": (5,) float32   (unweighted identity distance),
          "candidate_extra": (5, EXTRA_FEAT_DIM-1) float32,
          "label": (5,) float32             (one-hot on the true match, or
                                              all-zero if the true object
                                              was not in the candidate pool
                                              at all -- a genuine novel case)
        }
    """
    rng = np.random.default_rng(seed)
    process_noise = process_noise or DEFAULT_PROCESS_NOISE
    measurement_noise = measurement_noise or DEFAULT_MEASUREMENT_NOISE
    # Known-in-pool plausible near-misses are rare (see docstring above) --
    # give this a generous event budget by default, since it's an offline,
    # explicitly-triggered training script, not part of any hot path.
    max_events = max_events or (n_examples * 400)
    min_known_examples = min_known_examples if min_known_examples is not None else max(20, n_examples // 400)

    n_truth = len(truth_df)
    truth_a = truth_df["a"].to_numpy(dtype=float).copy()
    truth_e = truth_df["e"].to_numpy(dtype=float).copy()
    truth_i = truth_df["i"].to_numpy(dtype=float).copy()
    truth_raan = truth_df["Omega"].to_numpy(dtype=float).copy()
    truth_argp = truth_df["omega"].to_numpy(dtype=float).copy()
    truth_bc = truth_df["bc_m2_per_kg"].to_numpy(dtype=float).copy()
    truth_epoch = np.zeros(n_truth)
    truth_names = truth_df["name"].tolist()
    truth_types = truth_df["type"].tolist()
    truth_sizes = truth_df["size_meters"].to_numpy(dtype=float)
    truth_rcs = truth_df["rcs"].to_numpy(dtype=float)

    n_preknown = int(round(n_truth * embed_fraction))
    preknown_idx = rng.choice(n_truth, size=n_preknown, replace=False) if n_preknown > 0 else np.array([], dtype=int)

    cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc = [], [], [], [], [], []
    cat_types, cat_sizes, cat_rcs, cat_epoch = [], [], [], []
    cat_confidence, cat_last_update = [], []
    name_to_slot = {}
    for j in preknown_idx:
        cat_a.append(truth_a[j]); cat_e.append(truth_e[j]); cat_i.append(truth_i[j])
        cat_raan.append(truth_raan[j]); cat_argp.append(truth_argp[j]); cat_bc.append(truth_bc[j])
        cat_types.append(truth_types[j]); cat_sizes.append(truth_sizes[j]); cat_rcs.append(truth_rcs[j])
        cat_epoch.append(0.0)
        cat_confidence.append(1)
        cat_last_update.append(0.0)
        name_to_slot[truth_names[j]] = len(cat_a) - 1

    ann_index = leo_ann_index.LiveCatalogANNIndex(
        scalers, rebuild_every=rebuild_every, overflow_threshold=overflow_threshold,
        candidate_pool_size=candidate_pool_size,
    )

    density_radius = DENSITY_RADIUS_MULTIPLE * drift_tolerance

    known_examples = []
    novel_examples = []
    real_now = 0.0

    for _event in range(max_events):
        if len(novel_examples) >= n_examples and len(known_examples) >= min_known_examples:
            break

        idx = int(rng.integers(0, n_truth))
        real_now += float(rng.uniform(*detection_interval_seconds))

        dt_truth = real_now - truth_epoch[idx]
        new_a, new_e, new_i, new_raan, new_argp = op.propagate_elements(
            np.array([truth_a[idx]]), np.array([truth_e[idx]]), np.array([truth_i[idx]]),
            np.array([truth_raan[idx]]), np.array([truth_argp[idx]]), np.array([truth_bc[idx]]),
            dt_truth, process_noise=process_noise, rng=rng,
        )
        truth_a[idx], truth_e[idx], truth_i[idx] = new_a[0], new_e[0], new_i[0]
        truth_raan[idx], truth_argp[idx] = new_raan[0], new_argp[0]
        truth_epoch[idx] = real_now

        meas_a = truth_a[idx] + rng.normal(0, measurement_noise["a_km"])
        meas_e = max(0.0, truth_e[idx] + rng.normal(0, measurement_noise["e"]))
        meas_i = truth_i[idx] + rng.normal(0, measurement_noise["i_deg"])
        meas_raan = (truth_raan[idx] + rng.normal(0, measurement_noise["angle_deg"])) % 360
        meas_argp = (truth_argp[idx] + rng.normal(0, measurement_noise["angle_deg"])) % 360

        detection_vec = embed_mod.transform_identity_vector(
            meas_a, meas_e, meas_i, meas_raan, meas_argp,
            truth_types[idx], truth_sizes[idx], truth_rcs[idx], scalers,
        )

        n_cat = len(cat_a)
        true_name = truth_names[idx]
        true_slot = name_to_slot.get(true_name)

        if n_cat < N_CANDIDATES:
            # Too small a catalog for a meaningful 5-candidate pool yet --
            # still let the true object register organically if unregistered.
            if true_slot is None and rng.random() < register_prob:
                cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
                cat_raan.append(meas_raan); cat_argp.append(meas_argp)
                cat_bc.append(op.ballistic_coefficient(truth_sizes[idx], truth_types[idx]))
                cat_types.append(truth_types[idx]); cat_sizes.append(truth_sizes[idx]); cat_rcs.append(truth_rcs[idx])
                cat_epoch.append(real_now); cat_confidence.append(1); cat_last_update.append(real_now)
                name_to_slot[true_name] = len(cat_a) - 1
                ann_index.mark_dirty(len(cat_a) - 1)
            continue

        weighted_detection_vec = detection_vec * distance_weights
        if ann_index.needs_rebuild():
            ann_index.rebuild(cat_a, cat_e, cat_i, cat_raan, cat_argp, cat_bc,
                               cat_types, cat_sizes, cat_rcs, cat_epoch, real_now, distance_weights)
        candidate_idx = ann_index.query_candidates(weighted_detection_vec, n_cat)
        if len(candidate_idx) == 0:
            continue

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
        prop_a, prop_e, prop_i, prop_raan, prop_argp = op.propagate_catalog_analytic(
            cand_a, cand_e, cand_i, cand_raan, cand_argp, cand_bc, dt_cand
        )
        cand_vectors = embed_mod.build_embeddings_fast(
            prop_a, prop_e, prop_i, prop_raan, prop_argp, cand_types, cand_sizes, cand_rcs, scalers
        )

        cand_unweighted_dists = np.linalg.norm(cand_vectors - detection_vec, axis=1)
        cand_weighted_dists = np.linalg.norm(cand_vectors * distance_weights - weighted_detection_vec, axis=1)

        local_order = np.argsort(cand_unweighted_dists)
        top5_local = local_order[:N_CANDIDATES]
        top5 = candidate_idx[top5_local]

        top5_weighted = cand_weighted_dists[top5_local]
        gate_order = np.argsort(top5_weighted)
        best_dist = top5_weighted[gate_order[0]]
        second_dist = top5_weighted[gate_order[1]] if len(top5) > 1 else np.inf
        margin = second_dist - best_dist
        gate_accepted = best_dist < drift_tolerance and margin > margin_required
        best_idx = int(top5[gate_order[0]])

        is_plausible_near_miss = best_dist < plausibility_multiple * drift_tolerance

        if gate_accepted:
            # Mirrors resolve_update: single-pass commit here (see docstring).
            _local = int(np.where(candidate_idx == best_idx)[0][0])
            cat_a[best_idx] = (1 - ema_alpha) * prop_a[_local] + ema_alpha * meas_a
            cat_e[best_idx] = (1 - ema_alpha) * prop_e[_local] + ema_alpha * meas_e
            cat_i[best_idx] = (1 - ema_alpha) * prop_i[_local] + ema_alpha * meas_i
            cat_raan[best_idx] = (1 - ema_alpha) * prop_raan[_local] + ema_alpha * meas_raan
            cat_argp[best_idx] = (1 - ema_alpha) * prop_argp[_local] + ema_alpha * meas_argp
            cat_epoch[best_idx] = real_now
            cat_confidence[best_idx] += 1
            cat_last_update[best_idx] = real_now
            ann_index.mark_dirty(best_idx)
            continue

        # --- gate rejected: this is a genuine live decision point ---
        if not is_plausible_near_miss:
            # Mirrors GNN_PLAUSIBILITY_MULTIPLE in the tracking sim: the
            # live classifier is never even consulted here, so don't
            # harvest a training example for it either -- but still let
            # genuinely-novel objects register organically.
            if true_slot is None and rng.random() < register_prob * 0.5:
                cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
                cat_raan.append(meas_raan); cat_argp.append(meas_argp)
                cat_bc.append(op.ballistic_coefficient(truth_sizes[idx], truth_types[idx]))
                cat_types.append(truth_types[idx]); cat_sizes.append(truth_sizes[idx]); cat_rcs.append(truth_rcs[idx])
                cat_epoch.append(real_now); cat_confidence.append(1); cat_last_update.append(real_now)
                name_to_slot[true_name] = len(cat_a) - 1
                ann_index.mark_dirty(len(cat_a) - 1)
            continue

        cand_confidence = np.array([cat_confidence[j] for j in top5], dtype=float)
        cand_recency = np.array([real_now - cat_last_update[j] for j in top5], dtype=float)
        cand_residual = np.column_stack(embed_mod.physics_consistency_residual(
            np.array([cat_a[j] for j in top5]), np.array([cat_e[j] for j in top5]),
            np.array([cat_i[j] for j in top5]), np.array([cat_raan[j] for j in top5]),
            np.array([cat_argp[j] for j in top5]), np.array([cat_bc[j] for j in top5]),
            np.array([cat_epoch[j] for j in top5]),
            real_now, meas_a, meas_e, meas_i, meas_raan, meas_argp,
        ))
        cand_weighted_vecs = cand_vectors[top5_local] * distance_weights
        cand_density = ann_index.local_density(cand_weighted_vecs, density_radius)
        candidate_extra = build_candidate_features(cand_confidence, cand_recency, cand_residual, cand_density)

        label = np.zeros(N_CANDIDATES, dtype=np.float32)
        if true_slot is not None and true_slot in top5:
            label[int(np.where(top5 == true_slot)[0][0])] = 1.0

        example = {
            "query_vec": detection_vec.astype(np.float32),
            "candidate_vecs": cand_vectors[top5_local].astype(np.float32),
            "candidate_dists": cand_unweighted_dists[top5_local].astype(np.float32),
            "candidate_extra": candidate_extra,
            "label": label,
        }
        if label.sum() > 0:
            known_examples.append(example)
        elif len(novel_examples) < n_examples:
            novel_examples.append(example)
        # else: novel quota already full -- skip appending (keeps memory
        # bounded) but the catalog-evolution bookkeeping below still runs,
        # since we're still searching for more known-in-pool examples.

        # Keep the catalog evolving realistically after harvesting this
        # decision point (README 4.3's "repeated-failed-lookup structure"):
        if true_slot is not None:
            # Existing-slot correction path: the gate rejected but this
            # object IS already registered -- correct that slot (single
            # pass, see docstring).
            cat_a[true_slot] = (1 - ema_alpha) * cat_a[true_slot] + ema_alpha * meas_a
            cat_e[true_slot] = (1 - ema_alpha) * cat_e[true_slot] + ema_alpha * meas_e
            cat_i[true_slot] = (1 - ema_alpha) * cat_i[true_slot] + ema_alpha * meas_i
            cat_raan[true_slot] = (1 - ema_alpha) * cat_raan[true_slot] + ema_alpha * meas_raan
            cat_argp[true_slot] = (1 - ema_alpha) * cat_argp[true_slot] + ema_alpha * meas_argp
            cat_epoch[true_slot] = real_now
            cat_confidence[true_slot] += 1
            cat_last_update[true_slot] = real_now
            ann_index.mark_dirty(true_slot)
        elif rng.random() < register_prob:
            cat_a.append(meas_a); cat_e.append(meas_e); cat_i.append(meas_i)
            cat_raan.append(meas_raan); cat_argp.append(meas_argp)
            cat_bc.append(op.ballistic_coefficient(truth_sizes[idx], truth_types[idx]))
            cat_types.append(truth_types[idx]); cat_sizes.append(truth_sizes[idx]); cat_rcs.append(truth_rcs[idx])
            cat_epoch.append(real_now); cat_confidence.append(1); cat_last_update.append(real_now)
            name_to_slot[true_name] = len(cat_a) - 1
            ann_index.mark_dirty(len(cat_a) - 1)

    examples = known_examples + novel_examples
    n_known = len(known_examples)
    n_novel = len(novel_examples)
    print(f"Harvested {len(examples)} live-loop examples over {_event + 1} simulated detection "
          f"events ({n_known} true-object-in-pool / {n_novel} genuinely-novel-in-pool).")
    if n_known < min_known_examples:
        print(f"WARNING: only found {n_known}/{min_known_examples} target known-in-pool examples "
              f"before exhausting max_events={max_events}. Known-in-pool near-misses are a rare, "
              f"real class in this system (see docstring) -- increase n_examples/max_events, or "
              f"proceed knowing confident_novel_threshold calibration will be less reliable.")
    return examples


# ---------------------------------------------------------------------
# Asymmetric loss (README 4.4)
# ---------------------------------------------------------------------
def asymmetric_bce_loss(logits, targets, false_known_penalty=3.0):
    """
    Binary cross-entropy per query-candidate edge, with the negative
    (target=0) term up-weighted by `false_known_penalty`. This directly
    encodes the real cost asymmetry documented in README 4.4/README
    section on the three-way contract: a confident-wrong-KNOWN corrupts
    catalog state and compounds over future detections (this codebase's
    very first recorded bug), while a confident-wrong-NOVEL only costs a
    redundant, self-correcting registration. Penalizing false positives
    (predicting a candidate is the match when it isn't) more heavily than
    false negatives during TRAINING is what lets the two-threshold
    calibration in calibrate_confidence_thresholds() find a
    confident-KNOWN cutoff with a low false-accept rate without having to
    push the whole model toward under-confidence everywhere.

    IMPORTANT -- class-count balancing is separate from cost-asymmetry
    weighting, and both are needed. harvest_live_loop_examples() produces
    known-in-pool examples at a naturally low rate (this mirrors the real
    live distribution, README 4.3/section 3.2) -- typically on the order
    of 1 positive edge per ~100-200 negative edges in a batch, since each
    known example contributes 1 positive + 4 negative candidate edges and
    novel examples contribute 5 negative edges each with no positive.
    false_known_penalty alone only encodes the *cost* asymmetry (false-
    KNOWN costs more than false-NOVEL); it says nothing about the *count*
    imbalance, and naively multiplying a >3x cost weight onto a class
    that's already ~100-200x rarer by count compounds into an effective
    weighting so far in the negative direction that the loss-minimizing
    solution becomes "ignore positives, predict near-zero everywhere" --
    a genuine local optimum, not underfitting, and NOT fixable by
    supplying more data (it gets worse with more negatives, not better).
    pos_class_weight neutralizes the count imbalance first (computed from
    the actual batch composition, not hardcoded) so false_known_penalty
    is free to encode pure cost asymmetry on top of an already-balanced
    starting point, as originally intended.
    """
    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

    n_pos = targets.sum()
    n_neg = (1 - targets).sum()
    # Count-balancing weight: puts the positive class's total contribution
    # to the loss on par with the negative class's, regardless of the
    # batch's actual known:novel ratio. Falls back to 1.0 if there are no
    # positives in this batch (shouldn't happen once min_known_examples is
    # set sensibly, but avoids a div-by-zero if it ever does).
    pos_class_weight = (n_neg / n_pos).clamp(min=1.0) if n_pos > 0 else torch.tensor(1.0, device=logits.device)

    pos_term = targets * pos_class_weight * torch.log(probs)
    neg_term = (1 - targets) * false_known_penalty * torch.log(1 - probs)
    return -(pos_term + neg_term).mean()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def _examples_to_tensors(examples, device):
    node_feats_list, edge_dist_list, label_list = [], [], []
    for ex in examples:
        nf, ed = build_graph(ex["query_vec"], ex["candidate_vecs"], ex["candidate_dists"], ex["candidate_extra"])
        node_feats_list.append(nf)
        edge_dist_list.append(ed)
        label_list.append(ex["label"])
    node_feats = torch.from_numpy(np.stack(node_feats_list)).to(device)
    edge_dist = torch.from_numpy(np.stack(edge_dist_list)).to(device)
    labels = torch.from_numpy(np.stack(label_list)).to(device)
    return node_feats, edge_dist, labels


def calibrate_confidence_thresholds(model, val_examples, device, target_novel_fpr=0.02,
                                     target_known_recall=0.90, target_novel_catch_rate=0.90):
    """
    Two-threshold calibration (README 4.1/4.4): the model's learning
    target stays the simple "probability this candidate is the match"
    (asymmetric_bce_loss above); the THREE-WAY operating decision comes
    from calibrating two cutoffs against held-out data, not from training
    a third class directly:

      confident_known_threshold: the score above which a candidate is
        trusted as KNOWN. Set so that only `target_novel_fpr` of held-out
        genuinely-novel examples (true object not in the candidate pool at
        all) would have their best-scoring candidate cross it -- directly
        targets the actual failure mode from README section 3 (the model
        being confidently wrong on genuinely novel objects).

      confident_novel_threshold: the score below which a candidate pool is
        trusted as NOVEL. Set so that at least `target_novel_catch_rate` of
        held-out genuinely-novel examples' best score falls below it --
        i.e. calibrated against the SAME novel-score distribution as
        confident_known_threshold, just at the opposite tail. (Originally
        this was calibrated off known_best_scores instead -- symmetric-
        looking, but wrong: it answered "where do most known scores sit"
        rather than "where do most novel scores sit," and those are
        unrelated quantities that can cross each other, collapsing the
        AMBIGUOUS band. known_best_scores is used below only as a
        post-hoc sanity check, not to set either threshold.)

    Anything between the two thresholds is AMBIGUOUS by construction --
    purely deferred, no catalog mutation (README 4.1).
    """
    model.eval()
    known_best_scores = []
    novel_best_scores = []
    with torch.no_grad():
        for ex in val_examples:
            nf, ed = build_graph(ex["query_vec"], ex["candidate_vecs"], ex["candidate_dists"], ex["candidate_extra"])
            logits = model(torch.from_numpy(nf[None]).to(device), torch.from_numpy(ed[None]).to(device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            best = float(np.max(probs))
            if ex["label"].sum() > 0:
                known_best_scores.append(best)
            else:
                novel_best_scores.append(best)

    if novel_best_scores:
        confident_known_threshold = float(np.percentile(novel_best_scores, 100 * (1 - target_novel_fpr)))
        confident_novel_threshold = float(np.percentile(novel_best_scores, 100 * target_novel_catch_rate))
    else:
        confident_known_threshold = 0.9
        confident_novel_threshold = 0.1

    # Guard against a degenerate crossover (can happen with small/unlucky
    # validation splits) -- AMBIGUOUS band must be non-negative width.
    if confident_novel_threshold > confident_known_threshold:
        print(f"  WARNING: raw calibration crossed (confident_known={confident_known_threshold:.4f} "
              f"< confident_novel={confident_novel_threshold:.4f}, gap={confident_novel_threshold - confident_known_threshold:.4f}) "
              f"-- collapsing both to the midpoint. AMBIGUOUS band will have ZERO width. "
              f"If this keeps happening, raise n_examples (more novel_best_scores) and/or lower "
              f"target_novel_catch_rate / raise target_novel_fpr to widen the gap between percentiles.")
        mid = (confident_novel_threshold + confident_known_threshold) / 2.0
        confident_novel_threshold = confident_known_threshold = mid

    if known_best_scores:
        known_recall_at_novel_cut = float(np.mean(np.array(known_best_scores) > confident_novel_threshold))
        print(f"  (sanity check: {known_recall_at_novel_cut*100:.1f}% of {len(known_best_scores)} held-out "
              f"known examples score above confident_novel_threshold -- target was {target_known_recall*100:.0f}%. "
              f"If this is far off target, known_best_scores is too small/noisy to trust; raise "
              f"min_known_examples rather than adjusting thresholds off of it.)")

    print(f"Calibrated thresholds: confident_known={confident_known_threshold:.4f} "
          f"confident_novel={confident_novel_threshold:.4f} "
          f"(n_known_val={len(known_best_scores)}, n_novel_val={len(novel_best_scores)})")
    return confident_known_threshold, confident_novel_threshold


def train_model(truth_df, scalers, distance_weights, drift_tolerance, margin_required,
                 gate_stamp, n_examples=12000, val_fraction=0.15, epochs=250,
                 false_known_penalty=3.0, lr=1e-3, seed=0, min_known_examples=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the catalog GNN.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = harvest_live_loop_examples(
        truth_df, scalers, distance_weights, drift_tolerance, margin_required,
        n_examples=n_examples, seed=seed, min_known_examples=min_known_examples,
    )
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(examples))
    n_val = max(1, int(len(examples) * val_fraction))
    val_examples = [examples[i] for i in perm[:n_val]]
    train_examples = [examples[i] for i in perm[n_val:]]

    model = CatalogGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_nf, train_ed, train_labels = _examples_to_tensors(train_examples, device)
    val_nf, val_ed, val_labels = _examples_to_tensors(val_examples, device)

    # Track the best-val_loss checkpoint seen so far and restore it at the
    # end, rather than just keeping whatever epoch happens to be last.
    # Val is only evaluated every 25 epochs (see below) to keep the loop
    # cheap, so "best" is best-at-a-checked-epoch, not best-of-every-epoch --
    # coarse, but far better than "last," which on a small/noisy
    # known-in-pool validation split (see harvest_live_loop_examples's
    # docstring on how rare known-in-pool examples are) can end up
    # overfit well past the point where val performance peaked. See
    # run_full_evaluation.py's 07_gnn_training_curve.png plot for what
    # this looks like when it happens: train_loss keeps falling while
    # val_loss turns around and climbs.
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_nf, train_ed)
        loss = asymmetric_bce_loss(logits, train_labels, false_known_penalty=false_known_penalty)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_nf, val_ed)
                val_loss = asymmetric_bce_loss(val_logits, val_labels, false_known_penalty=false_known_penalty)
                val_probs = torch.sigmoid(val_logits)
                val_best = val_probs.max(dim=1).values
                val_has_true = val_labels.sum(dim=1) > 0
                if (~val_has_true).any():
                    val_fp_rate = float((val_best[~val_has_true] > 0.5).float().mean())
                else:
                    val_fp_rate = float("nan")
                if val_has_true.any():
                    val_recall = float((val_best[val_has_true] > 0.5).float().mean())
                else:
                    val_recall = float("nan")
            print(f"epoch {epoch:4d}  train_loss={loss.item():.4f}  val_loss={val_loss.item():.4f}  "
                  f"val_fp_rate(novel called known)={val_fp_rate:.3f}  val_recall(known)={val_recall:.3f}")

            val_loss_value = float(val_loss.item())
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_epoch = epoch
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        print(f"Restoring best checkpoint: epoch {best_epoch} (val_loss={best_val_loss:.4f}), "
              f"vs. final epoch {epochs - 1}.")
        model.load_state_dict(best_state_dict)
    else:
        # Should only happen if epochs == 0; fall back to whatever's in
        # the model as-is rather than crashing.
        print("WARNING: no validation checkpoint was ever recorded -- saving final in-memory weights as-is.")

    confident_known_threshold, confident_novel_threshold = calibrate_confidence_thresholds(
        model, val_examples, device
    )

    torch.save({
        "state_dict": model.state_dict(),
        "gate_stamp": gate_stamp,
        "confident_known_threshold": confident_known_threshold,
        "confident_novel_threshold": confident_novel_threshold,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }, MODEL_PATH)
    print(f"Saved trained catalog GNN to {MODEL_PATH} (checkpoint from epoch {best_epoch})")
    return model, confident_known_threshold, confident_novel_threshold


def load_or_train_model(truth_df, scalers, distance_weights, drift_tolerance, margin_required,
                         force_retrain=False, n_examples=12000, epochs=250,
                         false_known_penalty=3.0, seed=0, min_known_examples=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for the catalog GNN.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the harvester's own default here so it's captured in the
    # gate_stamp -- otherwise a checkpoint trained with the old implicit
    # default (30) would silently pass the stamp check against a run
    # requesting a different min_known_examples.
    resolved_min_known_examples = (
        min_known_examples if min_known_examples is not None
        else max(20, n_examples // 400)
    )

    gate_stamp = {
        "training_logic_version": TRAINING_LOGIC_VERSION,
        "drift_tolerance": round(float(drift_tolerance), 6),
        "margin_required": round(float(margin_required), 6),
        "distance_weights": np.asarray(distance_weights, dtype=np.float32).tolist(),
        "n_examples": int(n_examples),
        "epochs": int(epochs),
        "false_known_penalty": round(float(false_known_penalty), 6),
        "min_known_examples": int(resolved_min_known_examples),
    }

    if not force_retrain:
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if ckpt.get("gate_stamp") == gate_stamp:
                model = CatalogGNN().to(device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                print(f"Loaded existing catalog GNN from {MODEL_PATH} (gate settings match)")
                return model, ckpt["confident_known_threshold"], ckpt["confident_novel_threshold"]
            else:
                print(f"Existing catalog GNN at {MODEL_PATH} was trained under different settings -- "
                      f"retraining instead of reusing a stale checkpoint.")
        except FileNotFoundError:
            pass

    return train_model(truth_df, scalers, distance_weights, drift_tolerance, margin_required,
                        gate_stamp, n_examples=n_examples, epochs=epochs,
                        false_known_penalty=false_known_penalty, seed=seed,
                        min_known_examples=resolved_min_known_examples)


# ---------------------------------------------------------------------
# Inference: three-way decision (README 4.1)
# ---------------------------------------------------------------------
def classify_three_way(model, query_vec, candidate_vecs, candidate_dists, candidate_extra,
                        confident_known_threshold, confident_novel_threshold):
    """
    Returns (decision, matched_local_idx, confidence) where decision is one
    of "KNOWN", "AMBIGUOUS", "NOVEL":
      - KNOWN:     best candidate score >= confident_known_threshold.
                    matched_local_idx is that candidate's index into
                    candidate_vecs (0..4).
      - NOVEL:     best candidate score <= confident_novel_threshold.
                    matched_local_idx is None.
      - AMBIGUOUS: everything in between -- purely defer, no catalog
                    mutation of any kind (README 4.1). matched_local_idx is
                    None.
    """
    node_feats, edge_dist = build_graph(query_vec, candidate_vecs, candidate_dists, candidate_extra)
    device = next(model.parameters()).device
    nf = torch.from_numpy(node_feats[None]).to(device)
    ed = torch.from_numpy(edge_dist[None]).to(device)
    with torch.no_grad():
        logits = model(nf, ed)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    best_local = int(np.argmax(probs))
    best_conf = float(probs[best_local])

    if best_conf >= confident_known_threshold:
        return "KNOWN", best_local, best_conf
    if best_conf <= confident_novel_threshold:
        return "NOVEL", None, best_conf
    return "AMBIGUOUS", None, best_conf