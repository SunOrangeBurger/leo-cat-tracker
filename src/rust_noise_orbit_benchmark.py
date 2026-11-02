#!/usr/bin/env python3
"""Shared Rust observation stream: propagated-state ANN/GNN versus local JPDA.

This is a fixed warm-catalog experiment. It uses noisy Cartesian states,
seeded catalog tracks built from an earlier time window, and a held-out
future interval. Ground-truth labels are used only for seeding and training
and for scoring after inference. Neither decision function reads test labels.
"""

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment

from rust_noise_benchmark import K, MU, load_data, score, threshold_for_novel_fpr, train_gnn, gnn_probabilities
from jpda_baseline import jpda_marginals_exact


def propagate_circular(states, dt_seconds):
    """Predict circular two-body states from rows of Cartesian initial states."""
    r0, v0 = states[:, :3], states[:, 3:]
    radius = np.linalg.norm(r0, axis=1)
    rhat = r0 / radius[:, None]
    tangent = v0 - np.sum(v0 * rhat, axis=1)[:, None] * rhat
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    omega = np.sqrt(MU / radius**3)
    angle = omega * np.asarray(dt_seconds)
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    r = radius[:, None] * (c * rhat + s * tangent)
    v = np.sqrt(MU / radius)[:, None] * (-s * rhat + c * tangent)
    return np.column_stack([r, v])


def catalog_from_bootstrap(df, raw, ids, t, chosen_ids, propagator=propagate_circular):
    catalog = []
    for oid in chosen_ids:
        idx = np.flatnonzero((ids == oid) & (t < 600))
        # Undo each observation's circular motion, then average the noisy
        # states. Identity labels are used here to seed a warm catalog only.
        back = propagator(raw[idx], -t[idx])
        catalog.append(np.median(back, axis=0))
    return np.asarray(catalog)


def retrieve(raw, t, catalog0, scale, propagator=propagate_circular):
    """Time-batched KD-tree retrieval over propagated 6-D catalog states."""
    n, k = len(raw), K
    pos = np.empty((n, k), dtype=int)
    cand = np.empty((n, k, 6), dtype=np.float32)
    query = (raw / scale).astype(np.float32)
    for epoch in np.unique(t):
        rows = np.flatnonzero(t == epoch)
        pred = propagator(catalog0, np.full(len(catalog0), epoch)) / scale
        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(pred)
        _, local = nn.kneighbors(query[rows])
        pos[rows] = local
        cand[rows] = pred[local]
    return query, cand, pos


def graph_inputs(query, candidates):
    # Center every local graph on its observation. Absolute ECI coordinates
    # are thousands of km and saturate the tanh message layers; residuals
    # carry the association signal and preserve the same geometry.
    nodes = np.concatenate([np.zeros_like(query)[:, None, :], candidates - query[:, None, :]], axis=1)
    flag = np.zeros((len(nodes), K + 1, 1), np.float32)
    flag[:, 0, 0] = 1.
    nodes = np.concatenate([nodes, flag], axis=2).astype(np.float32)
    diff = nodes[:, :, None, :6] - nodes[:, None, :, :6]
    edges = np.clip(np.linalg.norm(diff, axis=3), 0, 30).astype(np.float32)
    return nodes, edges


def tabular_inputs(query, candidates):
    """Candidate residuals, rank, and all five local distances for a non-graph control."""
    residual = candidates - query[:, None, :]
    d2 = np.sum(residual**2, axis=2)
    all_distances = np.broadcast_to(np.sort(d2, axis=1)[:, None, :], (len(query), K, K))
    rank = np.broadcast_to(np.arange(K)[None, :, None], (len(query), K, 1))
    return np.concatenate([residual, d2[..., None], all_distances, rank], axis=2).reshape(-1, 13)


def jpda_map_fallback(d2, positions, detections, tracks, gate_sq=25.0,
                      detection_weight=.92, miss_weight=.08, clutter_weight=.02,
                      likelihood_scale=None):
    """One-to-one MAP event under the same miss/clutter weights as JPDA.

    Returns ``(detection_index, local_candidate_index)`` associations.
    Reject columns represent an unused detection; missed tracks are implicit.
    """
    dets = list(detections)
    ts = list(tracks)
    lookup = {track: j for j, track in enumerate(ts)}
    n_d = len(dets)
    cost = np.full((n_d, len(ts) + n_d), 1e6, dtype=float)
    cost[:, len(ts):] = 0.
    # Relative to a missed track and clutter detection, a match has odds
    # 0.92*exp(-d2/2)/(0.08*0.02). The assignment minimizes negative log odds.
    log_prior_odds = np.log(detection_weight / (miss_weight * clutter_weight))
    for row, di in enumerate(dets):
        for j, track in enumerate(positions[di]):
            if d2[di, j] <= gate_sq and int(track) in lookup:
                adjustment=0. if likelihood_scale is None else -np.log(likelihood_scale[di,j])
                cost[row, lookup[int(track)]] = .5 * d2[di, j] - log_prior_odds+adjustment
    rows, cols = linear_sum_assignment(cost)
    associations = []
    for row, col in zip(rows, cols):
        if col >= len(ts) or cost[row, col] >= 0: continue
        di = dets[row]
        local = np.flatnonzero(positions[di] == ts[col])
        if len(local): associations.append((di, int(local[0])))
    return associations


def jpda_probabilities(query, cand, positions, groups, gate_sq=25.0,
                       max_exact_detections=None, max_exact_tracks=32,
                       fallback_policy="greedy", diagnostics=None,
                       detection_weight=.92, miss_weight=.08, clutter_weight=.02,
                       likelihood_scale=None):
    """Local JPDA on a shared candidate pool.

    The default eight-object cap and greedy fallback preserve the archived
    fixed-catalog experiment. Online studies may opt into a detection-mask
    cap and one-to-one MAP fallback, both logged via ``diagnostics``.
    """
    probabilities = np.zeros((len(query), K), float)
    if likelihood_scale is not None:
        likelihood_scale=np.asarray(likelihood_scale,float)
        if likelihood_scale.shape != probabilities.shape or not np.isfinite(likelihood_scale).all() or np.any(likelihood_scale<=0):
            raise ValueError('likelihood scale must be finite, positive, and match candidate shape')
    d2 = np.sum((query[:, None, :] - cand)**2, axis=2)
    fallbacks = 0
    for event_indices in groups.values():
        track_to_det, det_to_track = defaultdict(set), defaultdict(set)
        for di in event_indices:
            for j, cat_idx in enumerate(positions[di]):
                if d2[di, j] <= gate_sq:
                    track_to_det[int(cat_idx)].add(di)
                    det_to_track[di].add(int(cat_idx))
        seen = set()
        for seed in event_indices:
            if seed in seen or seed not in det_to_track: continue
            q = deque([("d", seed)])
            ds, ts = set(), set()
            while q:
                kind, item = q.popleft()
                if kind == "d":
                    if item in ds: continue
                    ds.add(item)
                    q.extend(("t", x) for x in det_to_track[item])
                else:
                    if item in ts: continue
                    ts.add(item)
                    q.extend(("d", x) for x in track_to_det[item])
            seen.update(ds)
            ds, ts = sorted(ds), sorted(ts)
            if diagnostics is not None:
                diagnostics["largest_tracks"] = max(diagnostics.get("largest_tracks", 0), len(ts))
                diagnostics["largest_detections"] = max(diagnostics.get("largest_detections", 0), len(ds))
            degraded = (max(len(ds), len(ts)) > 8 if max_exact_detections is None
                        else len(ds) > max_exact_detections or len(ts) > max_exact_tracks)
            if degraded:
                fallbacks += 1
                if diagnostics is not None:
                    diagnostics.setdefault("fallback_cluster_sizes", []).append([len(ts), len(ds)])
                if fallback_policy == "map":
                    for di, j in jpda_map_fallback(d2, positions, ds, ts, gate_sq,
                            detection_weight, miss_weight, clutter_weight,likelihood_scale):
                        probabilities[di, j] = np.exp(-.5 * d2[di, j])
                elif fallback_policy == "greedy":
                    for di in ds:
                        gated = [j for j, cat_idx in enumerate(positions[di]) if int(cat_idx) in ts and d2[di, j] <= gate_sq]
                        if gated:
                            j = min(gated, key=lambda j: d2[di, j])
                            probabilities[di, j] = np.exp(-.5 * d2[di, j])
                else:
                    raise ValueError("fallback_policy must be 'greedy' or 'map'")
                continue
            track_local = {x: i for i, x in enumerate(ts)}
            match = np.zeros((len(ts), len(ds)), dtype=float)
            for dj, di in enumerate(ds):
                for j, cat_idx in enumerate(positions[di]):
                    if d2[di, j] <= gate_sq and int(cat_idx) in track_local:
                        factor=1. if likelihood_scale is None else likelihood_scale[di,j]
                        match[track_local[int(cat_idx)], dj] = detection_weight * np.exp(-.5 * d2[di, j])*factor
            beta = jpda_marginals_exact(match, miss_weight=miss_weight, clutter_weight=clutter_weight)
            for ti, cat_idx in enumerate(ts):
                for dj, di in enumerate(ds):
                    local = np.flatnonzero(positions[di] == cat_idx)
                    if len(local): probabilities[di, local[0]] = beta[ti, dj]
    return probabilities, fallbacks


def hungarian_probabilities(query, cand, positions, groups, birth_cost=25.0):
    """Global nearest assignment per station epoch with reject columns."""
    d2 = np.sum((query[:, None, :] - cand)**2, axis=2)
    probabilities = np.zeros((len(query), K), float)
    for event_indices in groups.values():
        detections = list(event_indices)
        tracks = sorted({int(t) for di in detections for t in positions[di]})
        lookup = {t: j for j, t in enumerate(tracks)}
        n_d = len(detections)
        cost = np.full((n_d, len(tracks) + n_d), birth_cost + 1000.)
        cost[:, len(tracks):] = birth_cost
        for row, di in enumerate(detections):
            for j, t in enumerate(positions[di]):
                cost[row, lookup[int(t)]] = d2[di, j]
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            if col >= len(tracks) or cost[row, col] >= birth_cost: continue
            di = detections[row]
            local = np.flatnonzero(positions[di] == tracks[col])
            if len(local): probabilities[di, local[0]] = np.exp(-.5 * cost[row, col])
    return probabilities


def calibrate_gnn_gate(prob, query, candidates, positions, truth_ids, catalog_ids, target_fpr=.02):
    """Pick confidence and physical-residual gates on validation only."""
    d2 = np.sum((query[:, None, :] - candidates)**2, axis=2)
    best = (0., 1., float("inf"))
    for gate in [4., 9., 16., 25., 49., 100., float("inf")]:
        gated = np.where(d2 <= gate, prob, 0.)
        for threshold in np.unique(np.quantile(gated.max(axis=1), np.linspace(.5, .999, 100))):
            threshold = float(threshold + 1e-6)
            result = score(gated, positions, truth_ids, catalog_ids, threshold)
            if result["novel_false_accept_rate"] <= target_fpr:
                recall = result["known_recall"] or 0.
                if recall > best[0]: best = (recall, threshold, gate)
    return best[1], best[2]


def known_tie_challenge(model, query, candidates, positions, truth_ids, catalog_ids,
                        scale, gnn_threshold, gnn_gate, jpda_threshold, seed):
    """Inject a same-orbit phase rival at controlled along-track separation."""
    true_local = catalog_ids[positions] == truth_ids[:, None]
    eligible = np.flatnonzero(true_local.any(axis=1))
    rng = np.random.default_rng(seed + 1000)
    if len(eligible) > 400:
        eligible = np.sort(rng.choice(eligible, 400, replace=False))
    summary = []
    for separation_km in [0., 1., 2., 5., 10., 20., 50.]:
        n = len(eligible)
        ch = np.empty((n, K, 6), np.float32)
        truth_slot = np.empty(n, int)
        q = query[eligible]
        for row, source in enumerate(eligible):
            ti = int(np.argmax(true_local[source]))
            true_state = candidates[source, ti].astype(float)
            r = true_state[:3] * scale[:3]
            v = true_state[3:] * scale[3:]
            normal = np.cross(r, v)
            normal /= np.linalg.norm(normal)
            angle = separation_km / np.linalg.norm(r)
            def rotate(vec):
                return (np.cos(angle)*vec + np.sin(angle)*np.cross(normal, vec)
                        + (1-np.cos(angle))*np.dot(normal, vec)*normal)
            rival = np.concatenate([rotate(r), rotate(v)]) / scale
            other = [j for j in range(K) if j != ti]
            far = sorted(other, key=lambda j: np.linalg.norm(candidates[source,j] - q[row]), reverse=True)[:3]
            flip = row % 2
            if flip:
                ch[row, :2] = [rival, true_state]
                truth_slot[row] = 1
            else:
                ch[row, :2] = [true_state, rival]
                truth_slot[row] = 0
            ch[row, 2:] = candidates[source, far]
        node, edge = graph_inputs(q, ch)
        gprob = gnn_probabilities(model, node, edge)
        d2 = np.sum((q[:, None, :] - ch)**2, axis=2)
        gprob = np.where(d2 <= gnn_gate, gprob, 0.)
        dummy_positions = np.tile(np.arange(K), (n, 1))
        groups = {j: [j] for j in range(n)}
        jprob, _ = jpda_probabilities(q, ch, dummy_positions, groups)
        nearest = np.argmin(d2, axis=1)
        def measure(prob, threshold):
            accepted = prob.max(axis=1) >= threshold
            correct = accepted & (prob.argmax(axis=1) == truth_slot)
            return {"correct_per_detection": float(correct.mean()),
                    "accepted_correctness": float(correct.sum()/max(1, accepted.sum())),
                    "accepted": int(accepted.sum())}
        summary.append({"separation_km": separation_km, "n": n,
                        "gnn": measure(gprob, gnn_threshold),
                        "jpda": measure(jprob, jpda_threshold),
                        "nearest_top1_accuracy": float(np.mean(nearest == truth_slot))})
    return summary


def run(csv_path, truth_path, output_path, seed=11):
    begin = time.perf_counter()
    df = load_data(csv_path, truth_path)
    raw = df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    ids, t = df.truth_id.to_numpy(), df.t_s.to_numpy()
    rng = np.random.default_rng(seed)
    available = np.unique(ids[t < 600])
    catalog_ids = np.sort(rng.choice(available, len(available)//2, replace=False))
    catalog0 = catalog_from_bootstrap(df, raw, ids, t, catalog_ids)
    # Estimate residual noise only from the supervised training interval.
    training = (t >= 600) & (t < 1200)
    validation = (t >= 1200) & (t < 1600)
    test = t >= 1600
    known_train = training & np.isin(ids, catalog_ids)
    cat_lookup = {oid: j for j, oid in enumerate(catalog_ids)}
    pred_true = np.array([propagate_circular(catalog0[[cat_lookup[ids[i]]]], np.array([t[i]]))[0]
                          for i in np.flatnonzero(known_train)])
    residual = raw[known_train] - pred_true
    scale = np.maximum(np.std(residual, axis=0), [.1, .1, .1, .0001, .0001, .0001])
    retrieval_start = time.perf_counter()
    query, candidates, positions = retrieve(raw, t, catalog0, scale)
    retrieval_seconds = time.perf_counter() - retrieval_start
    train_node, train_edge = graph_inputs(query[training], candidates[training])
    train_labels = (catalog_ids[positions[training]] == ids[training, None]).astype(np.float32)
    model = train_gnn(train_node, train_edge, train_labels, seed=seed)
    tabular = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                              learning_rate=.05, random_state=seed,
                                              early_stopping=False)
    n_pos = train_labels.sum()
    positive_weight = max(1., (train_labels.size - n_pos) / max(n_pos, 1.)) / 3.
    tabular.fit(tabular_inputs(query[training], candidates[training]),
                train_labels.reshape(-1),
                sample_weight=np.where(train_labels.reshape(-1) > 0, positive_weight, 1.))
    val_node, val_edge = graph_inputs(query[validation], candidates[validation])
    test_node, test_edge = graph_inputs(query[test], candidates[test])
    gnn_val = gnn_probabilities(model, val_node, val_edge)
    gnn_start = time.perf_counter()
    gnn_test = gnn_probabilities(model, test_node, test_edge)
    gnn_test_seconds = time.perf_counter() - gnn_start
    tabular_val = tabular.predict_proba(tabular_inputs(query[validation], candidates[validation]))[:, 1].reshape(-1, K)
    tabular_start = time.perf_counter()
    tabular_test = tabular.predict_proba(tabular_inputs(query[test], candidates[test]))[:, 1].reshape(-1, K)
    tabular_test_seconds = time.perf_counter() - tabular_start
    gnn_threshold, gnn_gate = calibrate_gnn_gate(
        gnn_val, query[validation], candidates[validation], positions[validation],
        ids[validation], catalog_ids)
    test_d2 = np.sum((query[test, None, :] - candidates[test])**2, axis=2)
    gnn_test = np.where(test_d2 <= gnn_gate, gnn_test, 0.)
    tabular_threshold = threshold_for_novel_fpr(tabular_val, ids[validation], catalog_ids)
    val_groups = df[validation].reset_index(drop=True).groupby(["timestamp_ms", "sensor_id"]).indices
    test_groups = df[test].reset_index(drop=True).groupby(["timestamp_ms", "sensor_id"]).indices
    jpda_val, val_fallback = jpda_probabilities(query[validation], candidates[validation], positions[validation], val_groups)
    jpda_start = time.perf_counter()
    jpda_test, test_fallback = jpda_probabilities(query[test], candidates[test], positions[test], test_groups)
    jpda_test_seconds = time.perf_counter() - jpda_start
    jpda_threshold = threshold_for_novel_fpr(jpda_val, ids[validation], catalog_ids)
    hungarian_val = hungarian_probabilities(query[validation], candidates[validation], positions[validation], val_groups)
    hungarian_start = time.perf_counter()
    hungarian_test = hungarian_probabilities(query[test], candidates[test], positions[test], test_groups)
    hungarian_test_seconds = time.perf_counter() - hungarian_start
    hungarian_threshold = threshold_for_novel_fpr(hungarian_val, ids[validation], catalog_ids)
    known_test = np.isin(ids[test], catalog_ids)
    ties = known_tie_challenge(model, query[test], candidates[test], positions[test],
                               ids[test], catalog_ids, scale, gnn_threshold, gnn_gate,
                               jpda_threshold, seed)
    result = {
        "experiment": "Rust noisy Cartesian stream, circular propagation, fixed warm catalog, held-out time; no catalog births/updates",
        "rows": {"bootstrap": int((t < 600).sum()), "train": int(training.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
        "catalog_objects": int(len(catalog_ids)), "feature_scale": scale.tolist(),
        "seed": seed,
        "train_positive_edges": int(train_labels.sum()),
        "candidate_recall_known_test": float(np.mean(np.any(catalog_ids[positions[test]] == ids[test, None], axis=1)[known_test])),
        "gnn_threshold": gnn_threshold, "gnn_gate_sq": gnn_gate,
        "tabular_threshold": tabular_threshold, "jpda_threshold": jpda_threshold,
        "hungarian_threshold": hungarian_threshold,
        "gnn": score(gnn_test, positions[test], ids[test], catalog_ids, gnn_threshold),
        "tabular": score(tabular_test, positions[test], ids[test], catalog_ids, tabular_threshold),
        "jpda": score(jpda_test, positions[test], ids[test], catalog_ids, jpda_threshold),
        "hungarian": score(hungarian_test, positions[test], ids[test], catalog_ids, hungarian_threshold),
        "jpda_degraded_clusters": test_fallback, "jpda_validation_degraded_clusters": val_fallback,
        "known_tie_challenge": ties,
        "timing_seconds": {"shared_retrieval_all_rows": retrieval_seconds,
                            "gnn_test_decision": gnn_test_seconds,
                            "tabular_test_decision": tabular_test_seconds,
                            "jpda_test_decision": jpda_test_seconds,
                            "hungarian_test_decision": hungarian_test_seconds},
        "total_wall_seconds": time.perf_counter() - begin,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        output_path.with_suffix(".events.npz"),
        truth_id=ids[test], known=known_test,
        candidate_ids=catalog_ids[positions[test]],
        gnn_prob=gnn_test, tabular_prob=tabular_test, jpda_prob=jpda_test,
        hungarian_prob=hungarian_test,
        candidate_d2=np.sum((query[test, None, :] - candidates[test])**2, axis=2),
        val_truth_id=ids[validation],
        val_candidate_ids=catalog_ids[positions[validation]],
        val_candidate_d2=np.sum((query[validation, None, :] - candidates[validation])**2, axis=2),
        val_gnn_prob=gnn_val, val_tabular_prob=tabular_val,
        val_jpda_prob=jpda_val, val_hungarian_prob=hungarian_val,
        catalog_ids=catalog_ids,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.truth, args.output, seed=args.seed), indent=2))


if __name__ == "__main__": main()
