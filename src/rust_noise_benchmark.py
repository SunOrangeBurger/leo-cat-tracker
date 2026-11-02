#!/usr/bin/env python3
"""Exploratory shared-stream Rust-noise evaluation of local GNN and JPDA.

This uses a fresh five-coordinate, nonsingular orbital signature trained from
Rust Cartesian observations. It does not reproduce the published 13-D
catalog-growth experiment. Truth labels build a warm catalog and supervise
training, but are absent from either test-time decision function.
"""

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from catalog_gnn import CatalogGNN
from jpda_baseline import enumerate_joint_events

MU = 398600.4418
K = 5


def signature(df):
    """Phase-invariant signature: semi-major axis, eccentricity, orbit normal."""
    r = df[["x_km", "y_km", "z_km"]].to_numpy(float)
    v = df[["vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    if np.any(rn <= 0) or np.any(hn <= 0):
        raise ValueError("invalid Cartesian state")
    energy = np.einsum("ij,ij->i", v, v) / 2 - MU / rn
    if np.any(energy >= 0):
        raise ValueError("unbound Cartesian state")
    a = -MU / (2 * energy)
    evec = np.cross(v, h) / MU - r / rn[:, None]
    e = np.linalg.norm(evec, axis=1)
    return np.column_stack([a, e, h / hn[:, None]])


def load_data(csv_path, truth_path):
    df = pd.read_csv(csv_path)
    with Path(truth_path).open() as f:
        truth = [json.loads(line)["object_index"] for line in f]
    if len(df) != len(truth):
        raise ValueError("public stream and evaluation truth have different lengths")
    df["truth_id"] = np.asarray(truth, dtype=int)
    # The first exported state is after one propagation step. Use that
    # observation as the local epoch for circular-state fitting and splits.
    df["t_s"] = (df.timestamp_ms - df.timestamp_ms.min()) / 1000
    return df


def make_examples(features, ids, catalog_features, catalog_ids, knn):
    dists, positions = knn.kneighbors(features, n_neighbors=K)
    labels = (catalog_ids[positions] == ids[:, None]).astype(np.float32)
    node = np.concatenate([features[:, None, :], catalog_features[positions]], axis=1)
    query_flag = np.zeros((len(features), K + 1, 1), np.float32)
    query_flag[:, 0, 0] = 1
    node = np.concatenate([node, query_flag], axis=2).astype(np.float32)
    edge = np.linalg.norm(node[:, :, None, :5] - node[:, None, :, :5], axis=3).astype(np.float32)
    edge = np.clip(edge, 0, 30)
    return node, edge, labels, positions, dists


def train_gnn(node, edge, labels, seed=11, epochs=25):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = CatalogGNN(feat_dim=node.shape[-1], hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    x = torch.from_numpy(node)
    e = torch.from_numpy(edge)
    y = torch.from_numpy(labels)
    n_pos = float(y.sum())
    # First balance rare positives, then assign 3x greater cost to a false
    # known edge, matching the final pipeline's asymmetric training intent.
    pos_weight = max(1., (y.numel() - n_pos) / max(n_pos, 1.)) / 3.
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for batch in np.array_split(rng.permutation(len(x)), max(1, (len(x) + 127) // 128)):
            logits = model(x[batch], e[batch])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[batch], pos_weight=torch.tensor(pos_weight))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def gnn_probabilities(model, node, edge):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(node), 256):
            logits = model(torch.from_numpy(node[start:start+256]),
                           torch.from_numpy(edge[start:start+256]))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def jpda_probabilities(features, catalog_features, groups, gate_sq=16.0):
    """Small-cluster joint-event JPDA with an explicit degraded-mode count."""
    knn = NearestNeighbors(n_neighbors=K, algorithm="kd_tree").fit(catalog_features)
    dists, positions = knn.kneighbors(features)
    probabilities = np.zeros((len(features), K), dtype=float)
    fallback = 0
    for event_indices in groups.values():
        # Bipartite adjacency within one sensor epoch; duplicate sightings
        # from different stations are never constrained to be exclusive.
        track_to_det = defaultdict(set)
        det_to_track = defaultdict(set)
        for di in event_indices:
            for dist, cat_idx in zip(dists[di], positions[di]):
                if dist * dist <= gate_sq:
                    track_to_det[int(cat_idx)].add(di)
                    det_to_track[di].add(int(cat_idx))
        visited = set()
        for seed in event_indices:
            if seed in visited or seed not in det_to_track:
                continue
            queue = deque([("d", seed)])
            cluster_d, cluster_t = set(), set()
            while queue:
                kind, item = queue.popleft()
                if kind == "d":
                    if item in cluster_d: continue
                    cluster_d.add(item)
                    queue.extend(("t", x) for x in det_to_track[item])
                else:
                    if item in cluster_t: continue
                    cluster_t.add(item)
                    queue.extend(("d", x) for x in track_to_det[item])
            visited.update(cluster_d)
            dets, tracks = sorted(cluster_d), sorted(cluster_t)
            if max(len(dets), len(tracks)) > 8:
                fallback += 1
                for di in dets:
                    gated = [j for j, cat_idx in enumerate(positions[di]) if int(cat_idx) in cluster_t and dists[di, j]**2 <= gate_sq]
                    if gated:
                        local = min(gated, key=lambda j: dists[di, j])
                        probabilities[di, local] = np.exp(-0.5 * dists[di, local]**2)
                continue
            td = {t: ti for ti, t in enumerate(tracks)}
            dd = {d: j for j, d in enumerate(dets)}
            feasible = {(td[t], dd[d]): True for d in dets for t in det_to_track[d]}
            weights = np.zeros((len(tracks), len(dets)))
            total = 0.
            for event in enumerate_joint_events(list(range(len(tracks))), list(range(len(dets))), feasible):
                weight = 1.
                for ti, local_di in event.items():
                    if local_di is None:
                        weight *= .08
                    else:
                        d = dets[local_di]
                        t = tracks[ti]
                        dist = np.linalg.norm(features[d] - catalog_features[t])
                        weight *= .92 * np.exp(-.5 * dist * dist)
                weight *= .02 ** (len(dets) - sum(v is not None for v in event.values()))
                total += weight
                for ti, local_di in event.items():
                    if local_di is not None:
                        weights[ti, local_di] += weight
            if total > 0:
                weights /= total
                for ti, t in enumerate(tracks):
                    for local_di, d in enumerate(dets):
                        local = np.where(positions[d] == t)[0]
                        if len(local): probabilities[d, local[0]] = weights[ti, local_di]
    return probabilities, positions, fallback


def score(prob, positions, truth_ids, catalog_ids, threshold):
    best = np.argmax(prob, axis=1)
    confidence = np.take_along_axis(prob, best[:, None], axis=1)[:, 0]
    accepted = confidence >= threshold
    predicted = catalog_ids[positions[np.arange(len(best)), best]]
    correct = accepted & (predicted == truth_ids)
    known = np.isin(truth_ids, catalog_ids)
    return {
        "observations": int(len(best)), "known_observations": int(known.sum()),
        "accepted": int(accepted.sum()), "correct_accepted": int(correct.sum()),
        "mismatched_accepted": int((accepted & ~correct).sum()),
        "known_recall": float(correct[known].mean()) if known.any() else None,
        "novel_false_accept_rate": float(accepted[~known].mean()) if (~known).any() else None,
        "accepted_correctness": float(correct.sum() / accepted.sum()) if accepted.any() else None,
    }


def threshold_for_novel_fpr(prob, truth_ids, catalog_ids, target=.02):
    novel = ~np.isin(truth_ids, catalog_ids)
    maxp = prob.max(axis=1)
    if not novel.any(): return 1.
    return float(np.quantile(maxp[novel], 1 - target, method="higher") + 1e-6)


def run(csv_path, truth_path, output_path):
    begin = time.perf_counter()
    df = load_data(csv_path, truth_path)
    raw = signature(df)
    ids = df.truth_id.to_numpy()
    t = df.t_s.to_numpy()
    bootstrap = t < 600
    rng = np.random.default_rng(11)
    bootstrap_ids = np.unique(ids[bootstrap])
    catalog_ids = np.sort(rng.choice(bootstrap_ids, len(bootstrap_ids)//2, replace=False))
    catalog_raw = np.array([np.median(raw[bootstrap & (ids == oid)], axis=0) for oid in catalog_ids])
    # Fixed, training-only scale. Orbit normal is dimensionless, while a is km.
    train_ids = ids[bootstrap]
    dispersion = []
    for oid in np.unique(train_ids):
        rows = raw[bootstrap & (ids == oid)]
        if len(rows) > 1: dispersion.extend(rows - np.median(rows, axis=0))
    scale = np.maximum(np.std(dispersion, axis=0), [1., .001, .001, .001, .001])
    features = raw / scale
    catalog_features = catalog_raw / scale
    knn = NearestNeighbors(n_neighbors=K, algorithm="kd_tree").fit(catalog_features)
    train = (t >= 600) & (t < 1200)
    validation = (t >= 1200) & (t < 1600)
    test = t >= 1600
    node, edge, labels, _, _ = make_examples(features[train], ids[train], catalog_features, catalog_ids, knn)
    model = train_gnn(node, edge, labels)
    val_node, val_edge, _, val_pos, _ = make_examples(features[validation], ids[validation], catalog_features, catalog_ids, knn)
    test_node, test_edge, _, test_pos, _ = make_examples(features[test], ids[test], catalog_features, catalog_ids, knn)
    gnn_val = gnn_probabilities(model, val_node, val_edge)
    gnn_test = gnn_probabilities(model, test_node, test_edge)
    gnn_threshold = threshold_for_novel_fpr(gnn_val, ids[validation], catalog_ids)
    val_df = df[validation].reset_index(drop=True)
    test_df = df[test].reset_index(drop=True)
    val_groups = val_df.groupby(["timestamp_ms", "sensor_id"]).indices
    test_groups = test_df.groupby(["timestamp_ms", "sensor_id"]).indices
    jpda_val, jpda_val_pos, val_fallback = jpda_probabilities(features[validation], catalog_features, val_groups)
    jpda_test, jpda_test_pos, test_fallback = jpda_probabilities(features[test], catalog_features, test_groups)
    jpda_threshold = threshold_for_novel_fpr(jpda_val, ids[validation], catalog_ids)
    result = {
        "experiment": "Rust Cartesian-noise static warm-catalog probe; not archived catalog-growth benchmark",
        "split_seconds": {"bootstrap_before": 600, "train": [600, 1200], "validation": [1200, 1600], "test_from": 1600},
        "rows": {"bootstrap": int(bootstrap.sum()), "train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())},
        "catalog_objects": int(len(catalog_ids)), "feature_scale": scale.tolist(),
        "train_positive_edges": int(labels.sum()),
        "candidate_recall_known_train": float(labels.sum() / max(1, np.isin(ids[train], catalog_ids).sum())),
        "candidate_recall_known_test": float(np.mean(np.any(catalog_ids[test_pos] == ids[test, None], axis=1)[np.isin(ids[test], catalog_ids)])),
        "gnn_threshold": gnn_threshold,
        "jpda_threshold": jpda_threshold,
        "gnn": score(gnn_test, test_pos, ids[test], catalog_ids, gnn_threshold),
        "jpda": score(jpda_test, jpda_test_pos, ids[test], catalog_ids, jpda_threshold),
        "jpda_degraded_clusters": test_fallback, "jpda_validation_degraded_clusters": val_fallback,
        "total_wall_seconds": time.perf_counter() - begin,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.truth, args.output), indent=2))


if __name__ == "__main__": main()
