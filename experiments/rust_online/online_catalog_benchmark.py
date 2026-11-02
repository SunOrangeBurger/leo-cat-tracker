#!/usr/bin/env python3
"""Paired online catalog-growth probe on the public Rust observation stream.

Labels are loaded by the experiment wrapper for bootstrap construction,
training/calibration, and final scoring. OnlineEngine receives only public
Cartesian measurements, timestamps, station IDs, and an unlabeled warm catalog.
All decision methods use identical candidate retrieval and lifecycle rules.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

TRACKER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRACKER_ROOT / "src"))
from rust_noise_benchmark import K, load_data, threshold_for_novel_fpr, train_gnn, gnn_probabilities
from rust_noise_orbit_benchmark import (catalog_from_bootstrap, graph_inputs,
                                        jpda_probabilities, propagate_circular,
                                        retrieve, tabular_inputs, calibrate_gnn_gate)


def physics_tabular_inputs(query, candidates, scale):
    """Candidate residuals in the query's radial/tangential/normal frame."""
    physical = query * scale
    r, v = physical[:, :3], physical[:, 3:]
    rhat = r / np.linalg.norm(r, axis=1)[:, None]
    tangential = v - np.sum(v * rhat, axis=1)[:, None] * rhat
    that = tangential / np.linalg.norm(tangential, axis=1)[:, None]
    nhat = np.cross(rhat, that)
    delta = (candidates - query[:, None, :]) * scale
    pos_scale, vel_scale = float(np.median(scale[:3])), float(np.median(scale[3:]))
    projected = np.stack([np.sum(delta[:, :, :3] * axis[:, None, :], axis=2) / pos_scale
                          for axis in (rhat, that, nhat)]
                         + [np.sum(delta[:, :, 3:] * axis[:, None, :], axis=2) / vel_scale
                            for axis in (rhat, that, nhat)], axis=2)
    d2 = np.sum((candidates - query[:, None, :]) ** 2, axis=2)
    sorted_d2 = np.broadcast_to(np.sort(d2, axis=1)[:, None, :], (len(query), K, K))
    rank = np.broadcast_to(np.arange(K)[None, :, None], (len(query), K, 1))
    candidate_pair = np.linalg.norm(candidates[:, :, None, :] - candidates[:, None, :, :], axis=3)
    candidate_pair += np.eye(K)[None, :, :] * 1e9
    nearest_rival = candidate_pair.min(axis=2)
    features = np.concatenate([projected, np.abs(projected), d2[..., None],
                               sorted_d2, nearest_rival[..., None], rank], axis=2)
    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def prepare_models(df, seed, propagator=propagate_circular):
    raw = df[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    ids, t = df.truth_id.to_numpy(), df.t_s.to_numpy()
    rng = np.random.default_rng(seed)
    available = np.unique(ids[t < 600])
    catalog_ids = np.sort(rng.choice(available, len(available) // 2, replace=False))
    catalog0 = catalog_from_bootstrap(df, raw, ids, t, catalog_ids, propagator=propagator)
    train = (t >= 600) & (t < 1200)
    validation = (t >= 1200) & (t < 1600)
    known_train = train & np.isin(ids, catalog_ids)
    lookup = {oid: j for j, oid in enumerate(catalog_ids)}
    pred = np.asarray([propagator(catalog0[[lookup[ids[i]]]], np.array([t[i]]))[0]
                       for i in np.flatnonzero(known_train)])
    residual = raw[known_train] - pred
    scale = np.maximum(np.std(residual, axis=0), [.1, .1, .1, .0001, .0001, .0001])
    query, cand, positions = retrieve(raw, t, catalog0, scale, propagator=propagator)
    labels = (catalog_ids[positions[train]] == ids[train, None]).astype(np.float32)
    node, edge = graph_inputs(query[train], cand[train])
    gnn = train_gnn(node, edge, labels, seed=seed)
    tabular = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                              learning_rate=.05, random_state=seed,
                                              early_stopping=False)
    n_pos = float(labels.sum())
    pos_weight = max(1., (labels.size - n_pos) / max(n_pos, 1.)) / 3.
    tabular.fit(tabular_inputs(query[train], cand[train]), labels.reshape(-1),
                sample_weight=np.where(labels.reshape(-1) > 0, pos_weight, 1.))
    physics_tabular = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                                      learning_rate=.05, random_state=seed,
                                                      early_stopping=False)
    physics_tabular.fit(physics_tabular_inputs(query[train], cand[train], scale),
                        labels.reshape(-1),
                        sample_weight=np.where(labels.reshape(-1) > 0, pos_weight, 1.))
    vq, vc, vp = query[validation], cand[validation], positions[validation]
    vn, ve = graph_inputs(vq, vc)
    gnn_val = gnn_probabilities(gnn, vn, ve)
    tab_val = tabular.predict_proba(tabular_inputs(vq, vc))[:, 1].reshape(-1, K)
    physics_val = physics_tabular.predict_proba(physics_tabular_inputs(vq, vc, scale))[:, 1].reshape(-1, K)
    # The local JPDA comparator uses gate_sq=25. Give both ANN scorers the
    # same physical candidate gate before calibrating their novel rejection.
    common_gate_sq = 25.
    val_d2 = np.sum((vq[:, None, :] - vc) ** 2, axis=2)
    tab_val = np.where(val_d2 <= common_gate_sq, tab_val, 0.)
    physics_val = np.where(val_d2 <= common_gate_sq, physics_val, 0.)
    groups = df[validation].reset_index(drop=True).groupby(["timestamp_ms", "sensor_id"]).indices
    jpda_val, _ = jpda_probabilities(vq, vc, vp, groups,
                                     max_exact_detections=12, max_exact_tracks=32,
                                     fallback_policy="map")
    thresholds = {
        "gnn": calibrate_gnn_gate(gnn_val, vq, vc, vp, ids[validation], catalog_ids),
        "tabular": (threshold_for_novel_fpr(tab_val, ids[validation], catalog_ids), common_gate_sq),
        "physics_tabular": (threshold_for_novel_fpr(physics_val, ids[validation], catalog_ids), common_gate_sq),
        "jpda": (threshold_for_novel_fpr(jpda_val, ids[validation], catalog_ids), float("inf")),
    }
    return catalog_ids, catalog0, scale, gnn, tabular, physics_tabular, thresholds


class OnlineEngine:
    """Truth-blind update and two-sighting birth policy shared by all scorers."""

    def __init__(self, catalog0, scale, scorer, model, threshold, gate,
                 update=True, births=False, birth_gate=2.0, defer_radius_sq=9.0,
                 min_birth_observations=2):
        self.catalog = [x.copy() for x in catalog0]
        self.n_seeded = len(self.catalog)
        self.scale = np.asarray(scale)
        self.scorer = scorer
        self.model = model
        self.threshold = threshold
        self.gate = gate
        self.update = update
        self.births = births
        self.birth_gate = birth_gate
        self.defer_radius_sq = defer_radius_sq
        self.min_birth_observations = min_birth_observations
        self.tentative = []
        self.decisions = {}
        self.candidate_tracks = {}
        self.birth_events = defaultdict(list)
        self.fallbacks = 0
        self.jpda_cluster_stats = {}

    def _probabilities(self, query, cand, pos, sensor_ids):
        if self.scorer == "gnn":
            nodes, edges = graph_inputs(query, cand)
            return gnn_probabilities(self.model, nodes, edges)
        if self.scorer == "tabular":
            return self.model.predict_proba(tabular_inputs(query, cand))[:, 1].reshape(-1, K)
        if self.scorer == "physics_tabular":
            return self.model.predict_proba(physics_tabular_inputs(query, cand, self.scale))[:, 1].reshape(-1, K)
        groups = {int(s): np.flatnonzero(sensor_ids == s).tolist() for s in np.unique(sensor_ids)}
        out, fallback = jpda_probabilities(query, cand, pos, groups,
                                           max_exact_detections=12, max_exact_tracks=32,
                                           fallback_policy="map",
                                           diagnostics=self.jpda_cluster_stats)
        self.fallbacks += fallback
        return out

    def process_epoch(self, row_ids, raw, times, sensor_ids):
        epoch = float(times[0])
        catalog = np.asarray(self.catalog)
        query, cand, pos = retrieve(raw, times, catalog, self.scale)
        prob = self._probabilities(query, cand, pos, sensor_ids)
        d2 = np.sum((query[:, None, :] - cand) ** 2, axis=2)
        prob = np.where(d2 <= self.gate, prob, 0.)
        best = prob.argmax(axis=1)
        accepted = prob[np.arange(len(raw)), best] >= self.threshold
        updates = defaultdict(list)
        for local, row_id in enumerate(row_ids):
            self.candidate_tracks[int(row_id)] = pos[local].copy()
            back = propagate_circular(raw[local:local+1], np.array([-epoch]))[0]
            if accepted[local]:
                track = int(pos[local, best[local]])
                self.decisions[int(row_id)] = track
                if self.update: updates[track].append(back)
            else:
                self.decisions[int(row_id)] = -1
                if self.births and d2[local, 0] >= self.defer_radius_sq:
                    self._consider_birth(int(row_id), back, epoch)
        # Batch updates avoid observation-order effects inside a scan.
        for track, measurements in updates.items():
            center = np.median(np.asarray(measurements), axis=0)
            self.catalog[track] = .9 * self.catalog[track] + .1 * center
        self.tentative = [x for x in self.tentative if epoch - x["last_time"] <= 600]

    def _consider_birth(self, row_id, back, epoch):
        state = back / self.scale
        closest, distance = None, float("inf")
        for index, item in enumerate(self.tentative):
            d = float(np.linalg.norm(state - item["state"]))
            if d < distance:
                closest, distance = index, d
        if closest is None or distance > self.birth_gate:
            self.tentative.append({"state": state.copy(), "measurements": [back],
                                   "rows": [row_id], "first_time": epoch, "last_time": epoch})
            return
        item = self.tentative[closest]
        item["measurements"].append(back)
        item["rows"].append(row_id)
        item["state"] = np.median(np.asarray(item["measurements"]), axis=0) / self.scale
        item["last_time"] = epoch
        if len(item["rows"]) < self.min_birth_observations or epoch - item["first_time"] < 30:
            return
        track = len(self.catalog)
        self.catalog.append(np.median(np.asarray(item["measurements"]), axis=0))
        for past_row in item["rows"]:
            self.decisions[past_row] = track
            self.birth_events[track].append(past_row)
        self.tentative.pop(closest)


def score_online(decisions, candidates, truth, catalog_ids, birth_events, fallbacks):
    seeded = set(map(int, catalog_ids))
    known = np.isin(truth, catalog_ids)
    pred = np.asarray([decisions.get(i, -1) for i in range(len(truth))])
    correct = np.zeros(len(truth), bool)
    seeded_accept = (pred >= 0) & (pred < len(catalog_ids))
    correct[seeded_accept] = catalog_ids[pred[seeded_accept]] == truth[seeded_accept]
    births = {}
    for track, rows in birth_events.items():
        labels = [int(truth[i]) for i in rows]
        dominant, count = Counter(labels).most_common(1)[0]
        births[int(track)] = {"dominant_truth": dominant, "purity": count / len(labels),
                              "observations": len(labels)}
    # Newly born tracks may gather later observations; assess all assigned rows.
    for track in np.unique(pred[pred >= len(catalog_ids)]):
        rows = np.flatnonzero(pred == track)
        labels = truth[rows].tolist()
        dominant, count = Counter(labels).most_common(1)[0]
        births[int(track)] = {"dominant_truth": int(dominant), "purity": count / len(labels),
                              "observations": len(labels)}
        correct[rows] = truth[rows] == dominant
    born_truth = [b["dominant_truth"] for b in births.values()]
    duplicates = sum(v - 1 for v in Counter(born_truth).values() if v > 1)
    novel = ~known
    candidate_recall = np.array([
        any(j < len(catalog_ids) and catalog_ids[j] == truth[i]
            for j in candidates[i]) for i in range(len(truth))], bool)
    result = {"test_observations": len(truth), "known_observations": int(known.sum()),
            "novel_observations": int(novel.sum()),
            "known_correct": int((known & correct & seeded_accept).sum()),
            "known_wrong_or_fragmented": int((known & (pred >= 0) & ~(correct & seeded_accept)).sum()),
            "known_rejected": int((known & (pred < 0)).sum()),
            "novel_to_seeded_false_merge": int((novel & seeded_accept).sum()),
            "novel_rejected": int((novel & (pred < 0)).sum()),
            "novel_to_born_track": int((novel & (pred >= len(catalog_ids))).sum()),
            "born_track_count": len(births),
            "born_track_mean_purity": float(np.mean([b["purity"] for b in births.values()])) if births else None,
            "born_duplicate_dominant_truth": int(duplicates),
            "born_majority_novel_truth": int(sum(b["dominant_truth"] not in seeded for b in births.values())),
            "born_distinct_novel_truth": len(set(born_truth) - seeded),
            "born_known_fragment_tracks": int(sum(b["dominant_truth"] in seeded for b in births.values())),
            "born_low_purity_tracks": int(sum(b["purity"] < .8 for b in births.values())),
            "known_correct_per_detection": float((known & correct & seeded_accept).sum()/max(1, known.sum())),
            "seeded_candidate_recall_at_5": float(candidate_recall[known].mean()) if known.any() else None,
            "novel_seeded_false_merge_rate": float((novel & seeded_accept).sum()/max(1, novel.sum())),
            "jpda_fallbacks": int(fallbacks)}
    assert result["known_correct"]+result["known_wrong_or_fragmented"]+result["known_rejected"] == result["known_observations"]
    assert result["novel_to_seeded_false_merge"]+result["novel_rejected"]+result["novel_to_born_track"] == result["novel_observations"]
    return result


def run(csv_path, truth_path, output_path, seed, update=True, births=False,
        birth_gate=2.0, defer_radius_sq=9.0, min_birth_observations=2):
    df = load_data(csv_path, truth_path)
    catalog_ids, catalog0, scale, gnn, tabular, physics_tabular, thresholds = prepare_models(df, seed)
    test = df.t_s.to_numpy() >= 1600
    public = df[test].reset_index(drop=True)
    raw = public[["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]].to_numpy(float)
    times = public.t_s.to_numpy()
    sensors = public.sensor_id.to_numpy()
    truth = public.truth_id.to_numpy()
    outcomes = {}
    for method, model in [("gnn", gnn), ("tabular", tabular),
                          ("physics_tabular", physics_tabular), ("jpda", None)]:
        threshold, gate = thresholds[method]
        engine = OnlineEngine(catalog0, scale, method, model, threshold, gate,
                              update=update, births=births, birth_gate=birth_gate,
                              defer_radius_sq=defer_radius_sq,
                              min_birth_observations=min_birth_observations)
        started = time.perf_counter()
        for epoch in np.unique(times):
            rows = np.flatnonzero(times == epoch)
            engine.process_epoch(rows, raw[rows], times[rows], sensors[rows])
        outcomes[method] = score_online(engine.decisions, engine.candidate_tracks, truth, catalog_ids,
                                        engine.birth_events, engine.fallbacks)
        outcomes[method]["online_runtime_seconds"] = time.perf_counter() - started
        if method == "jpda": outcomes[method]["cluster_diagnostics"] = engine.jpda_cluster_stats
    result = {"design": "paired online catalog, common retrieval/update/birth policy, validation-calibrated local scorers",
              "seed": seed, "update": update, "births": births, "catalog_seeded": len(catalog_ids),
              "birth_gate": birth_gate, "defer_radius_sq": defer_radius_sq,
              "min_birth_observations": min_birth_observations,
              "scale": scale.tolist(), "thresholds": {k: list(v) for k, v in thresholds.items()},
              "methods": outcomes}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--no-update", action="store_true")
    birth_mode = parser.add_mutually_exclusive_group()
    birth_mode.add_argument("--experimental-births", action="store_true")
    birth_mode.add_argument("--no-births", action="store_true", help="explicit safe-mode alias")
    parser.add_argument("--birth-gate", type=float, default=2.0)
    parser.add_argument("--defer-radius-sq", type=float, default=9.0)
    parser.add_argument("--min-birth-observations", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.truth, args.output, args.seed,
                         update=not args.no_update, births=args.experimental_births,
                         birth_gate=args.birth_gate, defer_radius_sq=args.defer_radius_sq,
                         min_birth_observations=args.min_birth_observations), indent=2))
