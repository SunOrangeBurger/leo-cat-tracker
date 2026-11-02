#!/usr/bin/env python3
"""Run a paired ten-seed Rust-stream attribution study and summarize it."""

import argparse
import json
import os
import platform
import subprocess
import sys
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

TRACKER_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def threshold_from_validation(scores, known, target_fpr=.02):
    novel_scores = scores[~known]
    return float(np.quantile(novel_scores, 1-target_fpr, method="higher") + 1e-6)


def calibrate_gate_with_known(d2, truth, candidates, known):
    correct_first = candidates[:, 0] == truth
    margin = d2[:, 1] - d2[:, 0]
    best = (-1., 0., float("inf"))
    for radius in [1., 2., 4., 9., 16., 25., 49., 100., float("inf")]:
        for gap in [0., .5, 1., 2., 4., 9., 16., 25.]:
            accept = (d2[:, 0] <= radius) & (margin >= gap)
            fpr = accept[~known].mean() if (~known).any() else 0.
            recall = np.mean(accept[known] & correct_first[known]) if known.any() else 0.
            if fpr <= .02 and recall > best[0]: best = (recall, radius, gap)
    return best[1], best[2]


def summarize_decision(truth, candidates, known, accepted, best):
    predicted = candidates[np.arange(len(truth)), best]
    correct = accepted & (predicted == truth)
    known_correct = int((correct & known).sum())
    known_wrong = int((accepted & known & ~correct).sum())
    known_rejected = int((~accepted & known).sum())
    novel_false_merge = int((accepted & ~known).sum())
    novel_rejected = int((~accepted & ~known).sum())
    accepted_total = int(accepted.sum())
    return {
        "known_correct": known_correct, "known_wrong": known_wrong,
        "known_rejected": known_rejected, "novel_false_merge": novel_false_merge,
        "novel_rejected_birth_candidate": novel_rejected,
        "correct_per_known_detection": known_correct / max(1, int(known.sum())),
        "accepted_correctness": known_correct / max(1, accepted_total),
        "novelty_precision": novel_rejected / max(1, novel_rejected + known_rejected),
        "novelty_recall": novel_rejected / max(1, novel_rejected + novel_false_merge),
    }, correct & known


def summarize_seed(folder):
    result = json.loads((folder / "results.json").read_text())
    data = np.load(folder / "results.events.npz")
    truth, candidates, known = data["truth_id"], data["candidate_ids"], data["known"]
    val_truth, val_candidates = data["val_truth_id"], data["val_candidate_ids"]
    val_known = np.isin(val_truth, data["catalog_ids"])
    decisions = {}
    per_event = {}
    for method in ["gnn", "tabular", "jpda", "hungarian"]:
        prob = data[f"{method}_prob"]
        threshold = result[f"{method}_threshold"]
        accepted = prob.max(axis=1) >= threshold
        best = prob.argmax(axis=1)
        decisions[method], per_event[method] = summarize_decision(truth, candidates, known, accepted, best)
    d2, val_d2 = data["candidate_d2"], data["val_candidate_d2"]
    val_nn_score = np.exp(-.5 * val_d2[:, 0])
    nn_threshold = threshold_from_validation(val_nn_score, val_known)
    nn_accepted = np.exp(-.5 * d2[:, 0]) >= nn_threshold
    decisions["nearest"], per_event["nearest"] = summarize_decision(
        truth, candidates, known, nn_accepted, np.zeros(len(truth), int))
    gate_radius, gate_margin = calibrate_gate_with_known(val_d2, val_truth, val_candidates, val_known)
    val_gate = (val_d2[:, 0] <= gate_radius) & (val_d2[:, 1] - val_d2[:, 0] >= gate_margin)
    gate = (d2[:, 0] <= gate_radius) & (d2[:, 1] - d2[:, 0] >= gate_margin)
    decisions["gate_novel"], per_event["gate_novel"] = summarize_decision(
        truth, candidates, known, gate, np.zeros(len(truth), int))
    # A looser nearest-candidate rescue is calibrated only among rejected
    # validation events, then applied after the identical first gate.
    val_reject_novel = ~val_gate & ~val_known
    fallback_threshold = (threshold_from_validation(val_nn_score[val_reject_novel],
                                                     np.zeros(int(val_reject_novel.sum()), bool))
                          if val_reject_novel.any() else 1.)
    rescued = gate | (np.exp(-.5 * d2[:, 0]) >= fallback_threshold)
    decisions["gate_nearest"], per_event["gate_nearest"] = summarize_decision(
        truth, candidates, known, rescued, np.zeros(len(truth), int))
    return {
        "seed": result["seed"], "known_test": int(known.sum()),
        "novel_test": int((~known).sum()),
        "candidate_recall_at_5": result["candidate_recall_known_test"],
        "decisions": decisions, "timing_seconds": result["timing_seconds"],
        "jpda_degraded_clusters": result["jpda_degraded_clusters"],
        "known_tie_challenge": result.get("known_tie_challenge", []),
    }, per_event


def ci(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    rng = np.random.default_rng(20261101)
    samples = rng.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
    return [mean, float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def run_one(seed, root, objects, sensors, steps, benchmark_only=False, uniform=False):
    folder = root / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if benchmark_only:
        cmd = [sys.executable, str(TRACKER_ROOT / "src" / "rust_noise_orbit_benchmark.py"),
               str(folder / "orbital_elements.csv"), str(folder / "observations.jsonl.truth.jsonl"),
               str(folder / "results.json"), "--seed", str(seed)]
    else:
        cmd = [sys.executable, str(SCRIPTS / "run_unified_pipeline.py"),
               "--output-dir", str(folder), "--seed", str(seed),
               "--objects", str(objects), "--sensors", str(sensors), "--steps", str(steps)]
        if not uniform: cmd.append("--stress-test")
    with (folder / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=TRACKER_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=11)
    parser.add_argument("--objects", type=int, default=3000)
    parser.add_argument("--sensors", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--uniform", action="store_true", help="use Rust's uniform-object distribution")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    if not args.analyze_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, s, root, args.objects, args.sensors, args.steps,
                                   args.benchmark_only, args.uniform) for s in seeds]
            for future in as_completed(futures):
                print("completed seed", future.result(), flush=True)
    runs, event_arrays = [], []
    for seed in seeds:
        row, events = summarize_seed(root / f"seed_{seed}")
        runs.append(row)
        event_arrays.append(events)
    methods = list(runs[0]["decisions"])
    aggregate = {}
    for method in methods:
        aggregate[method] = {metric: ci([r["decisions"][method][metric] for r in runs])
                             for metric in ["correct_per_known_detection", "accepted_correctness",
                                            "novelty_precision", "novelty_recall"]}
        aggregate[method]["outcome_totals"] = {
            key: sum(r["decisions"][method][key] for r in runs)
            for key in ["known_correct", "known_wrong", "known_rejected",
                        "novel_false_merge", "novel_rejected_birth_candidate"]}
    known_gnn = np.concatenate([x["gnn"] for x in event_arrays])
    known_jpda = np.concatenate([x["jpda"] for x in event_arrays])
    b = int((known_gnn & ~known_jpda).sum())
    c = int((~known_gnn & known_jpda).sum())
    paired_differences = np.array([
        r["decisions"]["gnn"]["correct_per_known_detection"]
        - r["decisions"]["jpda"]["correct_per_known_detection"] for r in runs])
    observed_mean = abs(float(paired_differences.mean()))
    signflip_means = [abs(float(np.mean(paired_differences * np.asarray(signs))))
                      for signs in itertools.product([-1, 1], repeat=len(runs))]
    signflip_p = float(np.mean(np.asarray(signflip_means) >= observed_mean - 1e-12))
    study = {
        "design": "paired fixed warm-catalog Rust Cartesian-noise study",
        "config": {"seeds": seeds, "objects": args.objects, "sensors": args.sensors,
                   "steps": args.steps, "stress_test": not args.uniform,
                   "pos_noise_km": .5, "vel_noise_km_s": .005,
                   "bootstrap_s": 600, "train_s": [600,1200],
                   "validation_s": [1200,1600], "test_from_s": 1600},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                        "decision_parallelism": "single process per method; two simultaneous independent seeds"},
        "candidate_recall_at_5": ci([r["candidate_recall_at_5"] for r in runs]),
        "methods": aggregate,
        "gnn_vs_jpda_mcnemar": {"gnn_only_correct": b, "jpda_only_correct": c,
                                "exact_two_sided_p": float(binomtest(b, b+c, .5).pvalue) if b+c else 1.},
        "gnn_minus_jpda_known_recall_by_seed": paired_differences.tolist(),
        "gnn_minus_jpda_known_recall_ci": ci(paired_differences),
        "paired_seed_signflip_p": signflip_p,
        "timing_seconds_per_seed": {
            key: ci([r["timing_seconds"][key] for r in runs])
            for key in runs[0]["timing_seconds"]},
        "jpda_degraded_clusters_total": sum(r["jpda_degraded_clusters"] for r in runs),
        "known_tie_challenge": [
            {"separation_km": runs[0]["known_tie_challenge"][j]["separation_km"],
             "gnn_correct_per_detection": ci([r["known_tie_challenge"][j]["gnn"]["correct_per_detection"] for r in runs]),
             "jpda_correct_per_detection": ci([r["known_tie_challenge"][j]["jpda"]["correct_per_detection"] for r in runs]),
             "nearest_top1_accuracy": ci([r["known_tie_challenge"][j]["nearest_top1_accuracy"] for r in runs])}
            for j in range(len(runs[0]["known_tie_challenge"]))
        ],
        "runs": runs,
    }
    (root / "study_summary.json").write_text(json.dumps(study, indent=2))
    print(json.dumps({"candidate_recall_at_5": study["candidate_recall_at_5"],
                      "methods": aggregate, "gnn_vs_jpda_mcnemar": study["gnn_vs_jpda_mcnemar"]}, indent=2))


if __name__ == "__main__": main()
