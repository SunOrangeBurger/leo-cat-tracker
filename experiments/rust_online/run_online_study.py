#!/usr/bin/env python3
"""Run and aggregate the paired online growing-catalog experiment."""

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TRACKER_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent


def mean_interval(values):
    x = np.asarray(values, float)
    rng = np.random.default_rng(20261102)
    samples = rng.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
    return [float(x.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def run_one(source, output, seed, no_update, experimental_births,
            birth_gate, defer_radius_sq, min_birth_observations):
    folder = source / f"seed_{seed}"
    cmd = [sys.executable, str(SCRIPTS / "online_catalog_benchmark.py"),
           str(folder / "orbital_elements.csv"), str(folder / "observations.jsonl.truth.jsonl"),
           str(output / f"seed_{seed}.json"), "--seed", str(seed),
           "--birth-gate", str(birth_gate), "--defer-radius-sq", str(defer_radius_sq),
           "--min-birth-observations", str(min_birth_observations)]
    if no_update: cmd.append("--no-update")
    if experimental_births: cmd.append("--experimental-births")
    with (output / f"seed_{seed}.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=TRACKER_ROOT, check=True)
    return seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory containing seed_N folders from the ten-seed Rust archive")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=11)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-update", action="store_true")
    birth_mode = parser.add_mutually_exclusive_group()
    birth_mode.add_argument("--experimental-births", action="store_true")
    birth_mode.add_argument("--no-births", action="store_true", help="explicit safe-mode alias")
    parser.add_argument("--birth-gate", type=float, default=2.0)
    parser.add_argument("--defer-radius-sq", type=float, default=9.0)
    parser.add_argument("--min-birth-observations", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    if not args.analyze_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, source, output, seed, args.no_update, args.experimental_births,
                                   args.birth_gate, args.defer_radius_sq, args.min_birth_observations)
                       for seed in seeds]
            for future in as_completed(futures):
                print("completed", future.result(), flush=True)
    runs = [json.loads((output / f"seed_{seed}.json").read_text()) for seed in seeds]
    keys = ["known_correct_per_detection", "novel_seeded_false_merge_rate",
            "seeded_candidate_recall_at_5", "online_runtime_seconds",
            "born_track_mean_purity", "born_track_count", "born_distinct_novel_truth",
            "born_duplicate_dominant_truth", "born_known_fragment_tracks", "born_low_purity_tracks"]
    aggregate = {}
    for method in ["gnn", "tabular", "physics_tabular", "jpda"]:
        aggregate[method] = {}
        for key in keys:
            values = [r["methods"][method][key] for r in runs]
            aggregate[method][key] = (None if all(v is None for v in values)
                                      else mean_interval([v or 0 for v in values]))
    for method in aggregate:
        aggregate[method]["total_known_correct"] = sum(r["methods"][method]["known_correct"] for r in runs)
        aggregate[method]["total_novel_to_seeded_false_merge"] = sum(r["methods"][method]["novel_to_seeded_false_merge"] for r in runs)
        aggregate[method]["total_born_tracks"] = sum(r["methods"][method]["born_track_count"] for r in runs)
    comparisons = {}
    for method in ["gnn", "tabular", "physics_tabular"]:
        difference = np.array([r["methods"][method]["known_correct_per_detection"]
                               - r["methods"]["jpda"]["known_correct_per_detection"] for r in runs])
        false_merge_difference = np.array([
            r["methods"][method]["novel_seeded_false_merge_rate"]
            - r["methods"]["jpda"]["novel_seeded_false_merge_rate"] for r in runs])
        observed = abs(difference.mean())
        signflip = np.array([abs(np.mean(difference * np.asarray(signs)))
                             for signs in itertools.product([-1, 1], repeat=len(runs))])
        comparisons[method] = {"known_recall_difference": mean_interval(difference),
                               "novel_seeded_false_merge_difference": mean_interval(false_merge_difference),
                               "paired_seed_signflip_p": float(np.mean(signflip >= observed-1e-12))}
    summary = {"design": "same Rust stream, candidate pool and lifecycle policy for GNN, ANN-tabular, and local JPDA",
               "seeds": seeds, "update": not args.no_update, "births": args.experimental_births,
               "birth_gate": args.birth_gate, "defer_radius_sq": args.defer_radius_sq,
               "min_birth_observations": args.min_birth_observations,
               "python": platform.python_version(),
               "code_sha256": {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
                               for name in ["online_catalog_benchmark.py", "run_online_study.py"]},
               "metrics": aggregate, "compared_with_jpda": comparisons, "runs": runs}
    (output / "online_study_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__": main()
