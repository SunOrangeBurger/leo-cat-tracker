#!/usr/bin/env python3
"""Export new seeded Rust measurement streams without training a classifier."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TRACKER_ROOT = Path(__file__).resolve().parents[2]
RUST = TRACKER_ROOT.parent / "rust-leo-sim"
sys.path.insert(0, str(TRACKER_ROOT / "src"))
from rust_stream_bridge import convert


def run_one(out, seed, objects, sensors, steps, stress):
    folder = out / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    public = folder / "observations.jsonl"
    cmd = [str(RUST / "target" / "debug" / "leo_sim"), "--export-observations", str(public),
           "--objects", str(objects), "--sensors", str(sensors), "--steps", str(steps),
           "--seed", str(seed), "--pos-noise-km", "0.5", "--vel-noise-km-s", "0.005"]
    if stress: cmd.append("--stress-test")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    convert(public, folder / "orbital_elements.csv")
    return seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--objects", type=int, default=3000)
    parser.add_argument("--sensors", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--uniform", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(run_one, args.output_dir.resolve(), seed, args.objects,
                            args.sensors, args.steps, not args.uniform)
                for seed in range(args.start_seed, args.start_seed + args.seeds)]
        for job in as_completed(jobs): print("exported", job.result(), flush=True)
