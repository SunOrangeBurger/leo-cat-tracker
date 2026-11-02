#!/usr/bin/env python3
"""Run Rust sensor export, Python conversion, and paired association probe.

Example:
  python pipeline/run_unified_pipeline.py --output-dir /tmp/leo-unified

The result is a fixed warm-catalog held-out experiment. The archived
growing-catalog paper benchmark is a separate experiment.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TRACKER_ROOT = Path(__file__).resolve().parents[2]
RUST = TRACKER_ROOT.parent / "rust-leo-sim"
PYTHON_SRC = TRACKER_ROOT / "src"
sys.path.insert(0, str(PYTHON_SRC))
from rust_stream_bridge import convert
from rust_noise_orbit_benchmark import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objects", type=int, default=3000)
    parser.add_argument("--sensors", type=int, default=24)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--pos-noise-km", type=float, default=0.5)
    parser.add_argument("--vel-noise-km-s", type=float, default=0.005)
    parser.add_argument("--stress-test", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    executable = RUST / "target" / "debug" / "leo_sim"
    if args.rebuild or not executable.exists():
        subprocess.run(["cargo", "build", "--offline"], cwd=RUST, check=True)
    public = out / "observations.jsonl"
    cmd = [str(executable), "--export-observations", str(public),
           "--objects", str(args.objects), "--sensors", str(args.sensors),
           "--steps", str(args.steps), "--seed", str(args.seed),
           "--pos-noise-km", str(args.pos_noise_km),
           "--vel-noise-km-s", str(args.vel_noise_km_s)]
    if args.stress_test: cmd.append("--stress-test")
    subprocess.run(cmd, check=True)
    converted = out / "orbital_elements.csv"
    stats = convert(public, converted)
    result = run(converted, Path(str(public) + ".truth.jsonl"), out / "results.json", seed=args.seed)
    print(json.dumps({"conversion": stats, "results": result}, indent=2))


if __name__ == "__main__":
    main()
