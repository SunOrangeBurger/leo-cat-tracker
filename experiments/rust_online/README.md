# Rust-stream online association experiment

This directory contains the reproducible association probes used by the companion IEEE manuscript. The Rust simulator is a sibling checkout of `rust-leo-sim` on its `codex/noisy-observation-export` branch. The Python code reads a public JSONL stream of noisy Cartesian position/velocity measurements. A separate truth file is opened by the experiment wrapper only for warm-catalog construction, training/calibration, and scoring; `OnlineEngine` receives no truth labels.

## Setup

From the `leo-cat-tracker` repository root:

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
cargo build --offline --manifest-path ../rust-leo-sim/Cargo.toml
```

The two repositories must be checked out side by side. The exporter uses the simulator's `target/debug/leo_sim` binary.

## Reproduce a paired online study

```sh
.venv/bin/python experiments/rust_online/export_rust_streams.py \
  --output-dir /tmp/leo-rust-3000 --start-seed 51 --seeds 20 \
  --objects 3000 --sensors 24 --steps 80

.venv/bin/python experiments/rust_online/run_online_study.py \
  --source /tmp/leo-rust-3000 --output-dir /tmp/leo-online-defer \
  --start-seed 51 --seeds 20 --jobs 3 --no-births
```

The default mode defers unknown observations. For the experimental two-sighting birth policy, add `--experimental-births --birth-gate 1.5 --defer-radius-sq 9 --min-birth-observations 2`. The stronger local JPDA comparator uses exact dynamic-programming marginals for clusters with at most 12 detections and 32 tracks, then a one-to-one MAP fallback. The archived fixed-catalog benchmark keeps its earlier eight-object/greedy settings for exact historical reproduction. All online decision rules receive the same top-five candidate pool, warm catalog, state-update rule, and birth/defer policy. The tabular scorers and JPDA share a squared-residual gate of 25; the GNN selects its gate from validation data. Rejection thresholds are selected on the earlier validation interval to target 2% novel-to-seeded false merges; that target may drift on test data.

The full pipeline trains on 600–1,200 s, calibrates on 1,200–1,600 s, and tests at 1,600 s onward. The initial catalog is built from labeled detections before 600 s, which stands in for an existing validated operational catalog. The current proof of concept uses noisy **full Cartesian states**, circular two-body propagation, and a short 40-minute stream. It is not an angles-only sensor or an SGP4/TLE validation.

`online_study_summary.json` gives per-seed outcomes, seed-bootstrap intervals, exact paired seed sign-flip p-values, candidate recall, timings, and checksums of the study scripts. Every outcome matrix partitions all known and novel test detections. The unit tests in `tests/test_jpda_marginals.py`, `tests/test_online_catalog.py`, and `tests/test_rust_stream_bridge.py` check exact JPDA marginals, lifecycle accounting, and stream validation.

## Safety boundary

The two-sighting birth rule is **experimental**. On the crowded 3,000-object Rust stream it produced about 90% mean track purity, which is too low for unattended catalog growth. Use the defer-unknown mode for known-object association and route rejected observations to a separately validated tracklet-confirmation process. A production tracker still needs covariance-aware state estimation, raw sensor models, missed detections/clutter, longer arcs, and independent catalog validation.
