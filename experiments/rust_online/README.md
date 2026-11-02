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

## Temporal candidate graph and CelesTrak-informed experiments

`temporal_association.py` encodes each candidate's last eight assigned public
sightings with a GRU, then exchanges messages among local candidate nodes using
attention. The current orbital-frame residual features are supplied to both the
GNN and a graph-free history-summary learner. Two neural ablations disable
messages or history. Training checkpoint selection uses validation observations.

`run_temporal_probe.py` compares those methods with the earlier orbital-frame
tabular scorer, the original local JPDA, and a stronger JPDA using empirical
assigned-history uncertainty in the orbital frame. The stronger JPDA selects
along-track inflation, gate, and clutter weight on validation. Learned scorers
receive that same selected candidate gate. This comparator estimates local
diagonal uncertainty; it is not a full covariance-propagating JPDA filter.

`celestrak_synthetic.py` resamples joint GP records from a frozen CelesTrak JSON
snapshot and uses SGP4 to generate TEME states. The inference predictor is
elliptic two-body propagation for every method. Sensor visibility, detection
probability and full-state Gaussian measurement noise are synthetic assumptions.
The snapshot supports orbital population structure, not learned sensor realism.
`--crowding 5` deliberately creates close neighbors around each sampled orbit
and is an engineered stress test. Ordinary sampling must also be reported.

```sh
.venv/bin/python experiments/rust_online/celestrak_synthetic.py \
  /path/to/frozen_active_gp.json /tmp/celestrak-seed-201 --seed 201 --objects 1000
.venv/bin/python experiments/rust_online/run_temporal_probe.py \
  /tmp/celestrak-seed-201 /tmp/temporal-201.json --seed 201 --kepler --births --training-births
.venv/bin/python experiments/rust_online/run_temporal_study.py \
  /path/to/frozen_active_gp.json /tmp/temporal-study --start-seed 201 --seeds 10 --births --training-births
```

`--births` enables the experimental four-scan predictive tracklet initiator.
Confirmed tracks enter the index with their actual public observation history.
Future assignment correctness is evaluated against the identity established by
the confirmation sightings, without allowing later labels to redefine that
identity. Purity and fragmentation must still be checked; a longer chain does
not guarantee safe initiation. `--no-update` supplies the frozen-state control.
Saved artifacts include checkpoints, initial states, scales, per-event test
scores, per-seed outcomes, selected JPDA configuration, and source checksums.

The historical calibration helper adds an absolute `1e-6` to score thresholds.
This can distort decisions when JPDA probabilities are near zero. The canonical
comparison therefore runs `audit_temporal_calibration.py` on frozen checkpoints:
it uses the next representable score above the validation cutoff for every
method, retunes the JPDA configuration, and performs complete online replays at
1%, 2%, and 5% validation false-merge targets. It never selects a deployable
threshold using test labels. `run_temporal_audit.py` applies this to all seeds.

```sh
.venv/bin/python experiments/rust_online/run_temporal_audit.py \
  /tmp/temporal-study /tmp/temporal-audited --start-seed 201 --seeds 10 --normalized
```

## Deployment limit

For one complete stream, `run_temporal_pipeline.py SOURCE OUTPUT --seed 101 --births`
trains with promotions and automatically performs the strict-cutoff,
Gaussian-normalized audit. It selects the common predictor from the manifest
or `--predictor`. The normalized baseline includes candidate covariance volume
in exact event weights and MAP fallback costs. Library threads are capped at
two; `--resume` on study/audit drivers skips completed seeds after interruption.
Use a fresh output directory when changing any experiment configuration.

The growing-training correction labels new pure teacher tracks only after a
public replay; mixed confirmations have no positive identity. During inference,
labels are absent from histories, retrieval, updates, and initiation. The final
reference results and their scope are recorded in `results/README.md`.

The two-sighting birth rule is **experimental**. On the crowded 3,000-object Rust stream it produced about 90% mean track purity, which is too low for unattended catalog growth. Use the defer-unknown mode for known-object association and route rejected observations to a separately validated tracklet-confirmation process. A production tracker still needs covariance-aware state estimation, raw sensor models, missed detections/clutter, longer arcs, and independent catalog validation.
