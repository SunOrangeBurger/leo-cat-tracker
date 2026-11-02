# Reference temporal comparison

The JSON files contain all ten crowded seeds (211–220), per-seed outcomes,
complete online operating curves, bootstrap summaries, and diagnostics for the
final growing-harvest GNN and Gaussian-normalized local JPDA audit. Absolute
workspace paths have been replaced with a logical generated-data path.

At matched retrospective test false-merge risk, GNN-minus-JPDA known recall is:

| Test risk | Difference | 95% seed-bootstrap interval | Bracketed seeds |
| --- | ---: | ---: | ---: |
| 2% | +2.02 pp | +0.48 to +3.62 pp | 10/10 |
| 5% | +4.10 pp | +1.84 to +6.62 pp | 10/10 |

These are exploratory, retrospective online-curve diagnostics. Primary
thresholds are selected on validation, not test labels. At the primary point,
GNN recall/false-merge rates are 5.24%/1.61%, versus 1.91%/0.72% for JPDA; that
point alone is a tradeoff. GNN crowded track purity is 80.57%, versus 77.22%.
The no-message control remains competitive. Autonomous catalog growth and a
unique need for graph attention are not established.

Reproduce the final crowded cohort from the repository root:

```sh
gzip -dc experiments/rust_online/data/active_gp_20261102.json.gz > /tmp/frozen_active_gp.json
.venv/bin/python experiments/rust_online/run_temporal_study.py /tmp/frozen_active_gp.json /tmp/temporal/crowded_growth_training --start-seed 211 --seeds 10 --objects 1000 --crowding 5 --births --training-births --jobs 2
.venv/bin/python experiments/rust_online/run_temporal_audit.py /tmp/temporal/crowded_growth_training /tmp/temporal/crowded_gaussian_final --start-seed 211 --seeds 10 --normalized --jobs 2
.venv/bin/python experiments/rust_online/run_temporal_audit.py /tmp/temporal/crowded_growth_training /tmp/temporal/crowded_gaussian_risk --start-seed 211 --seeds 10 --normalized --risk-extra --jobs 2
.venv/bin/python experiments/rust_online/finalize_temporal_results.py --root /tmp/temporal --cohort crowded_gaussian
```

Use `--resume` for an interrupted driver with the same configuration. CelesTrak
informs population structure; five-object crowding is deliberately engineered.
SGP4 generates truth and every method uses the same elliptic predictor. The
comparison concerns local association under shared retrieval and lifecycle,
not a fielded covariance-propagating JPDA tracker.
