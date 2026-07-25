# Learned Association & Cataloguing for LEO Space Objects

*A GNN-based alternative to JPDA/MHT for dense-regime satellite catalog maintenance — design, validation sprint, failure modes found, and fixes made.*

---

## 1. The problem this project is trying to solve

Space Situational Awareness systems have to solve two linked problems continuously, for every object in Low Earth Orbit:

- **Identification** — given a new sensor measurement, is this a previously catalogued object, and if so, which one?
- **Cataloguing** — if it isn't a known object, or isn't confidently matched to one, when should it be promoted to a new tracked entry, and how should the catalog's state estimate update?

The standard classical answer to the identification half is **JPDA** (Joint Probabilistic Data Association), with **MHT** (Multiple Hypothesis Tracking) as a stronger but more expensive alternative. Both are well-established. Both are also known to degrade combinatorially as object density increases — and that's precisely the regime that matters most right now, as large constellations pack more objects into tighter, near-identical orbital shells.

The hypothesis this project set out to test: **a learned, neighborhood-aware association system — vector embeddings plus a graph neural network — can match or beat JPDA/MHT specifically in dense, closely-spaced-object regimes**, by reasoning only over small local neighborhoods instead of the full joint hypothesis space, and by learning regime-specific drift/noise patterns that a fixed, hand-tuned probabilistic model doesn't capture.

This was written down as a genuine, falsifiable claim, not "beat everything at everything": beat JPDA/MHT specifically on track-continuity, false-track rate, and computational scaling, in dense-object scenarios, measured against synthetic ground truth. Three explicit outcomes were pre-agreed before any testing started, precisely so the result couldn't be argued into whatever was convenient after the fact:

- **Green light** — the pipeline beats JPDA by a real margin in the dense regime, and retrieval recall is high enough that the GNN is clearly adding value on genuinely hard cases.
- **Yellow light** — results are roughly tied, or the GNN only wins on easy cases and still loses to JPDA on the ambiguous ones. This specifically would mean the core approach needs rethinking, not more tuning.
- **Red light** — retrieval recall alone is poor, meaning the true match often isn't even in the candidate set. That would mean the problem is upstream of the GNN entirely.

## 2. Why not just do full JPDA/MHT better

A Rust-based simulator with JPDA already existed before this project's current phase, complete with SGP4 propagation and Gaussian measurement noise injection. It had two known, load-bearing problems:

- Running with SGP4 (real orbital drift) enabled significantly degraded simulator performance, so testing was often done with SGP4 disabled — which silently removes the exact phenomenon (differential precession and drag causing ambiguity) that makes the dense regime hard in the first place. That's a correctness risk, not just a speed inconvenience: a fast-but-driftless simulator produces baseline numbers that don't mean anything for the real problem.
- Ground truth was written to file, but semi-hardcoded into the simulator's own visualization layer rather than existing as an independent, structured stream — meaning ground truth, measurements, and JPDA's output needed to be decoupled into three clean streams before any of it could be reliably exported for analysis.

Rather than sink engineering time into fixing and extending an existing simulator whose core idea hadn't even been validated yet, the decision was to build a **deliberately throwaway-quality, fast-to-iterate Python validation sprint** first: a standalone data generator, a retrieval stage, a minimal GNN, and enough of a harness to tell — cheaply — whether the idea was worth the investment of fixing the Rust simulator at all. Everything in this repository is that validation sprint.

## 3. Architecture: why two stages, not one model

The system is deliberately split into two stages with very different computational profiles, and collapsing them into a single model was considered and rejected:

**Stage 0 — physics, not learned.** Every object's orbital state is five Keplerian elements plus derived features. Orbital drift (J2 oblateness and atmospheric drag) is deterministic, well-understood physics, so it's propagated analytically — not learned by any model — and every measurement is propagated to a common epoch before any distance comparison happens. Comparing un-propagated states from different times would otherwise conflate "different object" with "same object that has naturally precessed," which is exactly the kind of error that spawns phantom tracks or causes false merges.

**Stage 1 — vector retrieval, runs on every measurement.** Each object becomes a feature vector (angles handled via sin/cos to respect wraparound, size/RCS down-weighted as the noisiest signal), normalized so no dimension dominates by unit-scale accident, and looked up via an approximate-nearest-neighbor index. Most measurements should resolve here alone: if retrieval returns one dominant, unambiguous match, a simple state update suffices and Stage 2 never runs. This is what keeps the system fast — the expensive stage stays dormant unless something is actually ambiguous.

**Stage 2 — a GNN, invoked only on ambiguity.** Only runs when Stage 1's retrieval gate rejects a candidate that's still plausibly close. It operates on a *small local neighborhood* around the candidate — never the full catalog graph — which is the specific architectural reason this should scale better than JPDA/MHT as density increases: cost per decision doesn't grow with catalog size, because the GNN never looks at the whole catalog, only a handful of nearby candidates.

A handful of specific failure modes were designed against up front, before any code was written, because getting them wrong would have quietly invalidated everything downstream: timestamp is a propagation trigger, not a raw embedding axis; angles need circular-aware encoding; drag parameters have to be *sampled*, not hardcoded, since solar activity changes drag-driven decay by an order of magnitude between solar min and max; and a GNN doing full-catalog message passing on every measurement would throw away the entire efficiency premise of the design.

## 4. The validation sprint's first real result — and its hidden flaw

The validation sprint built exactly what was planned: a synthetic catalog generator, an embedding stage, an ANN-backed retrieval gate, and a GNN (then called `gnn_matcher.py`, trained by a standalone script `train_gnn.py`) as the second-opinion classifier for whatever the gate rejected but couldn't confidently rule out.

The problem was in how that GNN was validated. Every diagnostic available at the time — including the original `diagnose_gnn.py` — worked the same way: generate a fresh batch of synthetic examples with `generate_training_examples()`, run the trained model against them, and report accuracy. This is **circular by construction**. `generate_training_examples()` was also what produced the *training* data. A diagnostic that re-checks a model against its own training distribution can only ever tell you the model generalizes within the shape of data it already knows how to produce — it structurally cannot detect a live/train mismatch, because it never looks at anything the live system actually encountered.

The fix was `compare_live_vs_training.py` — the first diagnostic in this project's history to look at an actual live run instead of resimulating the model's own training assumptions. Its method was direct: log every real GNN invocation during a tracking-sim run (`outputs/gnn_live_invocations.json`), then compare that live distribution of match distances and margins against what `generate_training_examples()` would produce fresh, side by side.

**What it found was the circularity concern made concrete.** Live min_dist and margin came out *smaller* than training's, and the script's own diagnosis of why was direct: catalog growth. The synthetic catalog grew from roughly 10,000 to over 19,000 objects over the course of a run, producing a denser, harder candidate pool than `generate_training_examples()`'s fixed-size fresh scan could ever simulate — because that function always sampled from a static base catalog size, never one that had grown the way the live system's actually does. The GNN was being trained on a candidate-density regime that didn't match the one it would actually be deployed into, and the only reason anyone found out was because someone finally checked.

## 5. The fix: stop simulating a fiction, replay the real loop

The response wasn't to patch `generate_training_examples()` — it was to delete the whole approach it represented. The current harvester, `catalog_gnn.harvest_live_loop_examples()`, does something categorically different: it **replays a lightweight version of the actual tracking simulation's matching loop** — the same growing catalog, the same periodically-stale ANN index, the same "a gate-reject is the moment the real system would consult the classifier" structure — and harvests a training example at every point the live system would genuinely have invoked the GNN. There is no separate, fixed-size, disconnected "training distribution" anymore. Training data *is* a faithful replay of deployment conditions, including the catalog-growth dynamic that broke the old approach.

This is also why `compare_live_vs_training.py` survives into the current codebase essentially unchanged in purpose, even though every module it imports has been renamed and rewritten (`leo_catalog_embed` → `identity_embedding`, `gnn_matcher` → `catalog_gnn`, `gnn_matcher.pt` → `outputs/catalog_gnn.pt`, `generate_training_examples()` → `harvest_live_loop_examples()`). The diagnostic's job — catch a live/train mismatch that only a real run can reveal — is exactly as necessary now as it was when it first caught the catalog-density problem. The current version's own docstring states this as a standing obligation, not a one-off fix: re-run it after any change to the harvesting logic, the gate thresholds, or the distance weights, because a change to any of those is exactly the kind of thing that could reopen a live/train gap.

## 6. Second failure, found by actually looking at a training curve

Once the harvester was fixed, the next thing that broke wasn't the data — it was how the model was trained on it. A full pipeline run showed textbook overfitting: train_loss fell smoothly toward zero while val_loss bottomed out around epoch 100 and then climbed for the rest of the 250-epoch run, ending up higher than where it started. `val_recall(known)` — the model's ability to correctly recognize an already-catalogued object — degraded over the same stretch, jumping between exact fractions like 1/6, 2/6, 3/6.

Those exact fractions were the real tell. They meant the validation set being used to compute `val_recall(known)` had something like six known-in-pool examples in it, total. The harvester's own default floor for how many known-in-pool examples to collect — `max(20, n_examples // 400)` — was calibrated for a case that turned out to be far too small once split into train/val: a genuine known-in-pool near-miss (the gate rejecting an object that actually is already registered) is a real but rare failure mode by design, since it requires the fast gate to fail on an object it should normally catch easily. With so few examples to validate against, `val_loss` was being computed from noise, and the training loop — which had no early stopping or best-checkpoint logic at all — simply kept training straight through the point where the model stopped generalizing and saved whatever weights happened to be sitting there at epoch 250.

Two changes fixed this, and both are now permanent defaults rather than flags someone has to remember:

- `catalog_gnn.py`'s `train_model()` now tracks the best-val_loss checkpoint seen during training (checked every 25 epochs) and restores those weights before saving, instead of saving whatever the final epoch happened to produce. The saved checkpoint now also records which epoch it came from and what its val_loss was, so this is auditable rather than assumed.
- `train_catalog_gnn.py`'s `--min-known-examples` default was raised from `None` (which fell through to the harvester's `max(20, n_examples // 400)` floor) to `400`. A plain, no-flags invocation now collects a large enough known-in-pool validation set that `val_recall(known)` and `val_loss` are a stable signal instead of a coin flip over six examples.

After both fixes, the same kind of run showed val_recall(known) climbing to roughly 0.9 and holding there through epoch 250, with val_loss and train_loss tracking each other closely the whole way — the difference between a model that's actually calibrated and one that only looked calibrated because nobody had checked closely enough.

## 7. What the current system actually measures, and what it honestly can't yet

A full-scale run against a 20,000-object synthetic catalog, evaluated end-to-end (catalog generation → embedding → GNN training → 60-simulated-second tracking run → cross-checks), currently reports:

- **100% correct matches, 0% mismatches**, across 56,841 detections.
- **99.9% catalog completeness** and **100% ANN candidate recall** — the retrieval stage is finding the true match in its candidate pool essentially every time it's supposed to.
- **82.4% of detections resolved by the fast gate alone**, with the remainder split between genuinely new registrations (17.6%) and a small residual sliver (0.1%) landing in the GNN's ambiguous/second-opinion path — consistent with the design goal that Stage 2 stays dormant for the overwhelming majority of traffic.

The one number that does **not** currently support a strong claim, and is reported here as exactly that rather than smoothed over: **GNN overturn precision was 0/6** in that run — every time the GNN overturned a gate rejection to call something KNOWN, it turned out to be wrong. This is not necessarily evidence the GNN doesn't work; a sample of 6 real events is nowhere near enough to conclude anything either way, and it's a direct, structural consequence of how rare a genuine known-in-pool gate failure is by design — the same rarity that caused the overfitting problem in Section 6, just showing up again at the live-deployment level instead of the training-validation level. The honest framing for this project's current state is: the training-side calibration of "is this object I've already catalogued" is now measured on a large, stable validation set and looks solid (≈90% recall); the live-deployment version of that same question has not yet been observed often enough in a real run to say anything with statistical confidence. A larger-scale, longer-duration run — planned for more capable hardware than was available for this phase — is the intended next step to close that gap, rather than any further tuning of the current small-scale results.

## 8. Evaluation harness

Because none of the individual pipeline scripts expose structured results — each one only prints to stdout, plus one JSON artifact from the tracking sim — a harness (`run_full_evaluation.py`) was built to run the full pipeline end to end, parse each stage's console output against regexes matched to that script's actual print statements, and render a fixed set of diagnostic plots: decision-outcome breakdowns, match-resolution paths, distance/margin distributions (linear and log-scale, since a well-calibrated system's distances cluster so close to zero that a linear histogram alone hides their internal shape), lookup latency over time, catalog growth, auto-calibrated gate-threshold drift, the GNN's own training curve, and the live-vs-training comparison described in Section 4–5, now automated rather than something that has to be run and read by hand.

The harness also runs what's genuinely parallelizable in the pipeline's dependency graph — `check_catalog_against_embeddings.py` only depends on the catalog and embeddings, not the GNN or the sim, so it runs concurrently with GNN training and the tracking sim rather than strictly after them — while being explicit that most of the pipeline is a true sequential chain (each stage's output is the next stage's input) that can't be parallelized without reading a file before it exists.

## 9. Repository layout

```
leo-catalog-tracker/
├── src/                          # the actual pipeline
│   ├── data_generator.py                     # synthetic Keplerian catalog generation
│   ├── identity_embedding.py                 # feature vectors + normalization
│   ├── orbital_physics.py                    # J2/drag propagation (Stage 0)
│   ├── leo_ann_index.py                      # ANN retrieval index (Stage 1)
│   ├── catalog_gnn.py                        # GNN second-opinion classifier (Stage 2)
│   ├── train_catalog_gnn.py                  # offline trainer entry point
│   ├── ground_station_tracking_simulation.py # the live matching loop
│   ├── compare_live_vs_training.py           # live-vs-training distribution check (Section 4-5)
│   └── check_catalog_against_embeddings.py   # catalog/embedding-store cross-check
├── run_full_evaluation.py        # orchestrates the full pipeline + renders all plots
├── data/leo_catalog.csv          # generated synthetic catalog (input)
├── outputs/                      # every generated artifact: checkpoints, logs, plots, results
├── requirements.txt
└── README.md                     # this file
```

All path constants across every script are anchored to the repository root via `pathlib`, so any script runs identically whether invoked from the repo root or from inside `src/`.

## 10. Running it

```bash
# full pipeline: generate catalog, embed, train GNN, run tracking sim,
# cross-check, compare live-vs-training, render all plots
python run_full_evaluation.py

# reuse an existing catalog/embeddings, only retrain + re-simulate
python run_full_evaluation.py --skip-generate --skip-embed

# re-render plots from a previous run's saved results, no pipeline re-run
python run_full_evaluation.py --plots-only
```

See `run_full_evaluation.py --help` for the full set of options (catalog size, harvester example count, epoch count, sequential-vs-concurrent stage execution).

## 11. Where this stands relative to the original success criteria

Section 1's pre-agreed criteria were retrieval recall and GNN value-add on genuinely hard cases, measured against JPDA/MHT. This repository validates the retrieval and cataloguing half of that pipeline in isolation — recall, completeness, and match integrity are all strong by the numbers in Section 7 — but does not yet include the JPDA baseline comparison the original design called for as the actual head-to-head test. What exists here is the foundation that comparison would run on top of: a working embedding space, a working ANN-backed gate, and a GNN second opinion whose training-time calibration is now solid and whose live-deployment calibration is honestly reported as not yet statistically measured. The next steps, in order, are a larger-scale run to get a meaningful sample on the GNN-overturn question, and then the JPDA baseline comparison this project was originally designed to be judged against.