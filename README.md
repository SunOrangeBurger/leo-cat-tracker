# LEO Catalog Identity Matching

A simulated ground-station object-catalog tracker for LEO satellites,
debris, and rocket bodies: synthetic orbital population -> identity
embedding -> fast distance gate -> three-way GNN second opinion -> live
predict/match/update tracking loop.

This is a rebuild of an earlier version of this system whose GNN second
opinion measured well under 1% live overturn precision despite ~90%+
precision on its own held-out split. Section "Design rationale" below is
the full write-up of what was wrong and why; this section is the
quickstart for the system as rebuilt.

## Directory layout

```
.
├── README.md
├── requirements.txt
├── data_generator.py                       synthetic LEO population -> leo_catalog.csv
├── orbital_physics.py                      two-body + J2 + drag propagation (numeric & analytic)
├── identity_embedding.py                   13-dim identity vector + physics-residual feature
├── leo_ann_index.py                        periodically-rebuilt ANN candidate-pool index
├── catalog_gnn.py                          three-way (KNOWN/AMBIGUOUS/NOVEL) GNN matcher
├── train_catalog_gnn.py                    offline trainer for catalog_gnn.py
├── ground_station_tracking_simulation.py   the live predict/match/update loop
├── compare_live_vs_training.py             live-vs-training distribution diagnostic
├── check_catalog_against_embeddings.py     standalone catalog/embedding-store cross-check
└── outputs/                                leo_embeddings.pt, catalog_gnn.pt, logs (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the synthetic LEO population
python data_generator.py -n 20000 -o leo_catalog.csv --seed 42

# 2. Build the identity-embedding store (the ground station's pre-known
#    catalog -- EMBED_FRACTION of the population, default 50%)
python identity_embedding.py

# 3. Train the catalog GNN offline (harvests live-loop training data --
#    see catalog_gnn.py; takes several real minutes at the defaults, not
#    seconds -- known-in-pool near-misses are genuinely rare (see
#    catalog_gnn.harvest_live_loop_examples' docstring) so the harvester
#    keeps simulating past the novel quota until it finds enough of them.
#    Run whenever leo_catalog.csv or outputs/leo_embeddings.pt change)
python train_catalog_gnn.py

# 4. Run the tracking simulation
python ground_station_tracking_simulation.py

# Optional: verify the live GNN invocation distribution still matches what
# it was trained against (see "Design rationale" section 5, item 5)
python compare_live_vs_training.py

# Optional: quick sanity check that every catalog object is (or isn't)
# present in the embedding store, with lookup timing stats
python check_catalog_against_embeddings.py
```

If `outputs/catalog_gnn.pt` doesn't exist yet, or was trained under
different gate thresholds / distance weights than the current run,
`ground_station_tracking_simulation.py` disables GNN consultation for that
run rather than silently retraining -- rerun `train_catalog_gnn.py`
explicitly. This is deliberate (see "Design rationale" and
`train_catalog_gnn.py`'s docstring): training touches a full harvested
live-loop run and shouldn't happen as a side effect of starting a sim run.

## Config knobs worth knowing about

- `data_generator.py`: `-n` (population size), `--seed`.
- `identity_embedding.py`: `EMBED_FRACTION` (fraction of the population
  the ground station starts out knowing), `--cold-start` for 0%.
- `ground_station_tracking_simulation.py`: `SIMULATION_DURATION_SECONDS`,
  `ACCELERATION_FACTOR`, `USE_GNN_CONFIRMATION`, `UPDATE_CONFIRMATIONS_REQUIRED`.
  `DRIFT_TOLERANCE`/`MARGIN_REQUIRED` are `None` by default (auto-calibrated
  from the catalog's own measured density at startup, and periodically
  recalibrated as the active catalog changes) -- only hardcode these if you
  know what scale you're targeting.
- `train_catalog_gnn.py`: `--n-examples`, `--epochs`, `--false-known-penalty`.

---

# Design rationale

This section explains why the previous GNN-matcher design failed, what we
measured to prove it, and the architecture that replaces it. It's meant to
be readable end-to-end by someone who has never seen this codebase, and
precise enough that the accompanying implementation (the files listed
above) can be built directly from it.

## 1. What this system does

A ground station repeatedly detects objects in LEO (satellites, debris,
rocket bodies). Each detection is a noisy orbital-element measurement with
no identity attached. The system has to decide, per detection:

- **Is this an object we already have a catalog entry for?** If so, which
  one, and update its stored belief.
- **Is this an object we've never seen before?** If so, register it as a
  new catalog entry so future detections of it can be recognized.

The catalog starts partially populated (some fraction of the true
population is pre-known) and grows over the run as new objects are
confirmed. This is fundamentally an **open-set re-identification problem**
under drift: every object's true orbital elements evolve continuously
(drag decay, J2 precession), so "the same object" never produces the same
measurement twice, and "a different object" can briefly look similar to
another by coincidence.

## 2. What we built first, and what it got right

### 2.1 Identity embedding

Objects are embedded into a 13-dimensional vector space
(`identity_embedding.py`) built from orbital elements and physical
properties, deliberately excluding true anomaly (the fast-changing
position-along-orbit, which is not identity). Angles are cyclically
encoded (sin/cos), continuous quantities are log-transformed and
z-scored, object type is one-hot. Per-dimension distance weights correct
for the fact that some raw dimensions carry much less discriminating
power once z-scored (e.g. inclination is drawn from only 5 tight clusters
in the synthetic catalog, so same-cluster objects look near-identical on
that dimension regardless of true identity).

**This part works and should be kept.** It's a reasonable, physically
motivated identity representation.

### 2.2 Fast distance gate

A cheap accept/reject gate: propagate the catalog's stored belief to the
current time, embed, compute distance to the 5 nearest neighbors, and
accept if the nearest is within an auto-calibrated `drift_tolerance` AND
meaningfully closer than the 2nd-nearest (`margin_required`). Both
thresholds are derived from the catalog's own measured **within-cluster**
nearest-neighbor density (not a flat global percentile — same-cluster
pairs are what actually collide, and a global percentile is dominated by
well-separated cross-cluster pairs that mask this).

**This part works well.** In the final measured run, the gate alone
handles the large majority of detections correctly (91%+ accept rate,
most of it correct).

### 2.3 ANN performance layer

A periodically-rebuilt approximate-nearest-neighbor tree
(`leo_ann_index.py`) replaces brute-force O(N) scanning per detection,
amortizing the cost of propagating and re-embedding the whole catalog.
Candidates are pulled from the tree (oversized pool, ~40) plus a small
overflow buffer of recently-added/dirtied entries, then the small
candidate set is re-propagated and re-ranked exactly. This is a pure
performance optimization and is orthogonal to the matching-quality
problem below — **keep it, tune it later.**

### 2.4 GNN second opinion — this is what failed

When the fast gate rejects, a GNN looks at the query plus its 5 nearest
candidates (fully connected, 6-node graph) and makes a more informed
accept/reject call. If it also rejects, the object is registered as new.

## 3. What went wrong, and how we found it

### 3.1 The measured failure

Across multiple full retrains, under progressively "fixed" training data
distributions, **live GNN overturn precision stayed at 0.18%–0.94%** —
i.e. when the GNN overturned a gate-rejection to "known," it was right
well under 1% of the time. This was true even for a checkpoint that
measured **~90%+ precision on its own held-out validation split.**

### 3.2 Why every "fix" attempt failed to move the number

Every diagnostic tool that existed before this investigation
re-evaluated the model against its own training generator's synthetic
output. This is circular — it can only ever confirm the model generalizes
*within* the synthetic distribution. It cannot detect a live/train
distribution mismatch, because it never looks at a live run.

We built `compare_live_vs_training.py`, which logs every real GNN
invocation during an actual tracking-sim run (candidate distances, margin,
catalog size, ground-truth correctness) and compares that distribution
against a fresh training batch. This is the first tool that looks at
reality instead of re-checking the training data against itself.

**Result:** across a full run, **99.6% of GNN invocations were for
detections whose true object was not present in the candidate pool at
all** — genuine first-sightings of objects not yet in the catalog, not
ambiguous near-misses. Of the confident-wrong overturns specifically,
**0.0% had the true object anywhere in the candidate pool.** The GNN
wasn't losing close calls — it was being asked to pick the right answer
from 5 candidates when the right answer structurally wasn't one of them,
over and over, and any confident answer to that question is guaranteed
wrong.

### 3.3 The distance-gate fix, and why it also failed

The obvious next fix: only consult the GNN when the gate's nearest
candidate is a *plausible* near-miss (within some multiple of
`drift_tolerance`), not on every reject. We implemented this
(`GNN_PLAUSIBILITY_MULTIPLE`), and it helped overall system metrics
(mismatch rate roughly halved) but **did not fix GNN precision** (0.18%
after tightening, actually worse than before).

We then measured, directly, the known-vs-novel `min_dist` (nearest
candidate distance) percentile distributions from the training generator
itself:

```
KNOWN examples (n=2736) min_dist (unweighted), as a multiple of drift_tolerance:
  p50 = 0.02x   p75 = 0.65x   p90 = 1.57x   p95 = 2.23x   p99 = 3.51x   p100 = 8.06x

NOVEL examples (n=1264) min_dist (unweighted), as a multiple of drift_tolerance:
  p1 = 1.37x    p5 = 1.79x    p10 = 2.01x   p25 = 2.46x   p50 = 3.00x
```

**Known p95 (2.23x) sits above novel p1 (1.37x).** These distributions
overlap across almost their entire range. There is no distance threshold,
at any multiple, that cleanly separates "genuine match under drift" from
"novel object that happens to be nearby." This is not a tuning problem —
the data proves no threshold on this feature works.

### 3.4 Root cause, stated precisely

This is a property of the **catalog's geometry**, documented already in
`identity_embedding.py`'s own comments and confirmed by the percentile
data: inclination is drawn from only 5 narrow clusters, ~90% of
eccentricity sits in one narrow band, and RAAN/argument-of-periapsis
precess in lockstep for two objects sharing an (a, i) cluster — meaning
an initially-close *unrelated* pair stays close indefinitely (a
birthday-paradox effect: with ~400 objects per inclination cluster, close
initial pairs are common). **Two structurally unrelated objects can and
regularly do sit at the same embedding distance as a genuine drifted
match.** Raw identity-vector distance, at the single-snapshot level, does
not contain enough information to resolve this ambiguity. No amount of
retraining a classifier that only sees distance/margin as input features
can fix that — the bottleneck is the features, not the model.

A second, independent problem compounds this: **the GNN had no memory.**
It saw a static 6-node graph with no information about *when* each
candidate's belief was last corroborated, how many times it's been
confirmed, or whether the query is physically consistent with that
candidate's own trajectory model over time — all of which the live system
already tracks (`cat_epoch`, `cat_confidence`) and simply never passed to
the classifier.

A third problem: **training data was never realistic about the live
task's actual shape.** The old generator drew one fresh i.i.d. "novel"
example against a clean, static, full-size catalog per draw. Live, a
genuinely novel object is repeatedly observed across many detections, each
time compared against a **growing, periodically-stale** ANN-pooled
candidate set that structurally cannot contain it (because it isn't
registered yet). These are different distributions of *task difficulty*,
not just different scale.

### 3.5 Why we didn't just keep patching

Three independent, increasingly aggressive fixes (retrain under corrected
distribution → add a plausibility distance gate → tighten that gate)
produced no meaningful improvement in the one number that mattered
(overturn precision). At that point, further threshold tuning is not
proportionate — the honest conclusion is that this feature set cannot
answer this question, and the whole GNN needed to be rebuilt around
different inputs, a different training-data process, and a different
output contract.

## 4. The new design

### 4.1 Output contract: three-way, not binary

The old GNN produced a single binary accept/reject per gate-rejected
detection. This forces a confident guess even when the honest answer is
"not sure yet" — and because false-accepts corrupt catalog state (belief
gets EMA-blended into the wrong slot, poisoning future matches), a forced
guess is the worst possible failure mode under uncertainty.

The new GNN (`catalog_gnn.py`) outputs one of three states per detection:

- **KNOWN** — confident match to one specific candidate. Proceed exactly
  as the old "gate accept" / old "GNN accept" path did: corroboration
  gate, then EMA belief update.
- **NOVEL** — confident this object isn't any of the candidates.
  Register as new, exactly as before.
- **AMBIGUOUS** — not confident either way. **Purely defer**: no belief
  write, no registration, no catalog mutation of any kind. The detection
  is logged as "not known" for this pass. The *next* detection of the
  same physical object re-enters the full pipeline fresh (gate → GNN)
  with no memory of this deferral. This is the safest possible behavior
  under genuine uncertainty: it costs nothing but a delayed
  registration/update, versus the old failure mode which actively
  corrupted state.

This directly reflects the asymmetric cost structure already documented
in this codebase: a false KNOWN is expensive (corrupts a slot, compounds
over future detections); a false NOVEL is bounded (a redundant
registration, self-correcting via the existing-slot-recovery path); doing
nothing under real ambiguity costs almost nothing. The three-way contract
lets the model spend its confidence budget honestly instead of being
forced into a binary it isn't equipped to answer reliably.

### 4.2 Features: give the model information distance can't provide

The old feature set per candidate was: 13-dim identity vector + raw
query-candidate distance. We keep the identity vector but add:

**a) Corroboration history** (already tracked live, never reached the
model before):
- `cat_confidence[j]` — number of consecutive corroborated updates this
  slot has received. A slot corroborated 20 times is a well-established
  identity; a slot corroborated 0-1 times is much more likely to be a
  provisional/uncertain registration, or even a prior false accept that
  hasn't yet been caught.
- `time_since_last_update[j]` (real seconds since `cat_epoch[j]`) — a
  slot updated 30 seconds ago carries a tighter belief than one updated 6
  hours ago, independent of raw distance.

**b) Physics-consistency residual** — the feature genuinely orthogonal to
embedding-space distance (`identity_embedding.physics_consistency_residual`).
For each candidate, propagate that candidate's *own* stored elements
forward using the deterministic secular model to the query's exact
detection time, then compute the residual between that propagation and
the query's raw measurement, **decomposed by element** (Δa, Δe, Δi,
ΔRAAN, Δargp) rather than collapsed into one scalar. Two objects can sit
at the same embedding-space distance at a single instant while having
completely different underlying elements that happen to alias into
similar-looking angles/log-scaled magnitudes — the element-wise residual,
not just distance, exposes that. This is information the raw identity-
vector distance cannot represent by construction (it's normalized/
cyclically-encoded specifically to *not* carry raw element magnitude), so
it gives the model a genuinely different axis to separate on, not a
rescaled version of the same axis.

**c) Local catalog density around the candidate**
(`LiveCatalogANNIndex.local_density`) — how many other catalog objects
sit within some radius of this candidate right now. This directly
measures "is this candidate in a birthday-paradox-prone cluster" —
exactly the condition that produces deceptive near-collisions between
unrelated objects. A candidate in a dense cluster should require more
corroborating evidence (physics residual, corroboration history) before
being trusted; a candidate in a sparse region can be trusted on distance
alone much more readily. This feature lets the model learn to be
adaptively skeptical rather than applying one fixed threshold everywhere.

### 4.3 Training data: harvested from a simulated live loop, not i.i.d. draws

The old generator drew independent examples: fresh random `dt`, fresh
random novel object OR fresh random known-object-plus-noise, always
compared against a clean full-size catalog. This cannot represent the
live task's actual shape (growing catalog, ANN staleness, repeated
observations of the same not-yet-registered object).

`catalog_gnn.harvest_live_loop_examples()` runs an actual lightweight
simulation of the matching loop — same growth dynamics, same periodic ANN
rebuild staleness, same repeated-failed-lookup structure for objects not
yet registered — and harvests real `(candidate_set, label)` pairs directly
from that simulation's actual decision points (specifically: every
gate-reject whose nearest candidate is a plausible near-miss, mirroring
exactly when the live system would consult the classifier), across a
range of catalog sizes/ages so the model sees the task at multiple points
in its lifecycle, not just at the untouched starting catalog size.

### 4.4 Loss / calibration: asymmetric, matching real costs

Confident-wrong-KNOWN is penalized more heavily than confident-wrong-NOVEL
in the loss (`catalog_gnn.asymmetric_bce_loss`). The AMBIGUOUS band is not
a third label to predict directly — it emerges from calibrated confidence
thresholds on the KNOWN/NOVEL output (two thresholds, not one: a detection
must clear a confidently-KNOWN bar to accept, and clear a
confidently-NOVEL bar to register; anything between is AMBIGUOUS by
construction — see `catalog_gnn.calibrate_confidence_thresholds`). This
keeps the model's actual learning target simple (calibrated probability of
"this is the match") while making the operating decision three-way.

### 4.5 What stays the same

- The 13-dim identity embedding (§2.1) — kept as the base representation,
  extended with the new features above, not replaced.
- The fast distance gate (§2.2) — kept as the first-pass cheap filter;
  the GNN is still a second opinion for what the gate rejects, not a
  replacement for it.
- The ANN performance layer (§2.3) — orthogonal to this problem, kept as
  a pure performance optimization (`LiveCatalogANNIndex.local_density`
  was added to it for the new density feature, but its indexing/pooling
  behavior is unchanged, and it still indexes pure identity distance, not
  the richer per-candidate feature set — that only makes sense once you
  have a specific query to compare against).
- Ground-truth-only diagnostics (mismatch rate, ANN recall, RMSE, catalog
  completeness/redundancy) — kept in `ground_station_tracking_simulation.py`.

### 4.6 What's explicitly deferred (kept realistic, not overcomplicated)

- Multi-ground-station cross-referencing / sensor geometry effects.
- Non-secular perturbation sources beyond J2 + exponential-atmosphere
  drag (already the model in use).
- Adversarial or maneuvering objects (station-keeping burns, etc.) —
  the current physics model assumes ballistic/drag-only evolution.

These are documented here as known simplifications so a future
contributor doesn't mistake "not modeled" for "not considered."

## 5. Implementation map

1. **`identity_embedding.py`** — the 13-dim identity vector (unchanged)
   plus `physics_consistency_residual()`, a separate, composable
   per-candidate feature function, so the base identity embedding stays
   reusable on its own (e.g. for the ANN index, which keeps using pure
   identity distance).
2. **`catalog_gnn.py`** — the three-way graph classifier, the
   live-loop-harvesting training data generator, asymmetric loss, and the
   two-threshold KNOWN/AMBIGUOUS/NOVEL decision logic.
3. **`train_catalog_gnn.py`** — standalone trainer, kept separate from the
   simulation's hot path, same load-only-with-stamp-check contract for the
   simulation to consume.
4. **`ground_station_tracking_simulation.py`** — consumes the three-way
   decision: KNOWN → existing corroboration/EMA path, NOVEL → existing
   registration path, AMBIGUOUS → purely defer (log as not-known, no
   mutation).
5. **`compare_live_vs_training.py`** — the standing verification method
   for any future regression: log real invocations, compare against the
   training distribution.

Each file documents, at the top, which specific measured failure it exists
to fix, so a future reader can trace every design decision back to the
evidence that produced it.
