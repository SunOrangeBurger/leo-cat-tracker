# JPDA vs. GNN Pipeline: Benchmark Process Log

**Status as of this writing:** a real, isolated dense-regime result exists and
favors the GNN pipeline. One open question remains before treating this as a
complete PoC (see "What's still open" at the end).

---

## 1. Where this started

The repo (`leo-cat-tracker`) contained a fully working GNN-based satellite
tracking pipeline — data generation, identity embedding, GNN training, and a
tracking simulation — with results already saved in `outputs/eval_results.json`.
The README referenced JPDA (Joint Probabilistic Data Association) throughout
as the classical baseline this project was meant to be compared against and
beat.

**There was no JPDA implementation anywhere in the repository.** Not in the
current code, not in git history, not on any branch. The README's own final
section said as much — the JPDA head-to-head was documented as a "next step,"
not something already done. So the first finding was simple: the comparison
this project needed to make its central claim didn't exist yet.

## 2. Building a fair JPDA baseline

Rather than write a token/strawman JPDA implementation, the goal was a real
one: proper chi-square Mahalanobis gating, clustering of ambiguous
tracks/detections, full joint-event enumeration (the actual combinatorial
core of JPDA, per Bar-Shalom's formulation) with marginal association
probabilities, and covariance-inflating combined-innovation updates — the
textbook algorithm, not a nearest-neighbor gate wearing a JPDA label.

Critically, this baseline was built to consume **exactly** the same inputs as
the GNN pipeline: the same catalog CSV, the same seed, the same measurement
and process noise constants, and — after finding and matching a detail that
was initially missed — the same detection re-scheduling logic (every object
gets re-detected on each subsequent orbital pass, not just once). Getting
this matched was essential; without it, objects were only ever detected
once and JPDA never got the repeat detections it needs to demonstrate
anything at all.

A companion script, `compare_jpda_vs_gnn.py`, was built to load both
systems' result JSONs and print a fair, honest side-by-side — including two
built-in warnings designed to prevent the comparison from being
over-interpreted (see below).

## 3. First real run: parity, not proof

The first full, valid run (20,000-object catalog, 60 simulated seconds,
matching the pipeline's existing saved results) showed both systems
performing almost identically:

| | JPDA | GNN pipeline |
|---|---|---|
| Correct match rate | 100.0% | 100.0% |
| Mismatches | 0 | 8 |
| Catalog completeness | 100.0% | 99.4% |
| Wall time | ~132s | ~127s |

This looked like a wash, and the comparison script's own diagnostics
explained why: **JPDA's max simultaneous-candidate cluster size was 2 the
entire run**, and the GNN pipeline's own genuinely-ambiguous detection rate
was under 1%. Neither system was ever forced to arbitrate real ambiguity —
JPDA's actual weakness (combinatorial cost under crowding) never got
triggered, because the catalog's default object distribution simply doesn't
pack objects densely enough in orbital-element space for that to happen,
regardless of how many total objects are in it.

A secondary finding from this run mattered too: most of the raw "known
rate" gap between the two systems (64.5% JPDA vs. 81.7% GNN pipeline) turned
out to be a **gate/track-management design difference**, not evidence about
the GNN classifier itself — 99.99% of the GNN pipeline's known-matches came
through its ordinary fast gate, not its GNN "second opinion" path. Only 6–7
detections out of tens of thousands ever actually invoked the GNN classifier.
This distinction was kept explicit in all later reporting so the raw
known-rate numbers wouldn't be misread as "the GNN wins at association" when
that path was barely exercised.

## 4. Scaling up didn't fix it

The catalog was scaled to 60,000 objects (3x) under the same distribution,
on the theory that more objects would create more crowding. It didn't:
**even at 60k objects, max cluster size stayed at 2**, and the GNN pipeline's
ambiguous-consult rate stayed under 0.02%. JPDA turned out to be the
*faster* system at this scale (664s vs. 1,244s) — but the density-regime
check again confirmed neither system had been meaningfully tested, so this
result was correctly treated as inconclusive for the actual thesis, not as
evidence against it.

Along the way, this stage of the process also surfaced (and fixed) two
process errors worth recording, since they'd otherwise silently corrupt
results in a hard-to-detect way:

- **Catalog mismatch**: a regenerated 20k catalog did not reproduce the
  repo's original `data/leo_catalog.csv` byte-for-byte, meaning JPDA runs
  needed to be pointed at the *actual* checked-in catalog file, not a fresh
  regeneration, to be valid comparisons against the existing saved GNN
  results.
- **Hardcoded catalog paths**: `run_full_evaluation.py`'s own `catalog_path`
  variable is only used internally for its own data-generation step;
  `ground_station_tracking_simulation.py`, `identity_embedding.py`, and
  `check_catalog_against_embeddings.py` each independently hardcode
  `data/leo_catalog.csv`. Editing one script's path variable did nothing —
  the only reliable way to redirect the whole pipeline at a different
  catalog is to replace the file at that hardcoded path directly, and to
  delete any pre-existing `leo_embeddings.pt` / `catalog_gnn.pt` so stale
  artifacts aren't silently reused against the new catalog.

## 5. The real fix: an engineered dense-shell catalog

The core insight: `data_generator.py`'s distribution spreads objects across
disjoint inclination clusters and a wide semi-major-axis band, so local
neighbor density stays roughly constant as N grows — you get more clusters
of the same sparseness, not denser ones. Adding more objects under that same
distribution was never going to produce ambiguity, at any scale.

A new script, `dense_shell_catalog_generator.py`, was built to solve this
directly: it generates the normal sparse background population (unchanged),
plus one or more "shells" — clusters of objects deliberately packed into a
narrow band of semi-major axis and inclination, mimicking how a real
satellite megaconstellation shell is actually built (hundreds of satellites
sharing nearly identical altitude and inclination by design). Every other
object property (type, size, RCS, eccentricity, etc.) is sampled identically
to the background population — density is the only variable being changed,
so nothing else confounds the comparison. Shell objects are name-tagged
(`_shell{k}_`) so downstream scripts can isolate their metrics from the
diluting sparse background.

Before committing to a full run, the shell's density was validated directly
against JPDA's actual gate window (not just eyeballed) — an early attempt
was tuned too aggressively and produced a degenerate "first contact" case
where JPDA got essentially everything wrong immediately, which wasn't
representative of a real crowding scenario. A moderate configuration was
tuned instead (mean ~23, max ~38 genuinely gate-level neighbors per shell
object) and validated on a small-scale test before scaling up.

`jpda_baseline.py` was extended with a shell-aware breakdown: whenever
catalog object names contain `_shell`, the script now separately reports
shell-only correct-match rate, mismatch rate, cluster size, and latency
percentiles, isolated from the sparse background average. This is a no-op
on catalogs without shell objects.

## 6. The dense-shell result

A 23,000-object catalog (20,000 sparse background + a 3,000-object dense
shell) was run through both systems, using the corrected catalog-swap
procedure from Section 4.

**JPDA, shell-only:**
- Correct match rate: 92.7% (down from 100.0% in every sparse run)
- Mismatch rate: 7.3% (up from 0.0%)
- Mean lookup latency: 102,496us (vs. ~2,000–4,000us in sparse runs — a
  ~47x increase)
- p99 latency: 137,397us (vs. ~5,000–8,000us — a ~27x increase)
- Max cluster size reached: 5 (still under JPDA's own 8-way exact-enumeration
  cap, so this is authentic full JPDA math degrading, not a degraded
  fallback mode masking the result)

**GNN pipeline, same catalog, whole-run numbers:**
- Correct match rate: 100.0%
- Mismatch rate: 0.03% (17 mismatches, vs. JPDA's 336)
- Wall time: 121.2s (vs. JPDA's 820.1s — JPDA was ~6.8x slower)
- Catalog completeness: 100.0% (vs. JPDA's 99.4%)

This is the first result in the whole process that actually speaks to the
project's central hypothesis: JPDA visibly degrading in both accuracy and
latency under genuine crowding, on the same detection stream the GNN
pipeline handled cleanly and substantially faster.

## 7. What's still open

The GNN pipeline's own diagnostics on this run show its classifier
("second opinion") component was invoked 1,444 times in the dense shell —
and **all 1,444 resolved to NOVEL**, none to KNOWN or AMBIGUOUS. That means
the pipeline's overall architecture (gating + track management + GNN
fallback) handled the dense scenario well, but the GNN classifier itself was
never actually caught in the act of correctly resolving a genuine
multi-candidate tie — it was consulted, but every time it was asked, the
right answer happened to be "none of these," not "pick between these two."

This is a meaningful distinction for how the PoC claim should be framed:
- **Supported by current data:** "the full pipeline (of which the GNN
  classifier is one component) outperforms JPDA under realistic
  constellation density, on speed, accuracy, and completeness."
- **Not yet directly demonstrated:** "the GNN classifier itself correctly
  breaks genuine association ties that would otherwise cause an error" —
  because in this run it was never actually put in that exact position and
  observed to get it right.

Next investigative step, not yet done: understand why 0/1,444 shell-region
GNN consults resolved to KNOWN/AMBIGUOUS despite JPDA hitting real,
measurable ambiguity in the same region — either by inspecting
`gnn_live_invocations.json` directly, or by tuning the shell density/gate
thresholds so genuine ties are more likely to reach the GNN classifier and
be observed being resolved correctly.