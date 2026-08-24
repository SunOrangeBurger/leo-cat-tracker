# Density-Aware ANN Rebuild: Root Cause and Fix

**Status as of this writing:** the two implementation bugs described in
Sections 7 and 8 (off-by-one, radius mismatch) are real, fixed, and
confirmed working — the density trigger fires and does change rebuild
behavior. But Section 11 found a **third bug**, in the diagnostic
itself, not the rebuild logic: the live `true_object_in_pool` field
measures membership in the GNN's 5-candidate slice, not the 40-candidate
ANN pool the fix actually targets, so it was never capable of showing
`True` regardless of whether the fix worked. Sections 9 and 10's
threshold-tuning experiments were chasing a metric that couldn't move.
This needs to be re-measured correctly (Section 11) before any claim
about this fix's effect on true-object retrieval can be trusted in
either direction.

---

## 1. The symptom

In the dense-shell benchmark (a 23,000-object catalog: 20,000-object sparse
background + a 3,000-object crowded "shell" mimicking a real constellation
band), the GNN pipeline's catalog GNN was consulted 1,444 times across the
whole run. **Every single one of those 1,444 consults found the true
matching object entirely absent from its candidate pool** — the GNN was
never once asked to break a genuine tie between two real, present
candidates; it only ever correctly said "none of these match."

The first hypothesis was that the GNN's fixed `N_CANDIDATES = 5` input
width was too narrow — that the true object was present in the broader
40-candidate ANN pool but got bumped out of the GNN-visible top-5 by closer
decoys. **That hypothesis was directly tested and ruled out**: a diagnostic
patch logging the true object's exact rank within the ANN's 40-candidate
pool showed that whenever the true object was present in that pool, across
52,755 samples, it was rank 0 (the single closest candidate) **100% of the
time, with zero exceptions**. The top-5 cut was never actually excluding it.

That meant the true object wasn't losing a ranking competition inside the
pool — it was missing from the pool itself, upstream of any ranking at all.

## 2. The actual root cause

The ANN index (`src/leo_ann_index.py`) is a periodically-rebuilt
approximate search tree, not a live index. Its own module docstring states
the design tradeoff plainly: rebuilding on every detection would cost an
O(N) propagate + O(N) embed every time (the exact bottleneck this index was
built to eliminate — see the module's own measured numbers: ~50 minutes of
wall time at brute force vs. seconds with the periodic-rebuild approach).
Instead, the tree is rebuilt every `REBUILD_EVERY = 500` detections (or
when an overflow buffer of new/dirty entries exceeds
`OVERFLOW_REBUILD_THRESHOLD = 250`), and queries between rebuilds are
answered against that **frozen, stale snapshot** — every catalog object's
identity vector as it was at the moment of the last rebuild, not as it is
now.

This is explicitly justified as safe by one stated assumption:

> "Per-object drift in identity-vector space is small and grows roughly
> linearly with elapsed time... empirically measured: ~0.03 mean / ~0.13
> max after 30,000 real seconds of un-refreshed drift, vs. DRIFT_TOLERANCE
> ~0.18 for this catalog."

The critical detail: **`DRIFT_TOLERANCE` is a single, global number**,
auto-calibrated once per run from the whole catalog's 5th-percentile
nearest-neighbor spacing (`calibrate_thresholds()` in
`ground_station_tracking_simulation.py`). That calibration is dominated by
whatever population is largest in the catalog — in every benchmark run so
far, the sparse background (20,000 objects, spread across a wide
semi-major-axis/inclination range).

The dense shell's actual local spacing is far tighter than that
catalog-wide average. So the assumption "drift between rebuilds stays
smaller than the gap to the next object" — true almost everywhere in the
catalog — can fail specifically inside the shell, where dozens of objects
sit closer together than the global drift tolerance. When it fails, the
**stale** tree can rank a nearby decoy object as closer than the true
match, and if enough decoys do this simultaneously, the true match gets
pushed entirely outside the top-40 ANN pool — not just outside the GNN's
top-5. By the time the query reaches the GNN, the true object was never
retrievable in the first place.

This is a genuine implementation limitation, not an architectural ceiling
on the GNN-based approach. JPDA has no equivalent fix available for its
combinatorial blowup under density — that cost is intrinsic to the
algorithm. This ANN staleness issue, by contrast, sits in front of the
GNN and is fixable without touching the model at all.

## 3. The fix: a second, density-aware rebuild trigger

`needs_rebuild()` (the existing trigger) only fires on a global cadence.
This fix adds a second, independent trigger,
`needs_rebuild_for_query()` (new method on `LiveCatalogANNIndex` in
`leo_ann_index.py`), that fires per-query based on **local** conditions:
"is this specific detection sitting in a pocket dense enough, given
whatever staleness has accumulated since the last rebuild, that the cached
tree can no longer be trusted for it?"

It reuses `local_density()` — a method that already existed in
`leo_ann_index.py` for an unrelated purpose (a GNN input feature measuring
local crowding) — so the added per-detection cost is one cheap
`radius_neighbors()` call, not a full rebuild, until the density check
actually says yes.

### Where it's wired in

`ground_station_tracking_simulation.py`'s main detection loop previously
had a single rebuild condition:

```python
if ann_index.needs_rebuild():
    ann_index.rebuild(...)
    ...
```

This is now:

```python
global_rebuild_due = ann_index.needs_rebuild()
density_rebuild_due = (
    not global_rebuild_due
    and ANN_DENSITY_AWARE_REBUILD_ENABLED
    and ann_index.needs_rebuild_for_query(
        weighted_detection_vec,
        density_radius=DENSITY_RADIUS_MULTIPLE * drift_tolerance,
        density_threshold=ANN_DENSITY_REBUILD_THRESHOLD,
        min_detections_since_rebuild=ANN_DENSITY_REBUILD_MIN_DETECTIONS,
    )
)

if global_rebuild_due or density_rebuild_due:
    if density_rebuild_due:
        density_triggered_rebuild_count[0] += 1
    ann_index.rebuild(...)
    ...
```

The existing global check runs first and short-circuits the extra density
check on the common paths (no tree yet, tree just rebuilt) — so this adds
negligible overhead on a normal/sparse run, and only does real extra work
when a query is actually in a locally dense region.

**Nothing about the rebuild itself, the recalibration logic that follows a
rebuild, or the GNN/candidate-ranking logic downstream was changed.** This
fix only changes *when* a rebuild happens, not what a rebuild does or how
candidates are scored once retrieved.

### New configuration constants

All in `ground_station_tracking_simulation.py`, alongside the existing ANN
tuning block:

| Constant | Default | What it controls |
|---|---|---|
| `ANN_DENSITY_AWARE_REBUILD_ENABLED` | `True` | Kill switch — set `False` to fully revert to the original global-only rebuild behavior. |
| `ANN_DENSITY_REBUILD_THRESHOLD` | `6` | How many other objects within `DENSITY_RADIUS_MULTIPLE * drift_tolerance` of a query counts as "dense enough to be at risk." **This is a reasoned starting point, not yet empirically tuned** — see the tuning section below. |
| `ANN_DENSITY_REBUILD_MIN_DETECTIONS` | `5` | Minimum detections since the last rebuild before the density trigger is allowed to fire again, to prevent rebuild-thrashing on consecutive dense-region queries right after a rebuild already happened. |

### New diagnostics

The end-of-run summary now prints:

```
ANN index rebuilds     : <N> total (<D> density-triggered, <N-D> cadence/overflow-triggered)
```

so you can see directly, from the console output alone, whether the
density trigger fired at all on a given run (0 density-triggered rebuilds
on a sparse-only run is expected and correct — it means the trigger
correctly stayed dormant where it wasn't needed).

## 4. What this fix does NOT change

- Does not touch the GNN model, its architecture, or `N_CANDIDATES` —
  no retraining is required or implied by this fix.
- Does not change `REBUILD_EVERY`, `OVERFLOW_REBUILD_THRESHOLD`, or
  `CANDIDATE_POOL_SIZE` — the original global cadence and pool size are
  untouched and still apply exactly as before.
- Does not change JPDA or `jpda_baseline.py` in any way — this is purely a
  GNN-pipeline-side fix.
- Does not change scoring, gating, or acceptance logic downstream of
  candidate retrieval — only *when the tree is refreshed* changes; how
  candidates are ranked and accepted once retrieved is identical to
  before.

## 5. Tuning `ANN_DENSITY_REBUILD_THRESHOLD`

The default (`6`) was chosen by reasoning about the problem, not by
fitting it to a labeled result — that calibration step is still open.
Once you have a run with this fix active, the right way to tune it:

1. Run the dense-shell benchmark with the fix enabled at the default
   threshold.
2. Check the new `ANN index rebuilds` summary line — if
   `density-triggered` is 0, the threshold is too high (never firing);
   if it's very large relative to total detections, it may be firing on
   ordinary background density too (check by comparing against a
   sparse-only run's density-triggered count, which should stay at or
   near 0).
3. Cross-reference against the same shell-only mismatch-rate breakdown
   used earlier in this project (`jpda_baseline.py`'s shell-tagging
   convention, or the `name`-field-based filtering now available in
   `eval_results.json`'s `detections` array) — did shell-only mismatches
   actually drop relative to the pre-fix run?
4. Adjust `ANN_DENSITY_REBUILD_THRESHOLD` down if mismatches persist in
   the shell (the trigger isn't firing early enough), or up if rebuild
   frequency/wall-time regresses unacceptably without further mismatch
   improvement (the trigger is firing more than it needs to).

## 6. How to validate this fix

This fix has not yet been run end-to-end. Before treating the "GNN beats
JPDA under density" PoC claim as fully closed, run the dense-shell
benchmark again with this patch in place and compare against the
pre-fix numbers already on record:

**Pre-fix baseline (already measured, dense shell):**
- Shell-only mismatch rate: 0.24% (14/5,922 known shell detections)
- Shell-only mean lookup latency: ~1,292us (barely elevated over background)
- 1,444/1,444 GNN consults found the true object absent from the pool

**What to check post-fix:**
- Does the shell-only mismatch rate drop below 0.24%?
- Does `true_object_in_pool` (in `gnn_live_invocations.json`, now
  name-tagged) start showing `True` for at least some shell-region
  consults — i.e., is the GNN now actually being asked to resolve real
  ties, rather than only ever confirming novelty?
- Does the `ANN index rebuilds` summary show a nonzero density-triggered
  count specifically on the dense-shell run, and near-zero on a
  sparse-only run (confirming the trigger is targeted, not blanket)?
- What's the wall-time cost of the extra rebuilds — is it still
  substantially cheaper than JPDA's ~820s dense-shell run, or has the
  fix eaten into that advantage?

That comparison — not this document — is what actually closes the "GNN
classifier resolves genuine ties JPDA gets wrong" claim.

## 7. Known issue found during validation

**The first version of this fix did not work.** Running it end-to-end
produced results byte-for-byte identical to the pre-fix baseline —
`ANN index rebuilds : 258 total (0 density-triggered, 258
cadence/overflow-triggered)`. The density trigger never fired once across
the entire dense-shell run.

**Root cause:** `local_density()` unconditionally subtracted 1 from every
neighbor count, on the assumption that the query vector is itself already
a member of the tree (true for its two original call sites —
`top5_weighted_vecs` in the main sim loop and `cand_weighted_vecs` in
`catalog_gnn.py`'s training harvester, both of which query with vectors of
objects already in the catalog). `needs_rebuild_for_query()`'s new call
site is different: it queries with the **fresh, not-yet-registered
detection vector**, which is never in the tree. Subtracting 1 for a
self-match that doesn't exist silently undercounted every density reading
by one, which was enough to keep every real reading below
`ANN_DENSITY_REBUILD_THRESHOLD = 6` for the entire run.

**Fix:** `local_density()` now takes a `query_is_tree_member` parameter
(default `True`, preserving the original behavior exactly for both
existing call sites). `needs_rebuild_for_query()` explicitly passes
`query_is_tree_member=False`, since its query is always an external,
unregistered detection.

This is called out explicitly here, rather than silently corrected,
because it's a useful cautionary example: a shared low-level utility
method's implicit assumption (query is a tree member) was safe for every
caller that existed when it was written, and silently wrong for a new
caller added later with a different calling convention. If
`local_density()` gains further callers in the future, check this
assumption again.

The validation run described in Section 6 has not yet been re-run against
this corrected version — that is the actual next step.

## 8. Second known issue found during validation: radius mismatch

After fixing the off-by-one bug in Section 7, the density trigger *still*
fired zero times. A diagnostic (`query_density()`, sampling raw density
readings on ~1/20 detections) showed why directly: real local density at
the radius being used (`DENSITY_RADIUS_MULTIPLE * drift_tolerance ≈
0.269`) maxed out at **3** across the entire 64,265-detection run —
nowhere near `ANN_DENSITY_REBUILD_THRESHOLD = 6`. Raising the threshold
would not have helped; the radius itself was too tight to ever observe
meaningful crowding, in the shell or the background.

**Root cause:** `DENSITY_RADIUS_MULTIPLE = 2.0` (from `catalog_gnn.py`)
was reused on the assumption it would naturally capture shell crowding.
It doesn't, because it was tuned for a different purpose — a GNN training
feature meant to distinguish "extremely close" from "moderately close"
candidates, not to measure whether a query sits in a broadly crowded
region. Checked directly against `gnn_live_invocations.json`'s own
`min_dist` values (the actual embedding-space distance from each live GNN
consult to its nearest candidate, in the same weighted space
`local_density()` operates in): the old radius (~0.269) only covered
25.9% of real observed candidate distances — most real candidates,
including in the dense shell, were simply farther away than the radius
being searched.

**Fix:** a new, separate constant, `ANN_DENSITY_REBUILD_RADIUS_MULTIPLE =
3.5`, decoupled from `DENSITY_RADIUS_MULTIPLE` (which remains untouched
and still governs the GNN training feature at its original call site).
At `3.5 * drift_tolerance`, the radius covers 97.6% of the same real
observed candidate-distance distribution.

**This value is catalog-specific and was calibrated directly against this
run's actual data — it is not a universal constant.** If the dense-shell
catalog is regenerated with different width/density settings, or run
against a different background catalog, recalibrate
`ANN_DENSITY_REBUILD_RADIUS_MULTIPLE` the same way: pull `min_dist` from a
fresh `gnn_live_invocations.json` and check what multiple of
`drift_tolerance` covers the bulk of that distribution, rather than
assuming this number transfers.

## 9. Validation run against both corrections: partial pass

The run described as pending in Sections 7 and 8 has now happened
(`outputs/log_04_tracking_simulation.txt`, current as of this commit).
Both fixes are confirmed active in code — the off-by-one correction
(`query_is_tree_member=False` at both `needs_rebuild_for_query()` call
sites in `leo_ann_index.py`) and the decoupled radius
(`ANN_DENSITY_REBUILD_RADIUS_MULTIPLE = 3.5` in
`ground_station_tracking_simulation.py`) are both present, not just
described in this doc.

**Results, checked against the four criteria from Section 6:**

| Check | Pre-fix | Post-fix | Verdict |
|---|---|---|---|
| Shell-only mismatch rate | 0.24% (14/5,922) | **0.174%** (13/7,468) | Improved, modestly |
| `true_object_in_pool` ever `True` on a live consult | Never (0/1,444) | **Still never (0/1,437)** | **Fails** — the actual claim this fix exists to demonstrate |
| Density-triggered rebuild count (dense-shell run) | 0 | **99** (of 309 total) | Trigger is firing, confirmed non-blanket (fires far less than the 210 cadence-triggered rebuilds) |
| Wall-time cost vs. JPDA's ~820s dense-shell run | ~181s (pre-fix, effectively cadence-only) | **180.9s** | No meaningful regression — still a ~4.5x speed advantage over JPDA |

**Reading this honestly:** the density trigger is doing something now,
and it isn't free — the shell mismatch rate did drop and wall time didn't
regress. But it has not yet crossed the bar this document itself sets in
Section 6: the whole point was to get the true object back into the ANN's
candidate pool often enough that the GNN is actually resolving ties
instead of only ever confirming novelty. **Zero of 1,437 consults show
that.** The GNN pipeline's shell-region behavior is, mechanistically,
still "the true match was already evicted before the GNN was ever asked."
The mismatch-rate improvement is more likely a side effect of a shorter
average staleness window across the whole run (extra rebuilds happen
somewhere, so *some* queries benefit) than evidence the intended repair
path — ANN retrieval → GNN tie-break — is working as designed.

The diagnostic this run itself printed points at why the trigger is
under-firing: `raw_density distribution: min=0 p50=1 p90=2 p95=2 p99=4
max=9`, against `ANN_DENSITY_REBUILD_THRESHOLD=6`. The threshold sits
above the p99 of observed readings — it's only catching the extreme tail,
which is consistent with a trigger that fires (99 times) but not on most
of the queries that are actually at risk.

## 10. Threshold retune (6 → 3): fired far more, changed nothing on the metric that mattered

Based on the Section 9 diagnostic (`p95=2, p99=4` against a threshold of
`6`), `ANN_DENSITY_REBUILD_THRESHOLD` was lowered to `3`
(`src/ground_station_tracking_simulation.py`, single-line change) to
catch the p95–p99 range instead of only the extreme tail.

**Result:**

| Check | Threshold=6 (Section 9) | Threshold=3 | Change |
|---|---|---|---|
| Density-triggered rebuilds | 99 | **2,721** (of 2,722 total) | ~27x more — the trigger now fires on almost every single detection |
| Wall time | 180.9s | **440.9s** | 2.4x slower — still under JPDA's 820s, but no longer "negligible overhead" |
| Shell/total mismatches | 15 | 17 | Slightly worse, not better |
| `true_object_in_pool == True` | 0/1,437 | **0/1,433** | **Unchanged** |

Rebuilding before nearly every query and still getting zero `True`
readings falsifies the threshold-tuning approach outright — this isn't a
"needs a smaller number" problem. It also weakens the staleness-timing
theory floated at the end of Section 9 (rebuild not finishing before the
next query lands): `ann_index.rebuild(...)` is called synchronously,
in-line, immediately before `query_candidates()` in the same iteration of
the detection loop (`ground_station_tracking_simulation.py`, main loop) —
there is no concurrent/async path where a query could race an in-flight
rebuild. If the tree were actually rebuilt on 2,721 of 2,722 queries,
using genuinely current data each time, and the true object was *still*
never retrievable, staleness cannot be the explanation for those misses.
That contradiction is what led to Section 11.

## 11. Third bug found: `true_object_in_pool` was never measuring the ANN pool

**The metric itself was wrong, not the rebuild logic.** Tracing where
`true_object_in_pool` is actually written
(`ground_station_tracking_simulation.py`):

```python
"true_object_in_pool": bool(_true_idx_for_log is not None and _true_idx_for_log in top5),
```

`top5` is the GNN's 5-candidate input slice — the closest 5 by
*unweighted* distance out of whatever `query_candidates()` returned
(up to `CANDIDATE_POOL_SIZE = 40` candidates). This field has never
measured "is the true object anywhere in the ANN's 40-candidate pool,"
which is what this whole fix targets and what Section 6's validation
plan asked for. It measures a narrower, different thing: "is the true
object in the GNN's top-5 slice of that pool" — the exact `N_CANDIDATES`
cut that the investigation's Section 1 diagnostic (a since-removed patch,
`gnn_candidate_rank_diagnostic.json`) says it already ruled out as a
cause, using a *different* measurement that no longer exists in the
current codebase.

In short: two different diagnostics, at two different points in this
investigation, answered two different questions, and the doc treated
them as interchangeable:

- **Section 1's now-removed patch** measured rank *within the full
  40-candidate ANN pool*. Its "100% rank 0" finding is about that pool.
- **The live `true_object_in_pool` field**, still in the code and the one
  Sections 9–10 relied on, measures membership in the **top-5 slice**
  only.

A `True`/`False` split on the top-5 field says nothing conclusive about
whether the density-aware rebuild fixed 40-pool retrieval — the thing it
was actually built to fix. Every measurement in Sections 9 and 10 that
used this field to judge the fix's effectiveness needs to be treated as
inconclusive, not negative. It is not evidence the fix failed; it is
evidence the fix was never actually being measured.

**What to do next, before touching any rebuild logic further:**

1. Add a second field to the same log line, alongside the existing one,
   rather than replacing it (the top-5 question is still meaningful for
   GNN-input-width discussions, just not for this fix):
   ```python
   "true_object_in_ann_pool": bool(_true_idx_for_log is not None and _true_idx_for_log in candidate_idx),
   ```
   (`candidate_idx` is the full ≤40-item array `query_candidates()`
   returned earlier in the same loop iteration — already in scope at this
   point, no extra query needed.)
2. Re-run the dense-shell benchmark once, at whichever threshold value is
   preferred (reverting to `6` first is reasonable, since `3` bought
   nothing and cost 2.4x wall time) — no need to re-sweep thresholds until
   the pool-level metric is available to actually judge them by.
3. Recheck Section 6's real question against `true_object_in_ann_pool`,
   not `true_object_in_pool`. That result — not anything in Sections 9 or
   10 — is what will actually tell us whether the density-aware rebuild
   works.