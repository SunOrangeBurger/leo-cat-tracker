#!/usr/bin/env python3
"""
Compare a real run's GNN invocations against a fresh training batch.

Carried forward as-is in spirit from the previous codebase (README.md
section 5, item 5): this is the ONLY diagnostic that looks at an actual
live run instead of re-evaluating the model against its own synthetic
training distribution. Every earlier diagnostic in this system's history
(the ones this file's predecessor replaced) re-checked the model against
generate_training_examples()'s own output, which is circular -- it can
only ever confirm the model generalizes *within* the synthetic
distribution, never catch a live/train mismatch. This is the standing
verification method for any future regression, not a one-off debugging
script -- re-run it after any change to catalog_gnn.py's harvesting logic,
the gate thresholds, or the distance weights.

Usage:
    1. Run ground_station_tracking_simulation.py (writes
       outputs/gnn_live_invocations.json if any GNN consultations happened).
    2. Run this script -- it harvests a fresh batch via
       catalog_gnn.harvest_live_loop_examples() using the SAME gate
       thresholds the live run used, and prints both distributions
       side by side.
"""

import json

import numpy as np
import pandas as pd

import catalog_gnn as gnn_mod
from ground_station_tracking_simulation import (
    load_embedding_store, calibrate_thresholds,
    CATALOG_PATH, EMBEDDINGS_PATH, FALLBACK_NPZ_PATH, DRIFT_TOLERANCE, MARGIN_REQUIRED,
)

LIVE_LOG_PATH = "outputs/gnn_live_invocations.json"


def _percentiles(values, pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    if len(values) == 0:
        return {p: float("nan") for p in pcts}
    return {p: float(np.percentile(values, p)) for p in pcts}


def main():
    try:
        with open(LIVE_LOG_PATH) as f:
            live_log = json.load(f)
    except FileNotFoundError:
        print(f"No live invocation log at {LIVE_LOG_PATH} -- run "
              f"ground_station_tracking_simulation.py first (with at least some GNN "
              f"consultations) before running this diagnostic.")
        return

    if not live_log:
        print("Live invocation log is empty -- no GNN consultations happened in that run.")
        return

    live_true_in_pool = [r for r in live_log if r["true_object_in_pool"]]
    live_not_in_pool = [r for r in live_log if not r["true_object_in_pool"]]
    print(f"Live run: {len(live_log)} GNN invocations, "
          f"{len(live_true_in_pool)} ({len(live_true_in_pool)/len(live_log)*100:.1f}%) had the true "
          f"object in the candidate pool, {len(live_not_in_pool)} did not.")

    live_min_dist_in_pool = [r["min_dist"] for r in live_true_in_pool]
    live_min_dist_not_in_pool = [r["min_dist"] for r in live_not_in_pool]

    print("\nLive min_dist, true_object_in_pool=True:")
    print("  " + "  ".join(f"p{p}={v:.4f}" for p, v in _percentiles(live_min_dist_in_pool).items()))
    print("Live min_dist, true_object_in_pool=False:")
    print("  " + "  ".join(f"p{p}={v:.4f}" for p, v in _percentiles(live_min_dist_not_in_pool).items()))

    decision_counts = {}
    for r in live_log:
        decision_counts[r["decision"]] = decision_counts.get(r["decision"], 0) + 1
    print(f"\nLive decisions: {decision_counts}")

    # --- fresh training-distribution batch, same gate settings ---
    truth_df = pd.read_csv(CATALOG_PATH)
    store = load_embedding_store(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    scalers = store["scalers"]
    import identity_embedding as embed_mod
    distance_weights = np.asarray(store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32)
    weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights

    drift_tolerance = DRIFT_TOLERANCE
    margin_required = MARGIN_REQUIRED
    if drift_tolerance is None or margin_required is None:
        drift_tolerance, margin_required = calibrate_thresholds(
            weighted_store_vectors, cat_i=store["elements_i"], cat_types=store["types"], cat_e=store["elements_e"],
        )

    fresh_examples = gnn_mod.harvest_live_loop_examples(
        truth_df, scalers, distance_weights, drift_tolerance, margin_required,
        n_examples=min(len(live_log), 4000), seed=123,
    )
    fresh_known = [e for e in fresh_examples if e["label"].sum() > 0]
    fresh_novel = [e for e in fresh_examples if e["label"].sum() == 0]
    fresh_min_dist_known = [float(np.min(e["candidate_dists"])) for e in fresh_known]
    fresh_min_dist_novel = [float(np.min(e["candidate_dists"])) for e in fresh_novel]

    print(f"\nFresh harvested batch: {len(fresh_examples)} examples, "
          f"{len(fresh_known)} ({len(fresh_known)/max(1,len(fresh_examples))*100:.1f}%) true-object-in-pool.")
    print("Fresh min_dist, true_object_in_pool=True:")
    print("  " + "  ".join(f"p{p}={v:.4f}" for p, v in _percentiles(fresh_min_dist_known).items()))
    print("Fresh min_dist, true_object_in_pool=False:")
    print("  " + "  ".join(f"p{p}={v:.4f}" for p, v in _percentiles(fresh_min_dist_novel).items()))

    print("\nIf the live and fresh percentile rows above (for the same true_object_in_pool "
          "value) look substantially different, the harvester's live-loop simulation no "
          "longer matches the real tracking sim's actual dynamics -- re-check "
          "catalog_gnn.harvest_live_loop_examples()'s parameters (embed_fraction, "
          "register_prob, noise settings, plausibility_multiple) against "
          "ground_station_tracking_simulation.py's current config before trusting any "
          "checkpoint trained against it.")


if __name__ == "__main__":
    main()
