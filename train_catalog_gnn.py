#!/usr/bin/env python3
"""
Standalone catalog-GNN trainer. Successor to train_gnn.py (README.md
section 5, item 3).

Kept separate from the tracking sim's hot path, same reasoning as before:
training touches a full harvested live-loop run and takes real wall time,
so doing it as a side effect of starting a sim run makes iteration slow
and risks silently retraining (overwriting outputs/catalog_gnn.pt) any
time the auto-calibrated drift_tolerance/margin_required drift slightly
between runs. ground_station_tracking_simulation.py only ever LOADS a
checkpoint whose gate_stamp matches its own settings -- if none exists (or
it's stale), catalog-GNN consultation is disabled for that run rather than
silently retraining on the spot. Run this script explicitly whenever:

  - leo_catalog.csv or outputs/leo_embeddings.pt change (regenerated data,
    re-embedding, cold start, etc.)
  - catalog_gnn.py's training/harvesting logic changes
    (TRAINING_LOGIC_VERSION bump)
  - you want to inspect/tune training in isolation from the sim's hot path

A NOTE ON WALL-CLOCK TIME AND --n-examples: known-in-pool near-misses (the
harder, more informative class for calibration -- see
catalog_gnn.harvest_live_loop_examples' docstring) are genuinely rare in
this system, empirically on the order of 1 per several thousand harvested
detection events at full catalog scale. The harvester actively keeps
running past the novel-example quota until it finds
min_known_examples := max(20, n_examples // 400) of them (or gives up at
max_events with a warning) -- this is correct and by design, but it does
mean this script takes real minutes to run at the defaults, not seconds,
and a small --n-examples smoke test (a few hundred) will have very few
known-in-pool validation examples and a correspondingly noisy/degenerate
confident_novel_threshold. Don't mistake that for a bug in a quick test --
run at or near the defaults for a checkpoint you'd actually rely on.
"""

import argparse

import pandas as pd

import catalog_gnn as gnn_mod
from ground_station_tracking_simulation import (
    load_embedding_store,
    calibrate_thresholds,
    CATALOG_PATH,
    EMBEDDINGS_PATH,
    FALLBACK_NPZ_PATH,
    DRIFT_TOLERANCE,
    MARGIN_REQUIRED,
)


def main():
    parser = argparse.ArgumentParser(description="Train the catalog GNN offline.")
    parser.add_argument("--n-examples", type=int, default=12000,
                         help="Number of live-loop-harvested training examples to collect.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--false-known-penalty", type=float, default=3.0,
                         help="Asymmetric loss weight on false-KNOWN errors (README 4.4).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-known-examples", type=int, default=None,
                         help="Minimum known-in-pool (true object present in candidate pool) "
                              "examples the harvester must collect before stopping, regardless "
                              "of how many genuinely-novel examples it already has. Defaults to "
                              "the harvester's own max(20, n_examples // 400), which is too small "
                              "to train or validate the confident_novel_threshold reliably -- if "
                              "val_recall(known) is stuck at 0.000 during training, or the "
                              "calibrated thresholds are both near 0, raise this (e.g. 300-500).")
    args = parser.parse_args()

    truth_df = pd.read_csv(CATALOG_PATH)
    store = load_embedding_store(EMBEDDINGS_PATH, FALLBACK_NPZ_PATH)
    scalers = store["scalers"]

    import numpy as np
    import identity_embedding as embed_mod
    distance_weights = np.asarray(
        store.get("distance_weights", embed_mod.DISTANCE_WEIGHTS), dtype=np.float32
    )
    weighted_store_vectors = np.asarray(store["vectors"], dtype=np.float32) * distance_weights

    drift_tolerance = DRIFT_TOLERANCE
    margin_required = MARGIN_REQUIRED
    if drift_tolerance is None or margin_required is None:
        drift_tolerance, margin_required = calibrate_thresholds(
            weighted_store_vectors,
            cat_i=store["elements_i"],
            cat_types=store["types"],
            cat_e=store["elements_e"],
        )

    print(f"Truth population: {len(truth_df)} objects ({CATALOG_PATH})")
    print(f"Embedding store: {len(store['names'])} objects on record")
    print(f"drift_tolerance={drift_tolerance:.6f}  margin_required={margin_required:.6f}")
    print("-" * 70)

    if not gnn_mod.TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train the catalog GNN.")

    # force_retrain=True: this script's whole purpose is to (re)train, not
    # silently reuse whatever's on disk.
    model, confident_known_threshold, confident_novel_threshold = gnn_mod.load_or_train_model(
        truth_df, scalers, distance_weights, drift_tolerance, margin_required,
        force_retrain=True, n_examples=args.n_examples, epochs=args.epochs,
        false_known_penalty=args.false_known_penalty, seed=args.seed,
        min_known_examples=args.min_known_examples,
    )
    print(f"Done. Saved to {gnn_mod.MODEL_PATH}")
    print(f"confident_known_threshold={confident_known_threshold:.4f}  "
          f"confident_novel_threshold={confident_novel_threshold:.4f}")


if __name__ == "__main__":
    main()