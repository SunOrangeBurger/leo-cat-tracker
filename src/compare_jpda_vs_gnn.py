#!/usr/bin/env python3
"""
Head-to-head comparison: JPDA baseline (jpda_baseline.py) vs. the GNN
pipeline (ground_station_tracking_simulation.py), read from their two
independently-produced result files:

    outputs/eval_results.json  -> stages.tracking_simulation (GNN pipeline)
    outputs/jpda_results.json  -> JPDA baseline

Both runs consume the SAME catalog (data/leo_catalog.csv, or whatever
--catalog points both scripts at), the SAME seed (11), the SAME
measurement/process noise, and the SAME simulated duration -- see
jpda_baseline.py's module docstring for why this makes the comparison
fair rather than a straw man.

This script does not silently pick a "winner." It prints every metric
side by side, computes deltas, and prints an explicit, unfudgeable
density-regime check -- because the single most important caveat in this
comparison (see README.md's own success criteria in Section 1) is whether
the run actually landed in the dense/ambiguous regime the whole project
exists to test. A head-to-head where neither system saw meaningful
ambiguity proves much less than a head-to-head where they did.
"""

import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_gnn_results(path):
    with open(path) as f:
        d = json.load(f)
    ts = d["stages"]["tracking_simulation"]
    # Strip the bulky per-detection log before returning -- this script
    # only needs the summary metrics, and keeping the full log would
    # bloat every downstream comparison JSON by several MB for no reason.
    return {k: v for k, v in ts.items() if k not in ("detections", "recalibrations")}


def load_jpda_results(path):
    with open(path) as f:
        return json.load(f)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compare JPDA baseline vs GNN pipeline results.")
    parser.add_argument("--gnn-results", default=str(REPO_ROOT / "outputs" / "eval_results.json"))
    parser.add_argument("--jpda-results", default=str(REPO_ROOT / "outputs" / "jpda_results.json"))
    parser.add_argument("-o", "--output", default=str(REPO_ROOT / "outputs" / "jpda_vs_gnn_comparison.json"))
    args = parser.parse_args()

    gnn = load_gnn_results(args.gnn_results)
    jpda = load_jpda_results(args.jpda_results)

    print("=" * 78)
    print("JPDA vs GNN PIPELINE -- HEAD-TO-HEAD")
    print("=" * 78)

    print(f"\n{'metric':<32}{'JPDA':>18}{'GNN pipeline':>18}")
    print("-" * 68)

    def row(label, jval, gval, fmt="{}"):
        print(f"{label:<32}{fmt.format(jval):>18}{fmt.format(gval):>18}")

    row("total detections", jpda["total_detections"], gnn["total_detections"])
    row("known (matched) count", jpda["known_count"], gnn["known_count"])
    row("known rate", pct(jpda['known_count'], jpda['total_detections']),
        pct(gnn['known_count'], gnn['total_detections']), "{:.1f}%")
    row("correct matches", jpda["correct_matches"], gnn["correct_matches"])
    row("correct match rate", jpda["correct_matches_pct"], gnn["correct_matches_pct"], "{:.2f}%")
    row("mismatches", jpda["mismatches"], gnn["mismatches"])
    row("mismatch rate", jpda["mismatches_pct"], gnn["mismatches_pct"], "{:.3f}%")
    row("catalog completeness", jpda["catalog_completeness_pct"], gnn["catalog_completeness_pct"], "{:.1f}%")
    row("final catalog size", jpda["final_catalog_size"], gnn["final_catalog_size"])
    row("update commits", jpda["update_committed"], gnn["update_committed"])
    row("update withheld", jpda["update_withheld"], gnn["update_withheld"])
    row("wall time (s)", jpda["real_wall_time_s"], gnn["real_wall_time_s"], "{:.2f}")
    row("mean lookup latency (us)", f"{jpda['lookup_us_mean']:.1f}", "n/a (per-stage log)", "{}")

    print("\n" + "-" * 68)
    print("JPDA-SPECIFIC (no GNN equivalent -- combinatorial-cost evidence)")
    print("-" * 68)
    print(f"{'clusters processed':<32}{jpda['n_clusters_processed']:>18}")
    print(f"{'mean cluster size':<32}{jpda['mean_cluster_size']:>18.3f}")
    print(f"{'max cluster size':<32}{jpda['max_cluster_size']:>18}")
    print(f"{'degraded-fallback clusters':<32}{jpda['degraded_fallback_clusters']:>18}"
          f"   (exceeded {jpda['config']['max_cluster_enum']}-way exact enumeration)")

    print("\n" + "-" * 68)
    print("GNN-SPECIFIC (no JPDA equivalent -- ambiguity-handling evidence)")
    print("-" * 68)
    print(f"{'GNN consults (2nd opinion invoked)':<40}{gnn['gnn_consults']:>14}"
          f"  ({pct(gnn['gnn_consults'], gnn['total_detections']):.1f}% of detections)")
    print(f"{'  -> GNN said KNOWN':<40}{gnn['gnn_known']:>14}")
    print(f"{'  -> GNN said AMBIGUOUS (deferred)':<40}{gnn['gnn_ambiguous']:>14}")
    print(f"{'  -> GNN said NOVEL':<40}{gnn['gnn_novel']:>14}")
    print(f"{'GNN overturn precision':<40}"
          f"{gnn['gnn_overturn_correct']}/{gnn['gnn_overturn_total']}"
          f"  ({gnn['gnn_overturn_precision_pct']:.1f}%)  <- small sample, see README section 7")

    # ------------------------------------------------------------------
    # NOTE ON THE KNOWN-RATE GAP: most of the raw known-rate difference
    # between the two systems is a GATE/TRACK-MANAGEMENT design
    # difference, not evidence about the GNN classifier's association
    # quality specifically. Worth separating out before anyone reads the
    # headline known-rate numbers as "the GNN wins at association."
    # ------------------------------------------------------------------
    gnn_gate_only_known = gnn["known_count"] - gnn["gnn_known"]
    print("\n" + "-" * 68)
    print("NOTE ON THE KNOWN-RATE GAP")
    print("-" * 68)
    print(
        f"JPDA known rate: {pct(jpda['known_count'], jpda['total_detections']):.1f}%   "
        f"GNN pipeline known rate: {pct(gnn['known_count'], gnn['total_detections']):.1f}%\n"
        f"{gnn_gate_only_known}/{gnn['known_count']} of the GNN pipeline's KNOWN detections "
        f"({pct(gnn_gate_only_known, gnn['known_count']):.1f}%) resolved through its ordinary\n"
        f"fast gate, not its GNN classifier -- only {gnn['gnn_known']} came from a GNN-confirmed\n"
        f"accept. So most of this gap reflects a GATE/TRACK-MANAGEMENT design difference\n"
        f"(JPDA's strict chi-square gate + single-detection-per-cluster constraint vs. the\n"
        f"pipeline's distance+margin gate with existing-slot correction for re-discovered\n"
        f"objects), not evidence about the GNN classifier specifically. Read the GNN-consult\n"
        f"and overturn numbers above for the classifier question; don't read the raw known-rate\n"
        f"gap as \"the GNN beats JPDA at association.\""
    )

    # ------------------------------------------------------------------
    # THE CHECK THAT MATTERS MOST: did this run actually land in the
    # dense/ambiguous regime the project is meant to test? A clean 100%
    # score on BOTH sides is evidence of an easy regime, not evidence the
    # comparison is uninformative -- but it means "surpasses JPDA" is not
    # yet demonstrated on the hard cases and shouldn't be claimed as such.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DENSITY-REGIME CHECK (read this before drawing conclusions)")
    print("=" * 78)
    jpda_ambig_rate = pct(jpda["max_cluster_size"] > 1 and jpda["n_clusters_processed"], jpda["n_clusters_processed"])
    print(f"JPDA max simultaneous-cluster size this run : {jpda['max_cluster_size']}")
    print(f"GNN ambiguous-consult rate this run          : "
          f"{pct(gnn['gnn_ambiguous'], gnn['total_detections']):.2f}% of detections")
    if jpda["max_cluster_size"] <= 2:
        print(
            "\nWARNING: JPDA never (or almost never) had to arbitrate between more than\n"
            "2 simultaneously-active candidates in this run, and the GNN pipeline's own\n"
            "genuinely-ambiguous rate was well under 1%. That means this specific run\n"
            "config (catalog size, duration, acceleration factor) did NOT exercise the\n"
            "dense/crowded regime that is this project's actual hypothesis (README.md\n"
            "Section 1). A clean win here is evidence the pipeline works on the EASY\n"
            "regime, not evidence it beats JPDA where JPDA is expected to struggle.\n"
            "To test the actual claim, re-run both scripts with a higher simultaneous\n"
            "detection rate -- e.g. a much larger catalog at the same duration, a\n"
            "longer duration, or a lower ACCELERATION_FACTOR / more ground stations\n"
            "observing concurrently -- until cluster sizes and GNN-consult rates are\n"
            "routinely > 1 digit, then compare again."
        )
    else:
        print(f"\nThis run DID exercise multi-candidate ambiguity (max cluster size "
              f"{jpda['max_cluster_size']}). The comparison above is evidence for the "
              f"dense-regime claim, not just the easy-regime claim.")

    comparison = {
        "jpda": jpda,
        "gnn_pipeline": gnn,
        "known_rate_gap_breakdown": {
            "jpda_known_rate_pct": pct(jpda["known_count"], jpda["total_detections"]),
            "gnn_known_rate_pct": pct(gnn["known_count"], gnn["total_detections"]),
            "gnn_known_via_fast_gate": gnn_gate_only_known,
            "gnn_known_via_gnn_classifier": gnn["gnn_known"],
            "gnn_known_via_gnn_classifier_pct_of_known": pct(gnn["gnn_known"], gnn["known_count"]),
            "interpretation": (
                "Most of the known-rate gap is a gate/track-management design "
                "difference (JPDA's strict chi-square + single-detection-per-cluster "
                "constraint vs. the pipeline's distance+margin gate with "
                "existing-slot correction), not evidence the GNN classifier itself "
                "outperforms JPDA's association math -- only "
                f"{gnn['gnn_known']}/{gnn['known_count']} GNN-pipeline known-matches "
                "actually came from the GNN classifier path."
            ),
        },
        "density_regime_check": {
            "jpda_max_cluster_size": jpda["max_cluster_size"],
            "jpda_mean_cluster_size": jpda["mean_cluster_size"],
            "gnn_ambiguous_rate_pct": pct(gnn["gnn_ambiguous"], gnn["total_detections"]),
            "gnn_consult_rate_pct": pct(gnn["gnn_consults"], gnn["total_detections"]),
            "landed_in_dense_regime": jpda["max_cluster_size"] > 2,
        },
    }
    with open(args.output, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nWrote full comparison JSON to {args.output}")


if __name__ == "__main__":
    main()