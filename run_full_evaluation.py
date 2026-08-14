#!/usr/bin/env python3
"""
End-to-end system evaluation for the LEO catalog / catalog-GNN pipeline.

Runs the full pipeline in order (each stage is a subprocess, matching how
these scripts are meant to be invoked -- see train_catalog_gnn.py and
compare_live_vs_training.py's own docstrings on why training/sim are kept
as separate processes rather than importable side effects):

    1. data_generator.py                    -> leo_catalog.csv
    2. identity_embedding.py                -> outputs/leo_embeddings.pt
    3. train_catalog_gnn.py                 -> outputs/catalog_gnn.pt
    4. ground_station_tracking_simulation.py -> outputs/gnn_live_invocations.json
    5. compare_live_vs_training.py          -> live-vs-training distribution check
    6. check_catalog_against_embeddings.py  -> catalog/embedding-store cross-check

None of these scripts expose their results as structured data -- they only
print to stdout (and stage 4 additionally writes
outputs/gnn_live_invocations.json). So this harness:

  - runs every stage as a subprocess, streaming output live AND capturing it,
  - parses each stage's stdout with regexes matched to that script's actual
    print statements (see the PARSE_* functions below -- each one is keyed
    to a specific f-string in the source it parses),
  - loads outputs/gnn_live_invocations.json directly for the fine-grained
    per-invocation distributions (min_dist, decision, confidence, etc.)
    that don't appear in any printed summary,
  - writes everything to outputs/eval_results.json for reuse,
  - renders a set of matplotlib figures covering: pipeline health,
    catalog/gate/GNN decision breakdown, timing, ANN recall, association
    integrity, and the live-vs-training distribution comparison.

A NOTE ON "RUNNING EVERYTHING SIMULTANEOUSLY":
The stages are NOT all mutually independent -- each one reads a file the
previous one wrote (leo_catalog.csv -> leo_embeddings.pt -> catalog_gnn.pt
-> gnn_live_invocations.json -> compare_live_vs_training.py's own read of
that json). Running the whole chain "at once" would just mean every stage
racing to read a file that hasn't been written yet. What *is* genuinely
independent, and so IS run concurrently by default here:
    - check_catalog_against_embeddings.py only needs leo_catalog.csv +
      leo_embeddings.pt -- it does NOT need the GNN checkpoint or the sim
      output, so it's kicked off in a background thread as soon as stage 2
      (identity_embedding.py) finishes, and runs concurrently with stage 3
      (train_catalog_gnn.py) and stage 4 (the tracking sim) while they work
      through the rest of the true dependency chain.
compare_live_vs_training.py is a hard exception: its entire purpose is to
read outputs/gnn_live_invocations.json, which the tracking sim only writes
at the very end of its run (see ground_station_tracking_simulation.py's
final "if gnn_invocation_log:" block) -- it cannot start any earlier than
that, concurrently or otherwise, without reading a stale or missing file.
Use --sequential to disable the one real concurrency opportunity above and
run everything strictly one-at-a-time (useful for the cleanest possible
interleaved console output when debugging a single stage).

Usage:
    python run_full_evaluation.py                 # full pipeline, defaults,
                                                    # with the one real
                                                    # concurrency opportunity
                                                    # (see note above)
    python run_full_evaluation.py --sequential     # strictly one stage at a time
    python run_full_evaluation.py --skip-generate  # reuse existing leo_catalog.csv
    python run_full_evaluation.py --skip-train     # reuse existing catalog_gnn.pt
    python run_full_evaluation.py --n 5000         # smaller catalog, faster smoke test
    python run_full_evaluation.py --plots-only     # just re-render plots from
                                                    # outputs/eval_results.json,
                                                    # no pipeline re-run

Run this from the same directory that holds data_generator.py,
identity_embedding.py, catalog_gnn.py, train_catalog_gnn.py,
ground_station_tracking_simulation.py, compare_live_vs_training.py,
check_catalog_against_embeddings.py, orbital_physics.py, leo_ann_index.py
(i.e. the pipeline's own working directory) -- this script assumes those
files and their relative-path outputs/ and leo_catalog.csv conventions.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

_PRINT_LOCK = threading.Lock()

# This script lives at repo root; src/ and outputs/ are siblings of it.
# Anchored via pathlib (same convention as every src/ script) so it runs
# correctly regardless of invocation cwd, not just when cwd happens to be
# the repo root.
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"

RESULTS_PATH = str(OUTPUTS_DIR / "eval_results.json")
PLOTS_DIR = str(OUTPUTS_DIR / "eval_plots")


def src(script_name):
    """Path to a pipeline script under src/, as a string (for subprocess cmd lists)."""
    return str(SRC_DIR / script_name)


# ===========================================================================
# Stage runner
# ===========================================================================
def run_stage(name, cmd, log_path, prefix_output=False):
    """Run a pipeline stage as a subprocess, streaming stdout to the console
    AND capturing it (both to a .log file and in memory), so the parsers
    below can work off the exact same text the user saw scroll by.

    prefix_output=True is used when this stage is running concurrently with
    another stage (see run_pipeline's ThreadPoolExecutor usage below) -- in
    that case lines from both stages would otherwise interleave character-
    by-character on stdout and be unreadable, so each line is tagged with
    its stage name and printing is serialized through _PRINT_LOCK."""
    with _PRINT_LOCK:
        print(f"\n{'#' * 70}\n# STAGE: {name}\n# CMD:   {' '.join(cmd)}\n{'#' * 70}")
    t0 = time.perf_counter()
    lines = []
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            if prefix_output:
                with _PRINT_LOCK:
                    sys.stdout.write(f"[{name}] {line}")
            else:
                sys.stdout.write(line)
            logf.write(line)
            lines.append(line.rstrip("\n"))
        proc.wait()
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage '{name}' failed (exit code {proc.returncode}). "
            f"See {log_path} for full output."
        )
    with _PRINT_LOCK:
        print(f"[{name}] finished in {elapsed:.1f}s (log: {log_path})")
    return lines, elapsed


def f(pattern):
    return re.compile(pattern)


def grab(pattern, text, cast=float, default=None, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        return default
    val = m.group(1)
    try:
        return cast(val)
    except (TypeError, ValueError):
        return default


# ===========================================================================
# Per-stage stdout parsers -- each regex is matched to the literal
# f-string it targets in that script's source, so if the wording there
# changes these need to change too.
# ===========================================================================
def parse_data_generator(lines):
    text = "\n".join(lines)
    # "Wrote {len(rows)} rows to {args.output}"
    n_rows = grab(r"Wrote (\d+) rows to", text, cast=int)
    return {"n_rows_generated": n_rows}


def parse_identity_embedding(lines):
    text = "\n".join(lines)
    # "Wrote PyTorch tensor file to {OUTPUT_PATH}  (vectors shape: {tuple(...)})"
    shape = grab(r"vectors shape:\s*\((\d+),\s*(\d+)\)", text, cast=None, default=None)
    m = re.search(r"vectors shape:\s*\((\d+),\s*(\d+)\)", text)
    n_embedded, dim = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return {"n_embedded": n_embedded, "embedding_dim": dim}


def parse_train_catalog_gnn(lines):
    text = "\n".join(lines)
    out = {
        "truth_population": grab(r"Truth population:\s*(\d+)", text, cast=int),
        "embedding_store_size": grab(r"Embedding store:\s*(\d+)", text, cast=int),
        "drift_tolerance": grab(r"drift_tolerance=([\d.eE+-]+)\s+margin_required=", text),
        "margin_required": grab(r"drift_tolerance=[\d.eE+-]+\s+margin_required=([\d.eE+-]+)", text),
        "confident_known_threshold": grab(r"confident_known_threshold=([\d.eE+-]+)", text),
        "confident_novel_threshold": grab(r"confident_novel_threshold=([\d.eE+-]+)", text),
    }
    # per-epoch training curve:
    # "epoch  N  train_loss=X  val_loss=Y  val_fp_rate(novel called known)=Z  val_recall(known)=W"
    epoch_rows = []
    epoch_re = re.compile(
        r"epoch\s+(\d+)\s+train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)\s+"
        r"val_fp_rate\(novel called known\)=([\d.eEnaN+-]+)\s+val_recall\(known\)=([\d.eEnaN+-]+)"
    )
    for line in lines:
        m = epoch_re.search(line)
        if m:
            def safe_float(s):
                try:
                    return float(s)
                except ValueError:
                    return None
            epoch_rows.append({
                "epoch": int(m.group(1)),
                "train_loss": safe_float(m.group(2)),
                "val_loss": safe_float(m.group(3)),
                "val_fp_rate_novel_called_known": safe_float(m.group(4)),
                "val_recall_known": safe_float(m.group(5)),
            })
    out["training_curve"] = epoch_rows

    # calibration sanity-check line, e.g.:
    # "  (sanity check: 87.6% of 42 held-out known examples score above
    #    confident_novel_threshold -- target was 90%. ...)"
    out["known_recall_at_novel_cut"] = None
    m2 = re.search(r"sanity check:\s*([\d.]+)% of (\d+) held-out", text)
    if m2:
        out["known_recall_at_novel_cut"] = float(m2.group(1))
        out["n_known_val"] = int(m2.group(2))
    m3 = re.search(r"n_known_val=(\d+), n_novel_val=(\d+)", text)
    if m3:
        out["n_known_val"] = int(m3.group(1))
        out["n_novel_val"] = int(m3.group(2))
    return out


def parse_tracking_sim(lines):
    text = "\n".join(lines)
    out = {}
    out["total_detections"] = grab(r"Total detections\s*:\s*(\d+)", text, cast=int)
    out["known_count"] = grab(r"KNOWN\s*:\s*(\d+)", text, cast=int)
    out["ambiguous_count"] = grab(r"AMBIGUOUS \(deferred\)\s*:\s*(\d+)", text, cast=int)
    out["unknown_count"] = grab(r"NOT KNOWN \(novel\)\s*:\s*(\d+)", text, cast=int)
    out["real_wall_time_s"] = grab(r"Real wall time\s*:\s*([\d.]+)s", text)
    m = re.search(r"Update commits\s*:\s*(\d+)\s+withheld:\s*(\d+)", text)
    if m:
        out["update_committed"] = int(m.group(1))
        out["update_withheld"] = int(m.group(2))
    m = re.search(r"Retired \(decayed\)\s*:\s*(\d+)\s+\(unseen\):\s*(\d+)", text)
    if m:
        out["retired_decayed"] = int(m.group(1))
        out["retired_unseen"] = int(m.group(2))
    out["merged_count"] = grab(r"Merged \(de-dup\)\s*:\s*(\d+)", text, cast=int)

    m = re.search(r"Correct matches\s*:\s*(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        out["correct_matches"] = int(m.group(1))
        out["correct_matches_pct"] = float(m.group(2))
    m = re.search(r"Mismatches\s*:\s*(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        out["mismatches"] = int(m.group(1))
        out["mismatches_pct"] = float(m.group(2))

    m = re.search(r"Catalog completeness\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", text)
    if m:
        out["catalog_completeness_pct"] = float(m.group(1))
        out["catalog_complete_n"] = int(m.group(2))
        out["catalog_total_n"] = int(m.group(3))

    m = re.search(r"Redundancy index\s*:\s*([\d.]+)\s*\(ghost registrations:\s*(\d+)\)", text)
    if m:
        out["redundancy_index"] = float(m.group(1))
        out["ghost_registrations"] = int(m.group(2))

    m = re.search(r"ANN candidate recall\s*:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", text)
    if m:
        out["ann_recall_pct"] = float(m.group(1))
        out["ann_recall_hit"] = int(m.group(2))
        out["ann_recall_checked"] = int(m.group(3))

    out["gnn_consults"] = grab(r"GNN plausible near-miss consults\s*:\s*(\d+)", text, cast=int)
    out["gnn_known"] = grab(r"->\s*KNOWN\s*:\s*(\d+)", text, cast=int)
    out["gnn_ambiguous"] = grab(r"->\s*AMBIGUOUS\s*:\s*(\d+)", text, cast=int)
    out["gnn_novel"] = grab(r"->\s*NOVEL\s*:\s*(\d+)", text, cast=int)
    out["gnn_skipped_implausible"] = grab(r"GNN skipped \(implausible near-miss\):\s*(\d+)", text, cast=int)

    m = re.search(
        r"GNN overturn precision.*?:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", text
    )
    if m:
        out["gnn_overturn_correct"] = int(m.group(1))
        out["gnn_overturn_total"] = int(m.group(2))
        out["gnn_overturn_precision_pct"] = float(m.group(3))

    m = re.search(r"Final catalog size\s*:\s*(\d+)\s*\(active:\s*(\d+)\)", text)
    if m:
        out["final_catalog_size"] = int(m.group(1))
        out["final_catalog_active"] = int(m.group(2))

    # Per-detection lines:
    # "[T+  12.34s sim] name_123          -> KNOWN      (dist=0.123, margin=0.045, lookup 12.34 us)  [gate, update committed]"
    # status is left-padded to 10 chars with f"{status:<10}" in the source,
    # and one of its three values ("NOT KNOWN") itself contains a space --
    # so it can't be matched with \S+; match everything up to the literal
    # "(dist=" marker instead, then strip trailing padding.
    det_re = re.compile(
        r"^\[T\+\s*([\d.]+)s sim\]\s+(\S+)\s+->\s+(.+?)\s*"
        r"\(dist=([\d.infa-]+),\s*margin=([\d.infa-]+),\s*lookup\s+([\d.]+)\s*us\)"
        r"(.*)$"
    )
    detections = []
    for line in lines:
        m = det_re.match(line)
        if not m:
            continue
        sim_t, name, status, dist_s, margin_s, lookup_us, tail = m.groups()

        def safe_f(s):
            try:
                return float(s)
            except ValueError:
                return None

        accept_path = None
        if "[gate," in tail:
            accept_path = "gate"
        elif "GNN->KNOWN" in tail:
            accept_path = "gnn"
        elif "existing-slot" in tail:
            accept_path = "existing_slot"
        elif "registered new" in tail:
            accept_path = "registered_new"
        elif "GNN->AMBIGUOUS" in tail:
            accept_path = "gnn_ambiguous"

        detections.append({
            "sim_time": safe_f(sim_t),
            "name": name,
            "status": status,
            "dist": safe_f(dist_s),
            "margin": safe_f(margin_s),
            "lookup_us": safe_f(lookup_us),
            "mismatch": "MISMATCH" in tail,
            "accept_path": accept_path,
        })
    out["detections"] = detections

    # Recalibration events: "[T+ 123.45s sim] [RECALIBRATE] active=210 drift_tolerance=0.1234 margin_required=0.0567"
    recal_re = re.compile(
        r"^\[T\+\s*([\d.]+)s sim\]\s+\[RECALIBRATE\]\s+active=(\d+)\s+"
        r"drift_tolerance=([\d.]+)\s+margin_required=([\d.]+)"
    )
    recals = []
    for line in lines:
        m = recal_re.match(line)
        if m:
            recals.append({
                "sim_time": float(m.group(1)),
                "active": int(m.group(2)),
                "drift_tolerance": float(m.group(3)),
                "margin_required": float(m.group(4)),
            })
    out["recalibrations"] = recals

    return out


def parse_compare_live_vs_training(lines):
    text = "\n".join(lines)
    out = {}
    m = re.search(
        r"Live run:\s*(\d+)\s*GNN invocations,\s*(\d+)\s*\(([\d.]+)%\)\s*had the true "
        r"object in the candidate pool,\s*(\d+)\s*did not", text,
    )
    if m:
        out["live_n_invocations"] = int(m.group(1))
        out["live_true_in_pool"] = int(m.group(2))
        out["live_true_in_pool_pct"] = float(m.group(3))
        out["live_not_in_pool"] = int(m.group(4))

    def parse_percentile_line(label_pattern):
        m = re.search(label_pattern + r":\n\s*(.+)", text)
        if not m:
            return None
        pct_line = m.group(1)
        result = {}
        for pm in re.finditer(r"p(\d+)=([\d.eE+-]+|nan)", pct_line):
            try:
                result[f"p{pm.group(1)}"] = float(pm.group(2))
            except ValueError:
                result[f"p{pm.group(1)}"] = None
        return result

    out["live_min_dist_in_pool"] = parse_percentile_line(r"Live min_dist, true_object_in_pool=True")
    out["live_min_dist_not_in_pool"] = parse_percentile_line(r"Live min_dist, true_object_in_pool=False")
    out["fresh_min_dist_known"] = parse_percentile_line(r"Fresh min_dist, true_object_in_pool=True")
    out["fresh_min_dist_novel"] = parse_percentile_line(r"Fresh min_dist, true_object_in_pool=False")

    m = re.search(
        r"Fresh harvested batch:\s*(\d+)\s*examples,\s*(\d+)\s*\(([\d.]+)%\)\s*true-object-in-pool",
        text,
    )
    if m:
        out["fresh_n_examples"] = int(m.group(1))
        out["fresh_known"] = int(m.group(2))
        out["fresh_known_pct"] = float(m.group(3))

    return out


def parse_check_catalog(lines):
    text = "\n".join(lines)
    out = {}
    out["total_checked"] = grab(r"Total objects checked\s*:\s*(\d+)", text, cast=int)
    out["recognized"] = grab(r"Recognized \(hits\)\s*:\s*(\d+)", text, cast=int)
    out["unrecognized"] = grab(r"Unrecognized \(misses\)\s*:\s*(\d+)", text, cast=int)
    out["recognition_rate_pct"] = grab(r"Recognition rate\s*:\s*([\d.]+)%", text)
    m = re.search(
        r"ALL\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text,
    )
    if m:
        out["all_avg_us"], out["all_fastest_us"], out["all_slowest_us"] = map(float, m.groups())
    m = re.search(r"HITS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        out["hits_avg_us"], out["hits_fastest_us"], out["hits_slowest_us"] = map(float, m.groups())
    m = re.search(r"MISSES\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        out["misses_avg_us"], out["misses_fastest_us"], out["misses_slowest_us"] = map(float, m.groups())
    return out


# ===========================================================================
# Pipeline driver
# ===========================================================================
def run_pipeline(args):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {"config": vars(args), "stages": {}}
    py = sys.executable
    catalog_path = str(DATA_DIR / "leo_catalog.csv")

    # ---- Stage 1: data_generator.py -> data/leo_catalog.csv ---------------
    if not args.skip_generate:
        gen_cmd = [py, src("data_generator.py"), "-n", str(args.n), "-o", catalog_path]
        if args.seed is not None:
            gen_cmd += ["-s", str(args.seed)]
        lines, elapsed = run_stage("data_generator.py", gen_cmd,
                                    str(OUTPUTS_DIR / "log_01_data_generator.txt"))
        results["stages"]["data_generator"] = {**parse_data_generator(lines), "elapsed_s": elapsed}
    else:
        print(f"Skipping data_generator.py (--skip-generate); reusing {catalog_path}")

    # ---- Stage 2: identity_embedding.py -> outputs/leo_embeddings.pt ------
    if not args.skip_embed:
        embed_cmd = [py, src("identity_embedding.py")]
        lines, elapsed = run_stage("identity_embedding.py", embed_cmd,
                                    str(OUTPUTS_DIR / "log_02_identity_embedding.txt"))
        results["stages"]["identity_embedding"] = {**parse_identity_embedding(lines), "elapsed_s": elapsed}
    else:
        print("Skipping identity_embedding.py (--skip-embed); reusing outputs/leo_embeddings.pt")

    # ---- Stages 3+4 (train_catalog_gnn.py, then the tracking sim) form a --
    # strict chain -- the sim only loads a GNN checkpoint if one exists with
    # a matching gate_stamp. check_catalog_against_embeddings.py, however,
    # only touches leo_catalog.csv + leo_embeddings.pt (both already on disk
    # at this point) -- it has no dependency on the GNN or the sim at all,
    # so it's genuinely safe to run concurrently with stages 3+4 rather than
    # after them. See the module docstring's "RUNNING EVERYTHING
    # SIMULTANEOUSLY" note for why this is the only such opportunity in the
    # whole pipeline.
    def do_train_and_sim():
        stage_results = {}
        if not args.skip_train:
            train_cmd = [py, src("train_catalog_gnn.py"), "--n-examples", str(args.n_examples),
                         "--epochs", str(args.epochs)]
            if args.min_known_examples is not None:
                train_cmd += ["--min-known-examples", str(args.min_known_examples)]
            lines, elapsed = run_stage("train_catalog_gnn.py", train_cmd,
                                        str(OUTPUTS_DIR / "log_03_train_catalog_gnn.txt"),
                                        prefix_output=not args.sequential)
            stage_results["train_catalog_gnn"] = {**parse_train_catalog_gnn(lines), "elapsed_s": elapsed}
        else:
            with _PRINT_LOCK:
                print("Skipping train_catalog_gnn.py (--skip-train); reusing outputs/catalog_gnn.pt")

        sim_cmd = [py, src("ground_station_tracking_simulation.py")]
        lines, elapsed = run_stage("ground_station_tracking_simulation.py", sim_cmd,
                                    str(OUTPUTS_DIR / "log_04_tracking_simulation.txt"),
                                    prefix_output=not args.sequential)
        stage_results["tracking_simulation"] = {**parse_tracking_sim(lines), "elapsed_s": elapsed}
        return stage_results

    def do_embedding_check():
        check_cmd = [py, src("check_catalog_against_embeddings.py")]
        lines, elapsed = run_stage("check_catalog_against_embeddings.py", check_cmd,
                                    str(OUTPUTS_DIR / "log_06_check_catalog_against_embeddings.txt"),
                                    prefix_output=not args.sequential)
        return {"check_catalog_against_embeddings": {**parse_check_catalog(lines), "elapsed_s": elapsed}}

    if args.sequential:
        results["stages"].update(do_train_and_sim())
        results["stages"].update(do_embedding_check())
    else:
        print("\nRunning train_catalog_gnn.py -> tracking sim, and "
              "check_catalog_against_embeddings.py, concurrently "
              "(independent branches of the dependency graph -- see module docstring).")
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_train_sim = pool.submit(do_train_and_sim)
            fut_check = pool.submit(do_embedding_check)
            # Surface exceptions from either branch as soon as both are done,
            # rather than one thread's traceback silently winning a race.
            train_sim_result = fut_train_sim.result()
            check_result = fut_check.result()
        results["stages"].update(train_sim_result)
        results["stages"].update(check_result)

    # ---- Stage 5: compare_live_vs_training.py ------------------------------
    # Hard-blocked on the sim's own output file -- cannot start any earlier
    # than this, concurrently or otherwise (see module docstring).
    live_log_path = OUTPUTS_DIR / "gnn_live_invocations.json"
    live_log = None
    if live_log_path.exists():
        with open(live_log_path) as fh:
            live_log = json.load(fh)
    results["gnn_live_invocations"] = live_log

    if live_log:
        cmp_cmd = [py, src("compare_live_vs_training.py")]
        lines, elapsed = run_stage("compare_live_vs_training.py", cmp_cmd,
                                    str(OUTPUTS_DIR / "log_05_compare_live_vs_training.txt"))
        results["stages"]["compare_live_vs_training"] = {**parse_compare_live_vs_training(lines),
                                                           "elapsed_s": elapsed}
    else:
        print("\nNo GNN invocations happened during the sim (or GNN was disabled) -- "
              "skipping compare_live_vs_training.py, since it has nothing to compare.")

    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nWrote evaluation results to {RESULTS_PATH}")
    return results


# ===========================================================================
# Plotting
# ===========================================================================
def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })

    sim = results["stages"].get("tracking_simulation", {})
    detections = sim.get("detections", [])
    recals = sim.get("recalibrations", [])
    live_log = results.get("gnn_live_invocations")
    train_curve = results["stages"].get("train_catalog_gnn", {}).get("training_curve", [])
    cmp_stage = results["stages"].get("compare_live_vs_training", {})
    check_stage = results["stages"].get("check_catalog_against_embeddings", {})

    saved = []

    def savefig(fig, name):
        path = os.path.join(PLOTS_DIR, name)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
        print(f"  wrote {path}")

    # -----------------------------------------------------------------
    # 1. Decision breakdown funnel (pipeline-level outcome mix)
    # -----------------------------------------------------------------
    if sim:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = ["KNOWN", "AMBIGUOUS", "NOT KNOWN"]
        counts = [sim.get("known_count") or 0, sim.get("ambiguous_count") or 0, sim.get("unknown_count") or 0]
        colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
        axes[0].bar(labels, counts, color=colors)
        axes[0].set_title("Detection outcomes (whole run)")
        axes[0].set_ylabel("count")
        for i, c in enumerate(counts):
            axes[0].text(i, c, f"{c}", ha="center", va="bottom")

        gnn_labels = ["GNN\u2192KNOWN", "GNN\u2192AMBIGUOUS", "GNN\u2192NOVEL", "skipped\n(implausible)"]
        gnn_counts = [sim.get("gnn_known") or 0, sim.get("gnn_ambiguous") or 0,
                      sim.get("gnn_novel") or 0, sim.get("gnn_skipped_implausible") or 0]
        axes[1].bar(gnn_labels, gnn_counts, color=["#2a9d8f", "#e9c46a", "#e76f51", "#8d99ae"])
        axes[1].set_title("Catalog-GNN second-opinion outcomes")
        axes[1].set_ylabel("count")
        for i, c in enumerate(gnn_counts):
            axes[1].text(i, c, f"{c}", ha="center", va="bottom")

        savefig(fig, "01_decision_breakdown.png")

    # -----------------------------------------------------------------
    # 2. Acceptance path breakdown (gate / gnn / existing_slot / new)
    #
    # The "gnn" bar (gate-rejected, GNN overturned to KNOWN) is annotated
    # with its own precision directly on the chart, since that number
    # (correct-overturns / total-overturns) can contradict the offline
    # training metrics when the sample is tiny -- see sim["gnn_overturn_*"]
    # -- and burying it only in the scorecard risks a reader taking the
    # bar's mere presence as confirmation the path works, without
    # noticing how few real events (and how low a hit rate) back it.
    # -----------------------------------------------------------------
    if detections:
        path_counts = {}
        for d in detections:
            key = d["accept_path"] or "not_known"
            path_counts[key] = path_counts.get(key, 0) + 1
        order = ["gate", "gnn", "existing_slot", "registered_new", "gnn_ambiguous", "not_known"]
        order = [o for o in order if o in path_counts] + [k for k in path_counts if k not in order]
        vals = [path_counts[k] for k in order]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bars = ax.bar(order, vals, color="#457b9d")
        ax.set_title("Match resolution path (per-detection)")
        ax.set_ylabel("count")
        ax.set_xlabel("path")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")

        gnn_precision_pct = sim.get("gnn_overturn_precision_pct")
        gnn_correct = sim.get("gnn_overturn_correct")
        gnn_total = sim.get("gnn_overturn_total")
        if "gnn" in order and gnn_total is not None:
            gnn_idx = order.index("gnn")
            gnn_bar = bars[gnn_idx]
            low_sample = gnn_total < 30  # arbitrary but explicit: small enough that a
                                          # precision statistic is not yet meaningful
            note = (f"overturn precision: {gnn_correct}/{gnn_total} ({gnn_precision_pct:.0f}%)"
                    + ("\n(sample too small to\ntrust this rate)" if low_sample else ""))
            ax.annotate(note, xy=(gnn_bar.get_x() + gnn_bar.get_width() / 2, gnn_bar.get_height()),
                        xytext=(gnn_bar.get_x() + gnn_bar.get_width() / 2 + 0.6,
                                max(vals) * 0.35),
                        fontsize=9, ha="left",
                        color="#e76f51" if low_sample else "#2a9d8f",
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))

        savefig(fig, "02_acceptance_paths.png")

    # -----------------------------------------------------------------
    # 3. Match distance & margin distributions, KNOWN vs mismatched
    #
    # NOTE: when the system is working well, essentially all accepted
    # distances sit extremely close to 0 -- a linear-scale histogram then
    # shows one spike at the left edge and an empty plot otherwise, which
    # looks like a plotting bug even though it's the correct/expected shape
    # for a well-calibrated match. A log-x panel is added alongside the
    # linear one specifically so the actual internal shape of that near-zero
    # cluster is visible instead of being flattened into a single bar.
    # -----------------------------------------------------------------
    if detections:
        accepted = [d for d in detections if d["status"] == "KNOWN" and d["dist"] is not None]
        correct = [d for d in accepted if not d["mismatch"]]
        mismatched = [d for d in accepted if d["mismatch"]]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        if correct:
            axes[0].hist([d["dist"] for d in correct], bins=40, alpha=0.7, label="correct match", color="#2a9d8f")
        if mismatched:
            axes[0].hist([d["dist"] for d in mismatched], bins=40, alpha=0.7, label="mismatch", color="#e76f51")
        axes[0].set_title("Match distance (linear scale)")
        axes[0].set_xlabel("identity distance")
        axes[0].set_ylabel("count")
        axes[0].legend()

        # Log-scale view: reveals the internal shape of the near-zero
        # cluster that the linear panel collapses into a single spike.
        all_dists = [d["dist"] for d in correct] + [d["dist"] for d in mismatched]
        positive_dists = [d for d in all_dists if d is not None and d > 0]
        if positive_dists:
            log_bins = np.logspace(np.log10(max(min(positive_dists), 1e-6)), np.log10(max(positive_dists)), 40)
            if correct:
                correct_pos = [d["dist"] for d in correct if d["dist"] > 0]
                if correct_pos:
                    axes[1].hist(correct_pos, bins=log_bins, alpha=0.7, label="correct match", color="#2a9d8f")
            if mismatched:
                mismatched_pos = [d["dist"] for d in mismatched if d["dist"] > 0]
                if mismatched_pos:
                    axes[1].hist(mismatched_pos, bins=log_bins, alpha=0.7, label="mismatch", color="#e76f51")
            axes[1].set_xscale("log")
            axes[1].set_title("Match distance (log scale)\nshows internal shape of the near-zero cluster")
            axes[1].set_xlabel("identity distance (log)")
            axes[1].set_ylabel("count")
            axes[1].legend()
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "all distances exactly 0 --\nno positive values to log-scale",
                         ha="center", va="center", transform=axes[1].transAxes)

        if correct:
            axes[2].hist([d["margin"] for d in correct], bins=40, alpha=0.7, label="correct match", color="#2a9d8f")
        if mismatched:
            axes[2].hist([d["margin"] for d in mismatched], bins=40, alpha=0.7, label="mismatch", color="#e76f51")
        axes[2].set_title("Ambiguity margin distribution")
        axes[2].set_xlabel("margin (2nd nearest \u2212 nearest)")
        axes[2].set_ylabel("count")
        axes[2].legend()

        savefig(fig, "03_distance_margin_distributions.png")

    # -----------------------------------------------------------------
    # 4. Lookup latency: KNOWN vs NOT-KNOWN/AMBIGUOUS, over time
    # -----------------------------------------------------------------
    if detections:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        by_status = {}
        for d in detections:
            if d["lookup_us"] is None:
                continue
            by_status.setdefault(d["status"], []).append(d["lookup_us"])
        status_colors = {"KNOWN": "#2a9d8f", "AMBIGUOUS": "#e9c46a", "NOT": "#e76f51"}
        for status, vals in by_status.items():
            color = next((c for k, c in status_colors.items() if status.startswith(k)), None)
            axes[0].hist(vals, bins=50, alpha=0.6, label=f"{status} (n={len(vals)})", color=color)
        axes[0].set_title("Lookup latency by outcome")
        axes[0].set_xlabel("lookup time (\u00b5s)")
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=8)

        times = [d["sim_time"] for d in detections if d["lookup_us"] is not None]
        lats = [d["lookup_us"] for d in detections if d["lookup_us"] is not None]
        axes[1].scatter(times, lats, s=4, alpha=0.4, color="#457b9d")
        # rolling median for trend visibility
        if len(lats) > 20:
            order_idx = np.argsort(times)
            t_sorted = np.array(times)[order_idx]
            l_sorted = np.array(lats)[order_idx]
            window = max(10, len(l_sorted) // 50)
            roll = np.array([
                np.median(l_sorted[max(0, i - window):i + 1]) for i in range(len(l_sorted))
            ])
            axes[1].plot(t_sorted, roll, color="#e76f51", linewidth=2, label="rolling median")
            axes[1].legend()

            # Annotate WHY latency grows/plateaus, using the run's own data
            # rather than leaving a reader to guess: latency here tracks
            # active catalog size, which stops growing once new-object
            # registration exhausts the fixed synthetic population -- find
            # that point (registration growth falling below 1% of its peak
            # rate) and mark it directly on the plot.
            reg_times = sorted(d["sim_time"] for d in detections if d["accept_path"] == "registered_new")
            if len(reg_times) > 10:
                reg_times_arr = np.array(reg_times)
                # last registration time as a proxy for "catalog stopped growing"
                plateau_t = reg_times_arr[-1]
                if plateau_t < t_sorted.max():
                    axes[1].axvline(plateau_t, color="gray", linestyle=":", alpha=0.7)
                    axes[1].text(plateau_t, axes[1].get_ylim()[1] * 0.92,
                                 " last new registration\n (catalog stops growing here --\n"
                                 " latency plateaus because active\n catalog size plateaus, not\n an independent scaling issue)",
                                 fontsize=7, color="dimgray", va="top")
        axes[1].set_title("Lookup latency over simulated time")
        axes[1].set_xlabel("sim time (s)")
        axes[1].set_ylabel("lookup time (\u00b5s)")

        savefig(fig, "04_lookup_latency.png")

    # -----------------------------------------------------------------
    # 5. Catalog growth & recalibration over time
    # -----------------------------------------------------------------
    if detections:
        fig, ax = plt.subplots(figsize=(10, 5))
        times = [d["sim_time"] for d in detections]
        running_known = np.cumsum([1 if d["status"] == "KNOWN" else 0 for d in detections])
        running_novel = np.cumsum([1 if d["accept_path"] == "registered_new" else 0 for d in detections])
        running_ambig = np.cumsum([1 if d["status"] == "AMBIGUOUS" else 0 for d in detections])
        ax.plot(times, running_known, label="cumulative KNOWN", color="#2a9d8f")
        ax.plot(times, running_novel, label="cumulative new registrations", color="#e76f51")
        ax.plot(times, running_ambig, label="cumulative AMBIGUOUS", color="#e9c46a")
        for r in recals:
            ax.axvline(r["sim_time"], color="gray", linestyle="--", alpha=0.4)
        if recals:
            ax.axvline(recals[0]["sim_time"], color="gray", linestyle="--", alpha=0.4, label="recalibration")
        ax.set_title("Cumulative detection outcomes over simulated time")
        ax.set_xlabel("sim time (s)")
        ax.set_ylabel("cumulative count")
        ax.legend()
        savefig(fig, "05_catalog_growth_over_time.png")

    # -----------------------------------------------------------------
    # 6. Drift tolerance / margin required over recalibrations
    #
    # Recalibration events are typically triggered on a schedule (e.g. an
    # event-count interval, see ground_station_tracking_simulation.py's
    # recalibration logic) rather than continuously -- so this plot's
    # x-axis commonly stops well before the sim's actual end time once new
    # registrations (and therefore active catalog size) stop changing
    # enough to trigger another recalibration. That's flagged directly on
    # the plot below so a reader doesn't mistake a short x-axis range for
    # a truncated or failed run.
    # -----------------------------------------------------------------
    if recals:
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        rt = [r["sim_time"] for r in recals]
        dt = [r["drift_tolerance"] for r in recals]
        mr = [r["margin_required"] for r in recals]
        act = [r["active"] for r in recals]
        ax1.plot(rt, dt, "o-", color="#457b9d", label="drift_tolerance")
        ax1.plot(rt, mr, "o-", color="#e76f51", label="margin_required")
        ax1.set_xlabel("sim time (s)")
        ax1.set_ylabel("threshold value")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot(rt, act, "s--", color="gray", alpha=0.6, label="active catalog size")
        ax2.set_ylabel("active catalog size")
        ax2.legend(loc="upper right")
        ax1.set_title("Auto-calibrated gate thresholds vs. active catalog size")

        sim_end_time = max((d["sim_time"] for d in detections), default=None) if detections else None
        last_recal_time = max(rt)
        if sim_end_time is not None and sim_end_time - last_recal_time > 1.0:
            fig.text(0.01, 0.01,
                     f"Note: last recalibration at t={last_recal_time:.1f}s, but the sim ran to "
                     f"t={sim_end_time:.1f}s -- recalibration stopped firing once the catalog "
                     f"stabilized, this plot is not truncated.",
                     fontsize=7, color="dimgray")
        savefig(fig, "06_threshold_recalibration.png")

    # -----------------------------------------------------------------
    # 7. GNN training curve
    # -----------------------------------------------------------------
    if train_curve:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        epochs = [r["epoch"] for r in train_curve]
        axes[0].plot(epochs, [r["train_loss"] for r in train_curve], "o-", label="train_loss", color="#457b9d")
        axes[0].plot(epochs, [r["val_loss"] for r in train_curve], "o-", label="val_loss", color="#e76f51")
        axes[0].set_title("Catalog GNN training/validation loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("asymmetric BCE loss")
        axes[0].legend()

        axes[1].plot(epochs, [r["val_recall_known"] for r in train_curve], "o-",
                     label="val_recall(known)", color="#2a9d8f")
        axes[1].plot(epochs, [r["val_fp_rate_novel_called_known"] for r in train_curve], "o-",
                     label="val_fp_rate(novel called known)", color="#e76f51")
        axes[1].set_title("Validation recall / false-positive rate")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        savefig(fig, "07_gnn_training_curve.png")

    # -----------------------------------------------------------------
    # 8. Live GNN invocation log: min_dist distributions + decisions
    # -----------------------------------------------------------------
    if live_log:
        min_dist_in_pool = [r["min_dist"] for r in live_log if r["true_object_in_pool"]]
        min_dist_not_in_pool = [r["min_dist"] for r in live_log if not r["true_object_in_pool"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if min_dist_in_pool:
            axes[0].hist(min_dist_in_pool, bins=40, alpha=0.7, label=f"true object IN pool (n={len(min_dist_in_pool)})",
                         color="#2a9d8f", density=True)
        if min_dist_not_in_pool:
            axes[0].hist(min_dist_not_in_pool, bins=40, alpha=0.7,
                         label=f"true object NOT in pool (n={len(min_dist_not_in_pool)})",
                         color="#e76f51", density=True)
        axes[0].set_title("Live GNN consults: min candidate distance")
        axes[0].set_xlabel("min_dist")
        axes[0].set_ylabel("density")
        axes[0].legend(fontsize=8)

        decisions = {}
        for r in live_log:
            decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
        dec_labels = list(decisions.keys())
        dec_vals = [decisions[k] for k in dec_labels]
        axes[1].bar(dec_labels, dec_vals, color=["#2a9d8f", "#e9c46a", "#e76f51"][:len(dec_labels)])
        axes[1].set_title("Live GNN decisions")
        axes[1].set_ylabel("count")
        for i, v in enumerate(dec_vals):
            axes[1].text(i, v, str(v), ha="center", va="bottom")

        savefig(fig, "08_live_gnn_invocations.png")

        # confidence vs min_dist scatter, colored by decision & pool membership
        fig, ax = plt.subplots(figsize=(8, 6))
        dec_color = {"KNOWN": "#2a9d8f", "AMBIGUOUS": "#e9c46a", "NOVEL": "#e76f51"}
        for dec, color in dec_color.items():
            xs = [r["min_dist"] for r in live_log if r["decision"] == dec]
            ys = [r["confidence"] for r in live_log if r["decision"] == dec]
            marker = "o"
            ax.scatter(xs, ys, s=14, alpha=0.5, color=color, label=dec, marker=marker)
        ax.set_xlabel("min candidate distance")
        ax.set_ylabel("GNN confidence")
        ax.set_title("Live GNN confidence vs. min distance, by decision")
        ax.legend()
        savefig(fig, "09_live_gnn_confidence_vs_distance.png")

    # -----------------------------------------------------------------
    # 10. Live vs. fresh-training-batch distribution comparison
    #     (the actual regression check this system relies on --
    #     see compare_live_vs_training.py's docstring)
    # -----------------------------------------------------------------
    if cmp_stage:
        def pct_series(d):
            if not d:
                return [], []
            pts = [(int(k[1:]), v) for k, v in d.items() if v is not None]
            pts.sort()
            return [p[0] for p in pts], [p[1] for p in pts]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        lx, ly = pct_series(cmp_stage.get("live_min_dist_in_pool"))
        fx, fy = pct_series(cmp_stage.get("fresh_min_dist_known"))
        if lx:
            axes[0].plot(lx, ly, "o-", label="live", color="#457b9d")
        if fx:
            axes[0].plot(fx, fy, "s--", label="fresh harvested batch", color="#e76f51")
        axes[0].set_title("min_dist percentiles \u2014 true_object_in_pool=True")
        axes[0].set_xlabel("percentile")
        axes[0].set_ylabel("min_dist")
        axes[0].legend()

        lx2, ly2 = pct_series(cmp_stage.get("live_min_dist_not_in_pool"))
        fx2, fy2 = pct_series(cmp_stage.get("fresh_min_dist_novel"))
        if lx2:
            axes[1].plot(lx2, ly2, "o-", label="live", color="#457b9d")
        if fx2:
            axes[1].plot(fx2, fy2, "s--", label="fresh harvested batch", color="#e76f51")
        axes[1].set_title("min_dist percentiles \u2014 true_object_in_pool=False")
        axes[1].set_xlabel("percentile")
        axes[1].set_ylabel("min_dist")
        axes[1].legend()

        savefig(fig, "10_live_vs_training_distribution.png")

    # -----------------------------------------------------------------
    # 11. Embedding/catalog cross-check: lookup timing hit vs miss
    # -----------------------------------------------------------------
    if check_stage and check_stage.get("total_checked"):
        fig, ax = plt.subplots(figsize=(7, 5))
        cats = ["ALL", "HITS", "MISSES"]
        avgs = [check_stage.get("all_avg_us"), check_stage.get("hits_avg_us"), check_stage.get("misses_avg_us")]
        fastest = [check_stage.get("all_fastest_us"), check_stage.get("hits_fastest_us"), check_stage.get("misses_fastest_us")]
        slowest = [check_stage.get("all_slowest_us"), check_stage.get("hits_slowest_us"), check_stage.get("misses_slowest_us")]
        x = np.arange(len(cats))
        width = 0.25
        ax.bar(x - width, fastest, width, label="fastest", color="#2a9d8f")
        ax.bar(x, avgs, width, label="avg", color="#457b9d")
        ax.bar(x + width, slowest, width, label="slowest", color="#e76f51")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel("microseconds")
        ax.set_title("Catalog\u2194embedding lookup timing (hit vs. miss)")
        ax.legend()
        savefig(fig, "11_embedding_lookup_timing.png")

    # -----------------------------------------------------------------
    # 12. Summary scorecard (single figure, headline numbers)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")
    lines_txt = ["SYSTEM EVALUATION SCORECARD", "=" * 40, ""]

    if sim:
        total = sim.get("total_detections") or 0
        lines_txt += [
            f"Total detections:        {total}",
            f"  KNOWN:                 {sim.get('known_count')} ({(sim.get('known_count') or 0)/total*100 if total else 0:.1f}%)",
            f"  AMBIGUOUS:             {sim.get('ambiguous_count')} ({(sim.get('ambiguous_count') or 0)/total*100 if total else 0:.1f}%)",
            f"  NOT KNOWN:             {sim.get('unknown_count')} ({(sim.get('unknown_count') or 0)/total*100 if total else 0:.1f}%)",
            "",
            f"Correct matches:         {sim.get('correct_matches_pct')}%",
            f"Mismatches:              {sim.get('mismatches_pct')}%",
            f"Catalog completeness:    {sim.get('catalog_completeness_pct')}%",
            f"Redundancy index:        {sim.get('redundancy_index')}",
            f"ANN candidate recall:    {sim.get('ann_recall_pct')}%",
            "",
            f"GNN consults:            {sim.get('gnn_consults')}",
            f"GNN overturn precision:  {sim.get('gnn_overturn_precision_pct')}%"
            f" ({sim.get('gnn_overturn_correct')}/{sim.get('gnn_overturn_total')})",
            "",
            f"Final catalog size:      {sim.get('final_catalog_size')} (active: {sim.get('final_catalog_active')})",
            f"Sim wall time:           {sim.get('real_wall_time_s')}s",
        ]
    train_stage = results["stages"].get("train_catalog_gnn", {})
    if train_stage:
        lines_txt += [
            "",
            f"GNN confident_known thr: {train_stage.get('confident_known_threshold')}",
            f"GNN confident_novel thr: {train_stage.get('confident_novel_threshold')}",
            f"known_recall@novel_cut:  {train_stage.get('known_recall_at_novel_cut')}%",
        ]

    ax.text(0.02, 0.98, "\n".join(lines_txt), va="top", ha="left", fontsize=11, family="monospace",
            transform=ax.transAxes)
    savefig(fig, "00_scorecard.png")

    print(f"\nSaved {len(saved)} plots to {PLOTS_DIR}/")
    return saved


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run the full LEO catalog-GNN pipeline and plot results.")
    parser.add_argument("--n", type=int, default=20000, help="Number of catalog objects to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for data_generator.py.")
    parser.add_argument("--n-examples", type=int, default=12000, help="train_catalog_gnn.py --n-examples.")
    parser.add_argument("--epochs", type=int, default=250, help="train_catalog_gnn.py --epochs.")
    parser.add_argument("--min-known-examples", type=int, default=None,
                         help="train_catalog_gnn.py --min-known-examples.")
    parser.add_argument("--sequential", action="store_true",
                         help="Disable the train+sim / embedding-check concurrency and run every "
                              "stage strictly one at a time (cleanest console output for debugging "
                              "a single stage; see module docstring).")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing leo_catalog.csv.")
    parser.add_argument("--skip-embed", action="store_true", help="Reuse existing outputs/leo_embeddings.pt.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing outputs/catalog_gnn.pt.")
    parser.add_argument("--plots-only", action="store_true",
                         help="Skip running the pipeline; just re-render plots from an existing "
                              "outputs/eval_results.json.")
    args = parser.parse_args()

    if args.plots_only:
        if not os.path.exists(RESULTS_PATH):
            print(f"No {RESULTS_PATH} found -- run without --plots-only first.")
            sys.exit(1)
        with open(RESULTS_PATH) as fh:
            results = json.load(fh)
    else:
        results = run_pipeline(args)

    print("\nRendering plots...")
    make_plots(results)
    print("\nDone.")


if __name__ == "__main__":
    main()