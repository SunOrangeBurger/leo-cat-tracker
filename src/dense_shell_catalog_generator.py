#!/usr/bin/env python3
"""
Generate a "dense shell" LEO catalog: a variant of data_generator.py's
distribution that deliberately packs a large fraction of objects into a
narrow band of semi-major axis + inclination, mimicking a real
megaconstellation shell (e.g. Starlink packs hundreds of satellites into
a band a few km wide in altitude and a fraction of a degree in
inclination, by design, for station-keeping and collision-avoidance
reasons).

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
Every prior benchmark run (20k and 60k, both against data_generator.py's
default distribution) showed max JPDA cluster size = 2 and GNN ambiguous-
consult rate under 0.02%, regardless of catalog size. That is not a
density problem solvable by adding more objects under the *same*
distribution -- data_generator.py spreads inclination across 5 disjoint
clusters and semi-major axis uniformly across a 1840km-wide altitude
band, so the local neighbor density near any given object stays roughly
constant as N grows; you get more clusters, not denser ones.

To actually test the claim this project is about (README.md Section 1:
JPDA's combinatorial blowup under crowding vs. a GNN-based
second-opinion), the catalog has to contain a real cluster of objects
that are simultaneously plausible matches for a single detection under
the SAME measurement noise (MEASUREMENT_NOISE in both
ground_station_tracking_simulation.py and jpda_baseline.py) both systems
are scored under. This script does that by construction, not by hoping
it emerges from more random draws.

--------------------------------------------------------------------------
WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------
Produces a catalog that is the ORIGINAL data_generator.py distribution
for a background population, PLUS one or more dense "shells": clusters of
--shell-size objects each, with semi-major axis and inclination drawn
from a narrow band (width controlled by --shell-width-km /
--shell-width-deg) around a randomly chosen center. Every other property
(type, size, RCS, eccentricity, RAAN/argp/nu, ballistic coefficient) is
sampled exactly as data_generator.py would -- this script imports and
reuses those same sampling functions directly rather than
reimplementing them, so the only variable being changed is spatial
density, not anything else that could confound the comparison.

Objects are named with a `_shell{k}_` infix so you can identify which
ones came from a dense shell vs. the background population after the
fact (useful for isolating shell-only recall/mismatch stats later).

--------------------------------------------------------------------------
HOW TO CHOOSE SHELL WIDTH
--------------------------------------------------------------------------
JPDA's gate in jpda_baseline.py uses GATE_CHI2 = 9.21 against
measurement noise stddevs of a_km=0.02, i_deg=0.001 (see
MEASUREMENT_NOISE in both scripts). A gate radius of roughly
3*stddev is where objects start being simultaneously plausible matches.
Default shell widths here (--shell-width-km 2.0, --shell-width-deg 0.05)
are chosen to be much wider than measurement noise (so objects are
genuinely orbitally close, like a real constellation shell) but narrow
enough that many objects fall inside the SAME gate window -- i.e. dense
enough to force real multi-candidate ambiguity, not just "close in the
same broad inclination cluster" the way data_generator.py already
produces.
"""

import sys
import csv
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_generator import (
    sample_eccentricity, sample_angle, sample_type, sample_size_and_rcs,
    generate_row, write_csv, EARTH_RADIUS_KM, LEO_ALT_MIN, LEO_ALT_MAX,
)
import orbital_physics as op


def sample_shell_row(idx, rng, shell_id, center_a, center_i, width_km, width_deg):
    obj_type = sample_type(rng)
    size, rcs = sample_size_and_rcs(rng, obj_type)
    bc = op.ballistic_coefficient(size, obj_type)
    a = center_a + rng.uniform(-width_km / 2, width_km / 2)
    i = max(0.0, min(180.0, center_i + rng.uniform(-width_deg / 2, width_deg / 2)))
    return {
        "name": f"{obj_type}_shell{shell_id}_{idx}",
        "a": a,
        "e": sample_eccentricity(rng),
        "i": i,
        "Omega": sample_angle(rng),
        "omega": sample_angle(rng),
        "nu": sample_angle(rng),
        "type": obj_type,
        "size_meters": size,
        "rcs": rcs,
        "bc_m2_per_kg": bc,
    }


def generate_dense_catalog(n_background, n_shells, shell_size, shell_width_km,
                            shell_width_deg, seed=None):
    rng = random.Random(seed)

    rows = [generate_row(i, rng) for i in range(n_background)]

    shell_centers = []
    for s in range(n_shells):
        # Pick shell centers well within the valid altitude band so the
        # +/- width doesn't push objects outside LEO_ALT_MIN/MAX, and
        # spread multiple shells apart so they don't merge into one
        # mega-cluster (unless that's what you want -- use --n-shells 1
        # for a single maximally dense shell instead).
        center_alt = rng.uniform(LEO_ALT_MIN + shell_width_km, LEO_ALT_MAX - shell_width_km)
        center_a = EARTH_RADIUS_KM + center_alt
        center_i = rng.uniform(30.0, 100.0)  # avoid extreme edges near 0/180
        shell_centers.append((center_a, center_i))

    idx = n_background
    for s, (center_a, center_i) in enumerate(shell_centers):
        for _ in range(shell_size):
            rows.append(sample_shell_row(idx, rng, s, center_a, center_i,
                                          shell_width_km, shell_width_deg))
            idx += 1

    return rows, shell_centers


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dense-shell LEO catalog to force genuine multi-candidate "
                    "association ambiguity (background population is unchanged from "
                    "data_generator.py; only the shell objects are packed tight)."
    )
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--n-background", type=int, default=20000,
                         help="Objects drawn from the normal data_generator.py distribution.")
    parser.add_argument("--n-shells", type=int, default=1,
                         help="Number of separate dense shells to generate.")
    parser.add_argument("--shell-size", type=int, default=2000,
                         help="Objects per shell.")
    parser.add_argument("--shell-width-km", type=float, default=2.0,
                         help="Full width of the shell's semi-major-axis band (km).")
    parser.add_argument("--shell-width-deg", type=float, default=0.05,
                         help="Full width of the shell's inclination band (degrees).")
    parser.add_argument("-s", "--seed", type=int, default=None)
    args = parser.parse_args()

    rows, centers = generate_dense_catalog(
        args.n_background, args.n_shells, args.shell_size,
        args.shell_width_km, args.shell_width_deg, seed=args.seed,
    )
    write_csv(rows, args.output)

    total = len(rows)
    print(f"Wrote {total} rows to {args.output} "
          f"({args.n_background} background + {args.n_shells}x{args.shell_size} shell objects)")
    for s, (a, i) in enumerate(centers):
        print(f"  shell {s}: center a={a:.2f}km, i={i:.3f}deg, "
              f"width=+/-{args.shell_width_km/2:.2f}km / +/-{args.shell_width_deg/2:.3f}deg, "
              f"{args.shell_size} objects")
    print(f"\nExpected local density inside a shell: ~{args.shell_size} objects packed into "
          f"a {args.shell_width_km}km x {args.shell_width_deg}deg window -- compare this to "
          f"JPDA's gate window (a few measurement-noise stddevs) to sanity-check this is "
          f"actually dense enough before running the benchmark.")


if __name__ == "__main__":
    main()