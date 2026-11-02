#!/usr/bin/env python3
"""Validate and convert rust-leo-sim's public JSONL sensor stream.

This is a transport/feature bridge, not a reproduction of the reported
ANN-versus-JPDA benchmark. It deliberately does not read the adjacent truth
file. The simulator's near-circular orbits make argument of periapsis
undefined, so rows carry an explicit validity flag for that element.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MU_KM3_S2 = 398600.4418


def cartesian_to_elements(position, velocity):
    """Return osculating (a km, e, i, RAAN, argp) from an ECI state.

    Singular angles are set to zero and identified in the returned flags.
    This convention is deterministic but does not recover lost information.
    """
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if r.shape != (3,) or v.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("position and velocity must be finite three-vectors")
    rn = float(np.linalg.norm(r))
    if rn <= 0:
        raise ValueError("zero position vector")
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    if hn <= 0:
        raise ValueError("zero angular momentum")
    energy = float(np.dot(v, v)) / 2 - MU_KM3_S2 / rn
    if energy >= 0:
        raise ValueError("state is not a bound Earth orbit")
    a = -MU_KM3_S2 / (2 * energy)
    evec = np.cross(v, h) / MU_KM3_S2 - r / rn
    e = float(np.linalg.norm(evec))
    incl = math.degrees(math.acos(float(np.clip(h[2] / hn, -1, 1))))
    node = np.cross([0., 0., 1.], h)
    nn = float(np.linalg.norm(node))
    raan_defined = nn > 1e-8
    argp_defined = e >= 1e-3 and raan_defined
    raan = math.degrees(math.atan2(node[1], node[0])) % 360 if raan_defined else 0.
    if argp_defined:
        cos_w = float(np.clip(np.dot(node, evec) / (nn * e), -1, 1))
        argp = math.degrees(math.acos(cos_w))
        if evec[2] < 0:
            argp = 360 - argp
    else:
        argp = 0.
    return a, e, incl, raan, argp, raan_defined, argp_defined


def convert(public_jsonl: Path, output_csv: Path):
    columns = ["sensor_id", "timestamp_ms", "x_km", "y_km", "z_km",
               "vx_km_s", "vy_km_s", "vz_km_s", "snr", "a_km", "e",
               "i_deg", "raan_deg", "argp_deg", "raan_defined", "argp_defined"]
    counts = {"observations": 0, "undefined_raan": 0, "undefined_argp": 0}
    metadata_path = Path(str(public_jsonl) + ".meta.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata is not None and metadata.get("schema_version") != 1:
        raise ValueError("unsupported Rust observation schema version")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with public_jsonl.open() as source, output_csv.open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=columns)
        writer.writeheader()
        for lineno, line in enumerate(source, 1):
            record = json.loads(line)
            if set(record) != {"sensor_id", "timestamp_ms", "position", "velocity", "snr"}:
                raise ValueError(f"line {lineno}: unexpected sensor record fields")
            a, e, inc, raan, argp, raan_ok, argp_ok = cartesian_to_elements(
                record["position"], record["velocity"])
            row = dict(zip(columns[:9], [record["sensor_id"], record["timestamp_ms"],
                                         *record["position"], *record["velocity"], record["snr"]]))
            row.update(a_km=a, e=e, i_deg=inc, raan_deg=raan, argp_deg=argp,
                       raan_defined=int(raan_ok), argp_defined=int(argp_ok))
            writer.writerow(row)
            counts["observations"] += 1
            counts["undefined_raan"] += not raan_ok
            counts["undefined_argp"] += not argp_ok
    if metadata is not None and counts["observations"] != metadata.get("observations"):
        raise ValueError("manifest observation count differs from public stream")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_jsonl", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.public_jsonl, args.output_csv), indent=2))


if __name__ == "__main__":
    main()
