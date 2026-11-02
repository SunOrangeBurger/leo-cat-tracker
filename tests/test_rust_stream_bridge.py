"""Contract checks for the Rust-to-Python observation boundary."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rust_stream_bridge import MU_KM3_S2, cartesian_to_elements, convert
from rust_noise_orbit_benchmark import propagate_circular


class RustStreamBridgeTests(unittest.TestCase):
    def test_circular_orbit_marks_periapsis_undefined(self):
        radius = 7000.
        speed = math.sqrt(MU_KM3_S2 / radius)
        a, e, inc, raan, argp, raan_ok, argp_ok = cartesian_to_elements(
            [radius, 0, 0], [0, speed, 0])
        self.assertAlmostEqual(a, radius, places=6)
        self.assertAlmostEqual(e, 0, places=9)
        self.assertFalse(argp_ok)
        self.assertEqual(argp, 0.)

    def test_circular_propagation_preserves_radius_and_speed(self):
        radius = 7000.
        speed = math.sqrt(MU_KM3_S2 / radius)
        initial = np.array([[radius, 0, 0, 0, speed, 0]], float)
        period = 2 * math.pi * math.sqrt(radius**3 / MU_KM3_S2)
        future = propagate_circular(initial, np.array([period]))
        np.testing.assert_allclose(future[0], initial[0], atol=1e-8)

    def test_public_stream_rejects_identity_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "observations.jsonl"
            target = Path(tmp) / "converted.csv"
            radius = 7000.
            speed = math.sqrt(MU_KM3_S2 / radius)
            record = {"sensor_id": 0, "timestamp_ms": 0,
                      "position": [radius, 0, 0], "velocity": [0, speed, 0], "snr": 10.}
            source.write_text(json.dumps(record) + "\n")
            self.assertEqual(convert(source, target)["observations"], 1)
            record["object_index"] = 7
            source.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "unexpected sensor record fields"):
                convert(source, target)


if __name__ == "__main__":
    unittest.main()
