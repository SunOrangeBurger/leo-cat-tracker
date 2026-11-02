"""Exact DP JPDA must reproduce explicit small-cluster enumeration."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jpda_baseline import enumerate_joint_events, jpda_marginals_exact
from rust_noise_orbit_benchmark import jpda_map_fallback


def enumerate_reference(match, miss=.08, clutter=.02):
    n_tracks, n_dets = match.shape
    feasible = {(t, d): True for t in range(n_tracks) for d in range(n_dets) if match[t, d] > 0}
    beta = np.zeros_like(match)
    total = 0.
    for event in enumerate_joint_events(list(range(n_tracks)), list(range(n_dets)), feasible):
        used = sum(d is not None for d in event.values())
        weight = clutter ** (n_dets - used)
        for track, det in event.items():
            weight *= miss if det is None else match[track, det]
        total += weight
        for track, det in event.items():
            if det is not None: beta[track, det] += weight
    return beta / total


class JPDADynamicProgrammingTests(unittest.TestCase):
    def test_matches_exhaustive_enumeration(self):
        rng = np.random.default_rng(12)
        for n_tracks, n_dets in [(1, 1), (2, 3), (3, 2), (4, 4)]:
            for _ in range(5):
                match = rng.uniform(0, .9, (n_tracks, n_dets))
                match[rng.random(match.shape) < .35] = 0.
                np.testing.assert_allclose(jpda_marginals_exact(match),
                                           enumerate_reference(match), rtol=1e-12, atol=1e-12)

    def test_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            jpda_marginals_exact(np.array([[-1.]]))
        with self.assertRaises(ValueError):
            jpda_marginals_exact(np.ones((1, 21)))

    def test_large_cluster_fallback_is_one_to_one(self):
        d2 = np.array([[1., 9., 100., 100., 100.],
                       [2., 3., 100., 100., 100.]])
        positions = np.tile(np.arange(5), (2, 1))
        assignments = jpda_map_fallback(d2, positions, [0, 1], [0, 1])
        self.assertEqual(set(assignments), {(0, 0), (1, 1)})


if __name__ == "__main__":
    unittest.main()
