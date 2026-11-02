"""Contract checks for the truth-blind online catalog lifecycle."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "rust_online"))
from online_catalog_benchmark import OnlineEngine, score_online


class OnlineCatalogTests(unittest.TestCase):
    def test_safe_default_defers_birth(self):
        catalog = np.tile([7000., 0, 0, 0, 7.5, 0], (5, 1))
        engine = OnlineEngine(catalog, np.ones(6), "tabular", object(), 1., float("inf"))
        self.assertFalse(engine.births)

    def test_birth_requires_distinct_times_and_confirmation_count(self):
        catalog = np.tile([7000., 0, 0, 0, 7.5, 0], (5, 1))
        engine = OnlineEngine(catalog, np.ones(6), "tabular", object(), 1., float("inf"),
                              min_birth_observations=3)
        state = np.array([7100., 0, 0, 0, 7.4, 0])
        engine._consider_birth(0, state, 0.)
        engine._consider_birth(1, state, 30.)
        self.assertEqual(len(engine.catalog), 5)
        engine._consider_birth(2, state, 60.)
        self.assertEqual(len(engine.catalog), 6)
        self.assertEqual([engine.decisions[i] for i in range(3)], [5, 5, 5])

    def test_outcomes_partition_known_and_novel(self):
        truth = np.array([10, 10, 20, 30])
        candidates = {i: np.array([0, 0, 0, 0, 0]) for i in range(4)}
        outcomes = score_online({0: 0, 1: 2, 2: 0, 3: -1}, candidates, truth,
                                np.array([10]), {2: [1]}, 0)
        self.assertEqual(outcomes["known_correct"], 1)
        self.assertEqual(outcomes["known_wrong_or_fragmented"], 1)
        self.assertEqual(outcomes["novel_to_seeded_false_merge"], 1)
        self.assertEqual(outcomes["novel_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
