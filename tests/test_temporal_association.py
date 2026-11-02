import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'rust_online'))
from temporal_association import TrackHistory, HistoryGraph
from tracklet_initiation import TrackletInitiator
from kepler_dynamics import propagate_kepler
from audit_temporal_calibration import strict_threshold


class TemporalAssociationTests(unittest.TestCase):
    def test_calibration_does_not_impose_absolute_probability_floor(self):
        for dtype in [np.float32,np.float64]:
            probability=np.array([[0.,0.],[1e-12,0.]],dtype=dtype)
            threshold=strict_threshold(probability,np.array([99,1]),np.array([1]))
            self.assertGreater(threshold,0.)
            self.assertLess(threshold,1e-12)
            self.assertFalse(probability[0].max()>=threshold)
            self.assertTrue(probability[1].max()>=threshold)

    def test_elliptic_propagation_is_reversible_and_conserves_energy(self):
        initial = np.array([[7000.,0,0,0,7.,.5],[7200.,0,0,0,7.8,.1]])
        future = propagate_kepler(initial, np.array([1000.,3000.]))
        recovered = propagate_kepler(future, np.array([-1000.,-3000.]))
        np.testing.assert_allclose(recovered,initial,atol=1e-7)
        energy = lambda x: np.sum(x[:,3:]**2,axis=1)/2-398600.4418/np.linalg.norm(x[:,:3],axis=1)
        np.testing.assert_allclose(energy(future),energy(initial),atol=1e-10)

    def test_birth_requires_four_distinct_scans_and_predictive_consistency(self):
        initial = np.array([[7000.,0,0,0,7.,.5]])
        manager = TrackletInitiator(np.array([1.,1.,1.,.01,.01,.01]),propagate_kepler)
        self.assertEqual(manager.process_scan([0],initial,0.),[])
        for row,t in enumerate([30.,60.],1):
            self.assertEqual(manager.process_scan([row],propagate_kepler(initial,[t]),t),[])
        result = manager.process_scan([3],propagate_kepler(initial,[120.]),120.)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['rows'],[0,1,2,3])

    def test_candidate_permutation_equivariance(self):
        torch.manual_seed(4)
        model = HistoryGraph().eval()
        history = torch.randn(3, 5, 8, 8)
        local = torch.randn(3, 5, 20)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            a = model(history, local)
            b = model(history[:, permutation], local[:, permutation])
        torch.testing.assert_close(b, a[:, permutation], atol=1e-6, rtol=1e-5)

    def test_public_history_rejects_future_sighting(self):
        initial = np.array([[7000.,0,0,0,7.546,0]])
        history = TrackHistory(initial)
        history.add_scan(np.array([0]), initial, 60.)
        with self.assertRaisesRegex(ValueError, 'future history'):
            history.features(initial, 30., np.zeros((1,5), int), np.ones(6))

    def test_simultaneous_sensor_rows_produce_one_history_entry(self):
        initial = np.array([[7000.,0,0,0,7.546,0]])
        history = TrackHistory(initial)
        history.add_scan(np.array([0,0,-1]), np.repeat(initial,3,axis=0), 30.)
        self.assertEqual(len(history.rows[0]), 2)
        self.assertEqual(len(history.rows), 1)


if __name__ == '__main__':
    unittest.main()
