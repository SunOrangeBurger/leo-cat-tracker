import unittest
import numpy as np
from rust_noise_orbit_benchmark import jpda_probabilities


class GaussianLikelihoodTests(unittest.TestCase):
    def test_covariance_volume_enters_exact_marginals(self):
        scales=np.array([[1.,.25,.1,.01,.001]])
        probabilities,_=jpda_probabilities(np.zeros((1,6)),np.zeros((1,5,6)),
            np.arange(5)[None],{0:[0]},likelihood_scale=scales)
        weights=.92*scales
        expected=weights/(.08*.02+weights.sum())
        np.testing.assert_allclose(probabilities,expected,rtol=1e-12)

    def test_invalid_covariance_volume_is_rejected(self):
        with self.assertRaises(ValueError):
            jpda_probabilities(np.zeros((1,6)),np.zeros((1,5,6)),
                np.arange(5)[None],{0:[0]},likelihood_scale=np.zeros((1,5)))


if __name__=='__main__':unittest.main()
