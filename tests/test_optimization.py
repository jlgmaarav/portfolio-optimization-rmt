import unittest
import pandas as pd
import numpy as np
from src.optimization import get_denoised_covariance, get_hierarchical_weights, optimize_cvar, optimize_portfolio

class TestOptimization(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        self.returns = pd.DataFrame({
            'A': np.random.normal(0, 0.01, len(dates)),
            'B': np.random.normal(0, 0.01, len(dates)),
            'C': np.random.normal(0, 0.01, len(dates))
        }, index=dates)
        self.cov_matrix = self.returns.cov() * 252

    def test_get_denoised_covariance(self):
        """Test denoised covariance matrix."""
        cov = get_denoised_covariance(self.returns, method='RMT')
        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvals(cov) >= -1e-10))  # Allow small numerical errors
        self.assertTrue(cov.index.equals(self.returns.columns))
        # Test RMT eigenvalue bounds (Marchenko-Pastur)
        T, N = self.returns.shape
        Q = T / N
        lambda_plus = (1 + np.sqrt(1/Q))**2
        eigenvalues = np.linalg.eigvalsh(cov.corr())
        self.assertTrue(np.all(eigenvalues <= lambda_plus + 1e-6))

    def test_get_denoised_covariance_ledoit_wolf(self):
        """Test Ledoit-Wolf covariance."""
        cov = get_denoised_covariance(self.returns, method='LedoitWolf')
        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvals(cov) >= -1e-10))

    def test_get_denoised_covariance_empty(self):
        """Test empty returns."""
        cov = get_denoised_covariance(pd.DataFrame(), method='RMT')
        self.assertTrue(cov.empty)

    def test_get_hierarchical_weights(self):
        """Test HRP weights."""
        weights = get_hierarchical_weights(self.returns, self.cov_matrix)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue(np.all(weights >= 0))
        self.assertEqual(len(weights), 3)

    def test_optimize_cvar(self):
        """Test CVaR optimization."""
        weights = optimize_cvar(self.returns)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue(np.all(weights >= 0))
        self.assertEqual(len(weights), 3)

    def test_optimize_portfolio(self):
        """Test portfolio optimization."""
        weights_dict = optimize_portfolio(self.cov_matrix, self.returns)
        for strategy, weights in weights_dict.items():
            self.assertAlmostEqual(weights.sum(), 1.0, places=6)
            self.assertTrue(np.all(weights >= 0))
            self.assertEqual(len(weights), 3)

if __name__ == '__main__':
    unittest.main()