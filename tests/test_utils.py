import unittest
import pandas as pd
import numpy as np
import os
from src.utils import load_and_prepare_data, calculate_portfolio_metrics

class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        self.test_data = pd.DataFrame({
            'A': np.random.normal(0, 0.01, len(dates)),
            'B': np.random.normal(0, 0.01, len(dates))
        }, index=dates)
        self.test_data.to_csv('test_data.csv')

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

    def test_load_and_prepare_data(self):
        """Test loading and preparing data."""
        returns = load_and_prepare_data('test_data.csv')
        self.assertIsNotNone(returns)
        self.assertEqual(returns.shape[1], 2)
        self.assertTrue(np.all(returns.index == pd.to_datetime(returns.index)))
        self.assertFalse(returns.isna().any().any())
        # Test Winsorization (outlier clipping)
        self.assertTrue(np.all(returns.abs() <= returns.quantile(0.99)))

    def test_load_and_prepare_data_invalid_file(self):
        """Test loading invalid file."""
        returns = load_and_prepare_data('nonexistent.csv')
        self.assertIsNone(returns)

    def test_load_and_prepare_data_single_asset(self):
        """Test loading data with one asset."""
        single_asset_data = self.test_data[['A']]
        single_asset_data.to_csv('single_asset.csv')
        returns = load_and_prepare_data('single_asset.csv')
        self.assertIsNotNone(returns)
        self.assertEqual(returns.shape[1], 1)
        if os.path.exists('single_asset.csv'):
            os.remove('single_asset.csv')

    def test_calculate_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        portfolio_returns = pd.Series(np.random.normal(0, 0.01, 252))
        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(portfolio_returns)
        self.assertIsInstance(ar, float)
        self.assertIsInstance(av, float)
        self.assertIsInstance(sr, float)
        self.assertIsInstance(mdd, float)
        self.assertIsInstance(cvar, float)
        self.assertLessEqual(mdd, 0)
        self.assertLessEqual(cvar, 0)

    def test_calculate_portfolio_metrics_empty(self):
        """Test metrics with empty input."""
        portfolio_returns = pd.Series([])
        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(portfolio_returns)
        self.assertEqual((ar, av, sr, mdd, cvar), (0, 0, 0, 0, 0))

if __name__ == '__main__':
    unittest.main()