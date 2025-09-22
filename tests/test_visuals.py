import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.visuals import (
    plot_eigenvalue_distribution, plot_pca_comparison, plot_crisis_calm_comparison,
    plot_dendrogram, plot_mst, plot_efficient_frontier, plot_monte_carlo_simulation,
    plot_risk_contributions
)

class TestVisuals(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        self.returns = pd.DataFrame({
            'A': np.random.normal(0, 0.01, len(dates)),
            'B': np.random.normal(0, 0.01, len(dates))
        }, index=dates)
        self.cov_matrix = self.returns.cov() * 252
        self.weights = {'Test': pd.Series([0.5, 0.5], index=['A', 'B'])}

    @patch('streamlit.pyplot')
    @patch('streamlit.error')
    def test_plot_eigenvalue_distribution(self, mock_error, mock_pyplot):
        """Test eigenvalue distribution plot."""
        plot_eigenvalue_distribution(self.returns)
        mock_pyplot.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.error')
    def test_plot_pca_comparison(self, mock_error, mock_plotly):
        """Test PCA comparison plot."""
        plot_pca_comparison(self.returns)
        mock_plotly.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.pyplot')
    @patch('streamlit.error')
    def test_plot_crisis_calm_comparison(self, mock_error, mock_pyplot):
        """Test crisis vs calm comparison."""
        plot_crisis_calm_comparison(self.returns, '2020-01-01', '2020-06-30', '2020-07-01', '2020-12-31')
        mock_pyplot.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.pyplot')
    @patch('streamlit.error')
    def test_plot_dendrogram(self, mock_error, mock_pyplot):
        """Test dendrogram plot."""
        plot_dendrogram(self.returns)
        mock_pyplot.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.pyplot')
    @patch('streamlit.error')
    def test_plot_mst(self, mock_error, mock_pyplot):
        """Test MST plot."""
        plot_mst(self.cov_matrix)
        mock_pyplot.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.error')
    def test_plot_efficient_frontier(self, mock_error, mock_plotly):
        """Test efficient frontier plot."""
        plot_efficient_frontier(self.returns, self.cov_matrix)
        mock_plotly.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.error')
    def test_plot_monte_carlo_simulation(self, mock_error, mock_plotly):
        """Test Monte Carlo simulation plot."""
        plot_monte_carlo_simulation(self.returns, self.cov_matrix, self.weights)
        mock_plotly.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.plotly_chart')
    @patch('streamlit.error')
    def test_plot_risk_contributions(self, mock_error, mock_plotly):
        """Test risk contributions plot."""
        plot_risk_contributions(self.weights, self.cov_matrix)
        mock_plotly.assert_called()
        mock_error.assert_not_called()

    @patch('streamlit.error')
    def test_plot_with_empty_data(self, mock_error):
        """Test plots with empty inputs."""
        empty_returns = pd.DataFrame()
        empty_cov = pd.DataFrame()
        empty_weights = {}
        plot_eigenvalue_distribution(empty_returns)
        plot_pca_comparison(empty_returns)
        plot_crisis_calm_comparison(empty_returns, '2020-01-01', '2020-06-30', '2020-07-01', '2020-12-31')
        plot_dendrogram(empty_returns)
        plot_mst(empty_cov)
        plot_efficient_frontier(empty_returns, empty_cov)
        plot_monte_carlo_simulation(empty_returns, empty_cov, empty_weights)
        plot_risk_contributions(empty_weights, empty_cov)
        self.assertTrue(mock_error.called)

if __name__ == '__main__':
    unittest.main()