import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage
from sklearn.covariance import LedoitWolf
import streamlit as st
import logging
from typing import Optional, Dict, List
import warnings

logger = logging.getLogger(__name__)

@st.cache_data
def get_denoised_covariance(returns: pd.DataFrame, method: str = 'RMT') -> pd.DataFrame:
    """Calculate a denoised covariance matrix using RMT or Ledoit-Wolf.

    Args:
        returns (pd.DataFrame): DataFrame of daily returns.
        method (str): Denoising method ('RMT' or 'LedoitWolf').

    Returns:
        pd.DataFrame: Denoised covariance matrix (annualized).
    """
    if returns.empty:
        logger.warning("Empty returns DataFrame provided.")
        return pd.DataFrame()
    
    if method == 'LedoitWolf':
        lw = LedoitWolf(assume_centered=True)
        lw.fit(returns)
        cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        return cov_matrix * 252
    
    # RMT method
    cov_matrix_emp = returns.cov() * 252
    corr_matrix_emp = returns.corr()
    
    T, N = returns.shape
    if N == 0 or T <= N:
        logger.warning("T/N ratio must be >1 for RMT. Returning empirical covariance.")
        return cov_matrix_emp

    Q = T / N
    eigenvalues, eigenvectors = eigh(corr_matrix_emp)
    
    lambda_plus = (1 + np.sqrt(1/Q))**2
    lambda_minus = (1 - np.sqrt(1/Q))**2
    
    signal_indices = eigenvalues > lambda_plus
    denoised_corr_matrix = np.zeros_like(corr_matrix_emp)
    
    if np.any(signal_indices):
        signal_eigenvalues = eigenvalues[signal_indices]
        signal_eigenvectors = eigenvectors[:, signal_indices]
        denoised_corr_matrix += signal_eigenvectors @ np.diag(signal_eigenvalues) @ signal_eigenvectors.T
    
    if np.any(~signal_indices):
        noise_eigenvalues_mean = eigenvalues[~signal_indices].mean()
        noise_eigenvectors = eigenvectors[:, ~signal_indices]
        denoised_corr_matrix += (noise_eigenvalues_mean * noise_eigenvectors @ noise_eigenvectors.T)
    
    denoised_cov_matrix = denoised_corr_matrix * np.outer(np.sqrt(np.diag(cov_matrix_emp)), np.sqrt(np.diag(cov_matrix_emp)))
    return pd.DataFrame(denoised_cov_matrix, index=cov_matrix_emp.index, columns=cov_matrix_emp.index)

def get_hierarchical_weights(returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate portfolio weights using Hierarchical Risk Parity (HRP).

    Args:
        returns (pd.DataFrame): DataFrame of daily returns.
        cov_matrix (pd.DataFrame): Covariance matrix.

    Returns:
        pd.Series: HRP portfolio weights.
    """
    def get_quasi_diag(link: np.ndarray) -> List[int]:
        link = link.astype(int)
        sort_ix = [link[-1, 0], link[-1, 1]]
        num_items = int(link[-1, 3])  # Explicitly cast to int
        max_iter = 1000
        iter_count = 0
        
        while any(i >= num_items for i in sort_ix) and iter_count < max_iter:
            i = [i for i, x in enumerate(sort_ix) if x >= num_items][0]
            idx_to_split = sort_ix.pop(i)
            new_indices = link[int(idx_to_split) - num_items, :2].astype(int).tolist()
            sort_ix.extend(new_indices)
            iter_count += 1
        
        if iter_count >= max_iter:
            logger.warning("Max iterations reached in get_quasi_diag.")
        return sort_ix

    def get_recursive_bisection_weights(cov: pd.DataFrame, sort_ix: List[int]) -> pd.Series:
        # Fix for FutureWarning: Initialize with correct dtype
        weights = pd.Series(1.0, index=cov.index, dtype=np.float64)
        c_items = [sort_ix]
        while c_items:
            c_items = [i[j:k] for i in c_items for j, k in [(0, len(i) // 2), (len(i) // 2, len(i))] if len(i) > 1]
            for i in range(0, len(c_items), 2):
                if i + 1 >= len(c_items):
                    break
                left_cluster = c_items[i]
                right_cluster = c_items[i+1]
                
                left_cov_sub = cov.iloc[left_cluster, left_cluster]
                right_cov_sub = cov.iloc[right_cluster, right_cluster]
                
                left_diag_inv = 1 / np.diag(left_cov_sub)
                right_diag_inv = 1 / np.diag(right_cov_sub)
                
                left_vol = np.sqrt(left_diag_inv.T @ left_cov_sub.values @ left_diag_inv)
                right_vol = np.sqrt(right_diag_inv.T @ right_cov_sub.values @ right_diag_inv)
                
                alpha = right_vol / (left_vol + right_vol) if (left_vol + right_vol) > 0 else 0.5
                
                # Use .iloc for positional indexing instead of .loc
                weights.iloc[left_cluster] = weights.iloc[left_cluster] * alpha
                weights.iloc[right_cluster] = weights.iloc[right_cluster] * (1 - alpha)
        
        return weights.reindex(cov.index)

    # Suppress the clustering warning since it's expected behavior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        corr = cov_matrix.corr()
        distance_matrix = np.sqrt(0.5 * (1 - corr))
        link = linkage(distance_matrix.values, 'single')  # Use .values to avoid warning
    
    sort_ix = get_quasi_diag(link)
    weights = get_recursive_bisection_weights(cov_matrix, sort_ix)
    return weights

def optimize_cvar(returns: pd.DataFrame, confidence_level: float = 0.95) -> pd.Series:
    """Optimize portfolio by minimizing Conditional Value at Risk (CVaR).

    Args:
        returns (pd.DataFrame): Daily returns of assets.
        confidence_level (float): Confidence level for VaR and CVaR.

    Returns:
        pd.Series: CVaR-optimized portfolio weights.
    """
    n_assets = len(returns.columns)
    weights = cp.Variable(n_assets)
    portfolio_returns = returns.values @ weights
    
    alpha = 1 - confidence_level
    num_scenarios = len(returns)
    z = cp.Variable(num_scenarios)
    v = cp.Variable()
    
    objective = cp.Minimize(v + (1/((1 - alpha) * num_scenarios)) * cp.sum(z))
    constraints = [
        weights >= 0,
        cp.sum(weights) == 1,
        -portfolio_returns - v <= z,
        z >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    try:
        # Suppress SCS warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            problem.solve(solver=cp.SCS, verbose=False)
        
        if problem.status != 'optimal':
            logger.warning(f"CVaR optimization failed: {problem.status}. Using equal weights.")
            return pd.Series(1.0 / n_assets, index=returns.columns)
        return pd.Series(weights.value, index=returns.columns)
    except Exception as e:
        logger.warning(f"Error in CVaR optimization: {str(e)}. Using equal weights.")
        return pd.Series(1.0 / n_assets, index=returns.columns)

def optimize_portfolio(cov_matrix: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, pd.Series]:
    """Optimize portfolio using multiple methods.

    Args:
        cov_matrix (pd.DataFrame): Covariance matrix.
        returns (pd.DataFrame): Daily returns.

    Returns:
        Dict[str, pd.Series]: Dictionary of weights for each strategy.
    """
    assets = cov_matrix.columns
    n_assets = len(assets)
    expected_returns = returns.mean() * 252
    weights_dict = {}

    # Minimum Volatility
    from scipy.optimize import minimize
    def portfolio_volatility(weights, cov):
        return np.sqrt(weights.T @ cov @ weights)
    
    constraints_min_vol = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds_min_vol = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array(n_assets * [1.0 / n_assets])
    
    result_min_vol = minimize(portfolio_volatility, initial_weights, args=(cov_matrix.values,),
                              method='SLSQP', bounds=bounds_min_vol, constraints=constraints_min_vol)
    weights_dict['MinVol'] = pd.Series(result_min_vol.x, index=assets)
    
    # Mean-Variance
    weights_mv = cp.Variable(n_assets)
    target_return = expected_returns.mean()
    portfolio_return = weights_mv.T @ expected_returns.values
    portfolio_risk = cp.quad_form(weights_mv, cov_matrix.values)
    
    constraints_mv = [
        cp.sum(weights_mv) == 1,
        weights_mv >= 0,
        portfolio_return >= target_return
    ]
    
    problem = cp.Problem(cp.Minimize(portfolio_risk), constraints_mv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            problem.solve(solver='SCS', verbose=False)
        weights_dict['MeanVar'] = pd.Series(weights_mv.value, index=assets)
    except Exception:
        logger.warning("Mean-Variance optimization failed. Using equal weights.")
        weights_dict['MeanVar'] = pd.Series(n_assets * [1.0 / n_assets], index=assets)

    # Risk Parity
    def sum_of_squared_risk_contributions(weights, cov):
        sigma = np.sqrt(weights.T @ cov @ weights)
        marginal_contribution = (cov @ weights)
        risk_contributions = np.multiply(weights, marginal_contribution) / sigma
        target_risk_contribution = np.sum(risk_contributions) / len(weights)
        return np.sum((risk_contributions - target_risk_contribution)**2)

    result_rp = minimize(sum_of_squared_risk_contributions, initial_weights, args=(cov_matrix.values,),
                         method='SLSQP', bounds=bounds_min_vol, constraints=constraints_min_vol)
    weights_dict['RiskParity'] = pd.Series(result_rp.x, index=assets)
    
    # HRP
    weights_dict['HRP'] = get_hierarchical_weights(returns, cov_matrix)
    
    # CVaR
    weights_dict['CVaR'] = optimize_cvar(returns)
    
    return weights_dict