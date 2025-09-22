import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Optional
import sys
import os

logger = logging.getLogger(__name__)

def load_and_prepare_data(file_path):
    """Carga y prepara los datos de activos financieros desde un archivo CSV."""
    try:
        if hasattr(file_path, 'read'):  # Si es un objeto file-like de Streamlit
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:  # Si es una ruta de archivo
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

    data.index = pd.to_datetime(data.index)
    data = data.dropna(axis=1)
    
    returns = data.pct_change().dropna()
    for asset in returns.columns:
        q1 = returns[asset].quantile(0.25)
        q3 = returns[asset].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        returns[asset] = np.clip(returns[asset], lower_bound, upper_bound)
    
    return returns

def calculate_portfolio_metrics(portfolio_returns):
    """Calcula métricas de rendimiento para una cartera (recibe una Series)."""
    if portfolio_returns.empty:
        return 0, 0, 0, 0, 0
    
    # Limpiar valores NaN e infinitos
    portfolio_returns = portfolio_returns.dropna()
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(portfolio_returns) == 0:
        return 0, 0, 0, 0, 0
    
    annualized_return = (1 + portfolio_returns).prod()**(252 / len(portfolio_returns)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    max_drawdown = ((cumulative_returns - peak) / peak).min()
    
    # Calcular CVaR (95%)
    losses = -portfolio_returns[portfolio_returns < 0]
    if losses.empty:
        cvar = 0
    else:
        var = np.percentile(losses, 95)
        cvar = np.mean(losses[losses > var])
    
    return annualized_return, annualized_volatility, sharpe_ratio, max_drawdown, cvar

@st.cache_data
def get_denoised_covariance(returns, method='RMT'):
    """
    Calcula la matriz de covarianza filtrada (denoised) usando RMT o Ledoit-Wolf.
    
    Args:
        returns (pd.DataFrame): DataFrame con los retornos diarios.
        method (str): 'RMT' o 'LedoitWolf'.
    
    Returns:
        pd.DataFrame: Matriz de covarianza filtrada.
    """
    if returns.empty:
        return pd.DataFrame()
        
    if method == 'LedoitWolf':
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=True)
        lw.fit(returns)
        cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
        return cov_matrix * 252
    
    # Método RMT (mejorado)
    cov_matrix_emp = returns.cov() * 252
    corr_matrix_emp = returns.corr()
    
    T, N = returns.shape
    if N == 0 or T <= N:
        logger.warning("El ratio T/N debe ser mayor que 1 para usar RMT. Usando matriz empírica.")
        return cov_matrix_emp

    Q = T / N
    
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix_emp)
    
    # Límites de la distribución de Marchenko-Pastur
    lambda_plus = (1 + np.sqrt(1/Q))**2
    lambda_minus = (1 - np.sqrt(1/Q))**2
    
    # Identificar autovalores de señal y ruido
    signal_indices = eigenvalues > lambda_plus
    noise_indices = eigenvalues <= lambda_plus
    
    # Reconstruir la matriz de correlación filtrada
    denoised_corr_matrix = np.zeros_like(corr_matrix_emp)
    
    # Contribución de la señal
    signal_eigenvalues = eigenvalues[signal_indices]
    signal_eigenvectors = eigenvectors[:, signal_indices]
    if len(signal_eigenvalues) > 0:
        denoised_corr_matrix += signal_eigenvectors @ np.diag(signal_eigenvalues) @ signal_eigenvectors.T
    
    # Contribución del ruido (autovalores reemplazados por su media)
    if np.any(noise_indices):
        noise_eigenvalues_mean = eigenvalues[noise_indices].mean()
        noise_eigenvectors = eigenvectors[:, noise_indices]
        denoised_corr_matrix += (noise_eigenvalues_mean * noise_eigenvectors @ noise_eigenvectors.T)
        
    denoised_cov_matrix = denoised_corr_matrix * np.outer(np.sqrt(np.diag(cov_matrix_emp)), np.sqrt(np.diag(cov_matrix_emp)))
    
    return pd.DataFrame(denoised_cov_matrix, index=cov_matrix_emp.index, columns=cov_matrix_emp.columns)

def get_hierarchical_weights(returns, cov_matrix):
    """
    Calcula los pesos de la cartera usando el método Hierarchical Risk Parity (HRP).
    
    Args:
        returns (pd.DataFrame): DataFrame con los retornos diarios.
        cov_matrix (pd.DataFrame): Matriz de covarianza filtrada.
        
    Returns:
        pd.Series: Pesos de la cartera HRP.
    """
    from scipy.cluster.hierarchy import linkage
    
    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = [link[-1, 0], link[-1, 1]]
        num_items = link[-1, 3]
        max_iter = 1000
        iter_count = 0
        
        while any(i >= num_items for i in sort_ix) and iter_count < max_iter:
            i = [i for i, x in enumerate(sort_ix) if x >= num_items][0]
            idx_to_split = sort_ix.pop(i)
            new_indices = link[idx_to_split - num_items, :2].tolist()
            sort_ix.extend(new_indices)
            iter_count += 1
        
        if iter_count >= max_iter:
            logger.warning("Max iterations reached in get_quasi_diag.")
        return sort_ix

    def get_recursive_bisection_weights(cov, sort_ix):
        weights = pd.Series(1, index=cov.index)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in [(0, len(i) // 2), (len(i) // 2, len(i))] if len(i) > 1]
            for i in range(0, len(c_items), 2):
                left_cluster = c_items[i]
                right_cluster = c_items[i+1]
                
                left_cov_sub = cov.loc[left_cluster, left_cluster]
                right_cov_sub = cov.loc[right_cluster, right_cluster]
                
                left_diag_inv = 1 / np.diag(left_cov_sub)
                right_diag_inv = 1 / np.diag(right_cov_sub)
                
                left_vol = np.sqrt(left_diag_inv.T @ left_cov_sub @ left_diag_inv)
                right_vol = np.sqrt(right_diag_inv.T @ right_cov_sub @ right_diag_inv)
                
                alpha = right_vol / (left_vol + right_vol)
                weights.loc[left_cluster] *= alpha
                weights.loc[right_cluster] *= (1 - alpha)
        return weights.reindex(cov.index)

    # 1. Clustering
    corr = cov_matrix.corr()
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    link = linkage(distance_matrix, 'single')
    
    # 2. Ordenación de activos
    sort_ix = get_quasi_diag(link)
    
    # 3. Asignación de pesos recursiva
    weights = get_recursive_bisection_weights(cov_matrix, returns.columns[sort_ix])
    
    return weights

def optimize_cvar(returns, confidence_level=0.95):
    """
    Optimiza la cartera minimizando el Conditional Value at Risk (CVaR).
    
    Args:
        returns (pd.DataFrame): Retornos de los activos.
        confidence_level (float): Nivel de confianza para el VaR y CVaR.
        
    Returns:
        pd.Series: Pesos de la cartera optimizada por CVaR.
    """
    import cvxpy as cp
    
    n_assets = len(returns.columns)
    weights = cp.Variable(n_assets)
    portfolio_returns = returns.values @ weights
    
    # Parámetros para la optimización
    alpha = 1 - confidence_level
    num_scenarios = len(returns)
    z = cp.Variable(num_scenarios)  # Variables auxiliares
    v = cp.Variable()             # VaR
    
    # Objetivo: minimizar el CVaR
    objective = cp.Minimize(v + (1/((1 - alpha) * num_scenarios)) * cp.sum(z))
    
    # Restricciones
    constraints = [
        weights >= 0,
        cp.sum(weights) == 1,
        -portfolio_returns - v <= z,
        z >= 0
    ]
    
    problem = cp.Problem(objective, constraints)
    try:
        # Usar el solver SCS, que es más común que ECOS
        problem.solve(solver=cp.SCS)
        if problem.status != 'optimal':
            logger.warning(f"CVaR optimization failed: {problem.status}. Using equal weights.")
            return pd.Series(1.0 / n_assets, index=returns.columns)
        return pd.Series(weights.value, index=returns.columns)
    except Exception as e:
        logger.warning(f"Error al optimizar por CVaR: {e}. Usando pesos iguales.")
        return pd.Series(1.0 / n_assets, index=returns.columns)