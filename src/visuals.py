import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
import streamlit as st
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@st.cache_data
def plot_eigenvalue_distribution(returns: pd.DataFrame) -> None:
    """Plot the eigenvalue distribution for Random Matrix Theory analysis.

    Args:
        returns (pd.DataFrame): Daily returns of assets.
    """
    if returns.empty or len(returns.columns) < 2:
        st.error("Insufficient data for eigenvalue analysis.")
        return
    
    corr_matrix = returns.corr()
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(eigenvalues, bins=50, kde=True, ax=ax)
    ax.set_title("Eigenvalue Distribution of Correlation Matrix")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    st.pyplot(fig)

@st.cache_data
def plot_pca_comparison(returns: pd.DataFrame) -> None:
    """Compare variance explained by RMT and PCA."""
    if returns.empty or len(returns.columns) < 2:
        st.error("Insufficient data for PCA comparison.")
        return
    
    pca = PCA()
    pca.fit(returns)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig = px.line(
        x=range(1, len(explained_variance) + 1),
        y=cumulative_variance,
        title="Cumulative Variance Explained by PCA Components",
        labels={"x": "Component", "y": "Cumulative Variance Explained"}
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_crisis_calm_comparison(returns: pd.DataFrame, calm_start: str, calm_end: str, crisis_start: str, crisis_end: str) -> None:
    """Compare eigenvectors in calm vs crisis periods."""
    try:
        calm_returns = returns.loc[calm_start:calm_end]
        crisis_returns = returns.loc[crisis_start:crisis_end]
    except KeyError:
        st.error("Invalid dates for calm or crisis periods.")
        return
    
    if calm_returns.empty or crisis_returns.empty:
        st.error("No data available for the specified periods.")
        return
    
    calm_corr = calm_returns.corr()
    crisis_corr = crisis_returns.corr()
    
    calm_eigvals, calm_eigvecs = np.linalg.eigh(calm_corr)
    crisis_eigvals, crisis_eigvecs = np.linalg.eigh(crisis_corr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(range(len(calm_eigvecs[:, -1])), calm_eigvecs[:, -1], tick_label=returns.columns)
    ax1.set_title(f"First Eigenvector (Calm: {calm_start} to {calm_end})")
    ax1.set_ylabel("Weight")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(range(len(crisis_eigvecs[:, -1])), crisis_eigvecs[:, -1], tick_label=returns.columns)
    ax2.set_title(f"First Eigenvector (Crisis: {crisis_start} to {crisis_end})")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def plot_dendrogram(returns: pd.DataFrame) -> None:
    """Plot hierarchical clustering dendrogram for HRP."""
    if returns.empty or len(returns.columns) < 2:
        st.error("Insufficient data for dendrogram.")
        return
    
    corr = returns.corr()
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    link = linkage(distance_matrix, 'single')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(link, labels=returns.columns, orientation='top', leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram for HRP")
    ax.set_xlabel("Assets")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def plot_mst(cov_matrix: pd.DataFrame) -> None:
    """Plot Minimum Spanning Tree of assets."""
    if cov_matrix.empty or len(cov_matrix.columns) < 2:
        st.error("Invalid covariance matrix for MST.")
        return
    
    corr = cov_matrix.corr()
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    G = nx.from_numpy_array(distance_matrix.values)
    labels = {i: cov_matrix.columns[i] for i in range(len(cov_matrix.columns))}
    G = nx.relabel_nodes(G, labels)
    mst = nx.minimum_spanning_tree(G)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(mst, seed=42)
    nx.draw(mst, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, ax=ax)
    ax.set_title("Minimum Spanning Tree of Assets")
    st.pyplot(fig)

@st.cache_data
def plot_efficient_frontier(returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> None:
    """Plot efficient frontier of portfolios."""
    if returns.empty or cov_matrix.empty:
        st.error("Invalid data for efficient frontier.")
        return
    
    expected_returns = returns.mean() * 252
    risks, rets = [], []
    for _ in range(1000):
        w = np.random.dirichlet(np.ones(len(returns.columns)))
        rets.append(w @ expected_returns)
        risks.append(np.sqrt(w @ cov_matrix @ w))
    
    fig = px.scatter(
        x=risks, y=rets, title="Efficient Frontier",
        labels={"x": "Annualized Volatility", "y": "Annualized Return"}
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_monte_carlo_simulation(returns: pd.DataFrame, cov_matrix: pd.DataFrame, weights: Dict[str, pd.Series], num_simulations: int = 1000, horizon: int = 252) -> None:
    """Simulate and plot Monte Carlo portfolio returns."""
    if returns.empty or cov_matrix.empty or not weights:
        st.error("Invalid inputs for Monte Carlo simulation.")
        return
    
    expected_returns = returns.mean()
    simulation_results = {}
    
    for strategy, w in weights.items():
        sim_returns = np.random.multivariate_normal(
            expected_returns.values,
            cov_matrix.values / 252,
            size=(num_simulations, horizon)
        )
        portfolio_sim = np.dot(sim_returns, w.values)
        cumulative_returns = np.cumprod(1 + portfolio_sim, axis=1)[:, -1] - 1
        simulation_results[strategy] = cumulative_returns * 252
    
    simulation_df = pd.DataFrame(simulation_results)
    fig = px.histogram(
        simulation_df, nbins=50,
        title="Monte Carlo Simulation of Annualized Portfolio Returns",
        labels={"value": "Annualized Return", "variable": "Strategy"}
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_risk_contributions(weights: Dict[str, pd.Series], cov_matrix: pd.DataFrame) -> None:
    """Plot risk contributions of assets for each strategy."""
    if cov_matrix.empty or not weights:
        st.error("Invalid inputs for risk contribution plot.")
        return
    
    risk_contribs = {}
    for strategy, w in weights.items():
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        marginal_contrib = cov_matrix @ w
        risk_contrib = (w * marginal_contrib) / portfolio_vol
        risk_contribs[strategy] = risk_contrib
    
    risk_df = pd.DataFrame(risk_contribs, index=cov_matrix.index)
    fig = px.bar(
        risk_df, barmode='group',
        title="Risk Contributions by Asset and Strategy",
        labels={"value": "Risk Contribution", "index": "Asset"}
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_rmt_sensitivity(returns: pd.DataFrame, t_n_ratios: list = [1.0, 2.0, 5.0]) -> None:
    """Plot eigenvalue distributions for varying T/N ratios.

    Args:
        returns (pd.DataFrame): Daily returns of assets.
        t_n_ratios (list): List of T/N ratios to simulate.

    Examples:
        >>> plot_rmt_sensitivity(returns, t_n_ratios=[1.0, 2.0, 5.0])
    """
    if returns.empty or len(returns.columns) < 2:
        st.error("Insufficient data for RMT sensitivity analysis.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for q in t_n_ratios:
        n = len(returns.columns)
        t = int(n * q)
        if t > len(returns):
            st.warning(f"T/N ratio {q} requires {t} time steps, but only {len(returns)} available. Skipping.")
            continue
        subset_returns = returns.iloc[-t:]
        corr_matrix = subset_returns.corr()
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        sns.histplot(eigenvalues, bins=50, kde=True, label=f"T/N={q}", stat='density', ax=ax)
    
    ax.set_title("RMT Eigenvalue Distribution for Varying T/N Ratios")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)