import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import cProfile
import pstats
import sys
import os
import logging
from typing import Dict

# Añadir el directorio raíz del proyecto al PATH de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ahora importar los módulos locales
from src.utils import load_and_prepare_data, calculate_portfolio_metrics
from src.optimization import get_denoised_covariance, optimize_portfolio
from src.visuals import (
    plot_eigenvalue_distribution, plot_pca_comparison, plot_crisis_calm_comparison,
    plot_dendrogram, plot_mst, plot_efficient_frontier, plot_monte_carlo_simulation,
    plot_risk_contributions, plot_rmt_sensitivity
)
from src.generate_report import generate_pdf_report

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Portfolio Analysis and Optimization Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_stress_test(returns: pd.DataFrame, start_date: str, end_date: str, shock_magnitude: float, weights: Dict[str, pd.Series]) -> None:
    """Run stress test simulation and display results."""
    try:
        returns_stress = returns.loc[start_date:end_date].copy()
        if returns_stress.empty:
            st.error("No data available for the specified stress period.")
            return
    except KeyError:
        st.error("Stress period dates not in data index.")
        return
    
    shock_returns = returns_stress * (1 + shock_magnitude)
    
    st.subheader("Stress Test Results")
    metrics_summary = pd.DataFrame()
    for strategy, w in weights.items():
        portfolio_returns = (shock_returns * w).sum(axis=1)
        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(portfolio_returns)
        metrics_summary[strategy] = [f"{ar:.2%}", f"{av:.2%}", f"{sr:.2f}", f"{mdd:.2%}", f"{cvar:.2%}"]
    metrics_summary.index = ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "CVaR (95%)"]
    st.table(metrics_summary)
    
    cumulative_returns_df = pd.DataFrame({
        strategy: (1 + (shock_returns * w).sum(axis=1)).cumprod()
        for strategy, w in weights.items()
    })
    fig = px.line(cumulative_returns_df, title="Cumulative Returns Under Stress")
    st.plotly_chart(fig, use_container_width=True)
    
    csv = metrics_summary.to_csv()
    st.download_button(
        label="Download Stress Test Metrics",
        data=csv,
        file_name="stress_test_metrics.csv",
        mime="text/csv"
    )

@st.cache_data
def run_backtest(returns: pd.DataFrame, window_size: int, rebalance_freq: int, transaction_cost: float, cov_method: str, benchmark_returns: pd.Series = None) -> pd.DataFrame:
    """Run sliding window backtest with corrected matrix operations."""
    strategies = ['MinVol', 'MeanVar', 'RiskParity', 'HRP', 'CVaR', '1/N']
    if benchmark_returns is not None:
        strategies.append('S&P 500')
    
    n_assets = len(returns.columns)
    weights_eq = np.full(n_assets, 1.0 / n_assets)
    
    # Initialize results storage
    all_portfolio_returns = []
    dates_list = []
    
    # Initialize last weights
    last_weights = {s: weights_eq.copy() for s in strategies if s != 'S&P 500'}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = (len(returns) - window_size) // rebalance_freq
    step_count = 0
    
    for i in range(window_size, len(returns), rebalance_freq):
        opt_start = i - window_size
        opt_end = i
        test_end = min(i + rebalance_freq, len(returns))
        
        # Optimization window
        opt_returns = returns.iloc[opt_start:opt_end]
        
        # Test window
        test_returns = returns.iloc[i:test_end]
        test_dates = test_returns.index
        
        try:
            # Calculate covariance matrix
            cov_matrix = get_denoised_covariance(opt_returns, method=cov_method)
            if cov_matrix.empty:
                logger.warning(f"Empty covariance matrix at step {i}. Skipping.")
                continue
            
            # Optimize portfolios
            new_weights_dict = optimize_portfolio(cov_matrix, opt_returns)
            new_weights_dict['1/N'] = pd.Series(weights_eq, index=returns.columns)
            
            # Calculate returns for each strategy
            period_results = {}
            
            for strategy in strategies:
                if strategy == 'S&P 500':
                    if benchmark_returns is not None:
                        period_results[strategy] = benchmark_returns.loc[test_dates]
                else:
                    w_current = last_weights[strategy]
                    w_new = new_weights_dict[strategy].values
                    
                    # Calculate transaction costs
                    rotation = np.sum(np.abs(w_new - w_current))
                    cost_factor = (1 - transaction_cost * rotation)
                    
                    # Calculate portfolio returns
                    portfolio_returns_period = (test_returns.values @ w_current) * cost_factor
                    period_results[strategy] = pd.Series(portfolio_returns_period, index=test_dates)
                    
                    # Update weights for next period
                    last_weights[strategy] = w_new
            
            # Store results
            all_portfolio_returns.append(pd.DataFrame(period_results))
            
        except Exception as e:
            logger.warning(f"Error in backtest step {i}: {str(e)}")
            continue
        
        # Update progress
        step_count += 1
        progress = step_count / total_steps
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Backtest Progress: {progress*100:.1f}%")
    
    progress_bar.progress(1.0)
    status_text.text("Backtest completed!")
    
    # Combine all results
    if all_portfolio_returns:
        final_returns = pd.concat(all_portfolio_returns, ignore_index=False)
        return final_returns
    else:
        st.error("No valid backtest results generated.")
        return pd.DataFrame()

def profile_backtest(returns: pd.DataFrame, window_size: int, rebalance_freq: int, transaction_cost: float, cov_method: str, benchmark_returns: pd.Series = None) -> str:
    """Profile the backtest function and return performance stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    run_backtest(returns, window_size, rebalance_freq, transaction_cost, cov_method, benchmark_returns)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        stats.print_stats()
    return f.getvalue()

def main():
    """Main Streamlit app for portfolio analysis and optimization."""
    st.title("Advanced Portfolio Optimization")
    st.markdown("This tool leverages Random Matrix Theory (RMT), Hierarchical Risk Parity (HRP), and Monte Carlo simulations for robust portfolio management, inspired by physical system modeling.")
    
    st.sidebar.header("Data Upload")
    data_file = st.sidebar.file_uploader("Upload 'data.csv'", type=["csv"])
    
    if data_file is None:
        st.info("Please upload a CSV file to start the analysis.")
        return
    
    full_returns = load_and_prepare_data(data_file)
    if full_returns is None:
        return
    
    st.sidebar.success("Data loaded and prepared.")
    
    # SEPARAR SPY COMO BENCHMARK
    benchmark_returns = None
    portfolio_returns = full_returns.copy()
    
    # Extraer SPY si existe
    if 'SPY' in full_returns.columns:
        benchmark_returns = full_returns['SPY'].copy()
        portfolio_returns = full_returns.drop('SPY', axis=1)  # Excluir SPY del portfolio
        
        # Verificar que SPY tiene datos válidos
        valid_benchmark_data = benchmark_returns.dropna()
        if len(valid_benchmark_data) == 0:
            benchmark_returns = None
    
    st.sidebar.header("Asset Selection")
    available_assets = portfolio_returns.columns.tolist()
    selected_assets = st.sidebar.multiselect(
        "Select assets for analysis",
        options=available_assets,
        default=available_assets
    )
    
    if len(selected_assets) < 2:
        st.warning("Please select at least two assets.")
        return
    
    returns = portfolio_returns[selected_assets]
    
    # Display info about benchmark
    benchmark_status = "included" if benchmark_returns is not None else "not available"
    st.success(f"Analyzing {len(selected_assets)} assets (SPY excluded from portfolio). Benchmark (S&P 500) {benchmark_status}.")
    
    if benchmark_returns is not None:
        valid_benchmark = benchmark_returns.dropna()
        st.info(f"S&P 500 benchmark: {len(valid_benchmark)} valid observations from {valid_benchmark.index.min().date()} to {valid_benchmark.index.max().date()}")
    
    cov_method = st.sidebar.selectbox("Covariance Method", ('RMT', 'LedoitWolf'), key='global_cov_method')
    cov_matrix = get_denoised_covariance(returns, method=cov_method)
    if cov_matrix.empty:
        st.error("Failed to compute covariance matrix.")
        return
    
    weights = optimize_portfolio(cov_matrix, returns)
    weights['1/N'] = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "RMT Analysis", "RMT vs PCA", "Periods Analysis", "HRP Analysis",
        "Stress Testing", "Backtesting", "Monte Carlo", "Risk Contributions", "RMT Sensitivity"
    ])
    
    with tab1:
        st.header("RMT Correlation Matrix Analysis")
        st.markdown("Analyzes the correlation matrix using Random Matrix Theory to separate signal from noise.")
        plot_eigenvalue_distribution(returns)
        if not cov_matrix.empty:
            corr_denoised = cov_matrix.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_denoised, ax=ax, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title("Denoised Correlation Matrix")
            st.pyplot(fig)
            plot_mst(cov_matrix)
    
    with tab2:
        st.header("RMT vs Traditional PCA Comparison")
        st.markdown("Compares variance explained by PCA components to RMT analysis.")
        plot_pca_comparison(returns)
    
    with tab3:
        st.header("Risk Structure in Calm vs Crisis Periods")
        st.markdown("Compares the dominant eigenvector in calm and crisis periods.")
        col1, col2 = st.columns(2)
        with col1:
            calm_start = st.text_input("Calm Start (YYYY-MM-DD)", "2017-01-01")
            calm_end = st.text_input("Calm End (YYYY-MM-DD)", "2018-12-31")
        with col2:
            crisis_start = st.text_input("Crisis Start (YYYY-MM-DD)", "2020-02-01")
            crisis_end = st.text_input("Crisis End (YYYY-MM-DD)", "2020-04-30")
        if st.button("Compare Periods"):
            plot_crisis_calm_comparison(returns, calm_start, calm_end, crisis_start, crisis_end)
    
    with tab4:
        st.header("Hierarchical Risk Parity (HRP) Analysis")
        st.markdown("Visualizes asset clustering for HRP, similar to network analysis in physics.")
        plot_dendrogram(returns)
    
    with tab5:
        st.header("Stress Testing")
        st.markdown("Simulates a market shock to evaluate portfolio resilience.")
        stress_start = st.text_input("Stress Start (YYYY-MM-DD)", str(returns.index[-252].date()))
        stress_end = st.text_input("Stress End (YYYY-MM-DD)", str(returns.index[-1].date()))
        shock_magnitude = st.slider("Shock Magnitude (% drop)", -100, 0, -20) / 100
        if st.button("Run Stress Test"):
            run_stress_test(returns, stress_start, stress_end, shock_magnitude, weights)
    
    with tab6:
        st.header("Sliding Window Backtesting")
        st.markdown("Evaluates portfolio strategies over time with transaction costs, compared to S&P 500.")
        st.sidebar.header("Backtest Parameters")
        window_size = st.sidebar.slider("Optimization Window (days)", 252, 1000, 252)
        rebalance_freq = st.sidebar.slider("Rebalance Frequency (days)", 30, 252, 60)
        transaction_cost = st.sidebar.slider("Transaction Costs (% per rebalance)", 0.0, 1.0, 0.1, 0.01) / 100
        if st.button("Run Backtest"):
            if len(returns) < window_size + rebalance_freq:
                st.error("Data period too short for backtest parameters.")
            else:
                final_returns = run_backtest(returns, window_size, rebalance_freq, transaction_cost, cov_method, benchmark_returns)
                
                if not final_returns.empty:
                    st.subheader("Backtest Metrics")
                    metrics_summary = pd.DataFrame()
                    for strategy in final_returns.columns:
                        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(final_returns[strategy])
                        metrics_summary[strategy] = [f"{ar:.2%}", f"{av:.2%}", f"{sr:.2f}", f"{mdd:.2%}", f"{cvar:.2%}"]
                    metrics_summary.index = ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "CVaR (95%)"]
                    st.table(metrics_summary)
                    
                    csv = metrics_summary.to_csv()
                    st.download_button(
                        label="Download Backtest Metrics",
                        data=csv,
                        file_name="backtest_metrics.csv",
                        mime="text/csv"
                    )
                    
                    cumulative_returns_df = (1 + final_returns).cumprod()
                    fig = px.line(cumulative_returns_df, title="Cumulative Returns vs S&P 500 Benchmark")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    plot_efficient_frontier(returns.iloc[-window_size:], cov_matrix)
                    
                    st.subheader("Performance Profiling")
                    profile_stats = profile_backtest(returns, window_size, rebalance_freq, transaction_cost, cov_method, benchmark_returns)
                    st.text_area("Profiling Results", profile_stats, height=300)
    
    with tab7:
        st.header("Monte Carlo Simulation")
        st.markdown("Simulates future portfolio returns based on historical data, akin to stochastic processes in physics.")
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, key="monte_carlo_sims")
        horizon = st.slider("Simulation Horizon (days)", 30, 1000, 252, key="monte_carlo_horizon")
        if st.button("Run Monte Carlo Simulation"):
            plot_monte_carlo_simulation(returns, cov_matrix, weights, num_simulations, horizon)
    
    with tab8:
        st.header("Risk Contributions")
        st.markdown("Analyzes how each asset contributes to portfolio risk, similar to energy decomposition in physical systems.")
        plot_risk_contributions(weights, cov_matrix)
    
    with tab9:
        st.header("RMT Sensitivity Analysis")
        st.markdown("Analyzes eigenvalue distributions for varying T/N ratios.")
        t_n_ratios = st.multiselect("Select T/N Ratios", [0.5, 1.0, 2.0, 5.0, 10.0], default=[1.0, 2.0, 5.0])
        if st.button("Run Sensitivity Analysis"):
            plot_rmt_sensitivity(returns, t_n_ratios)
    
    # Report Generation
    st.sidebar.header("Generate Report")
    if st.sidebar.button("Generate PDF Report"):
        generate_pdf_report(returns, cov_matrix, weights)
        with open("portfolio_report.pdf", "rb") as file:
            st.sidebar.download_button(
                label="Download PDF Report",
                data=file,
                file_name="portfolio_report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()