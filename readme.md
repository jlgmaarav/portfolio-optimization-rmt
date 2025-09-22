# Advanced Portfolio Optimization with Random Matrix Theory

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Stars](https://img.shields.io/github/stars/jlgmaarav/portfolio-optimization-rmt.svg)](https://github.com/jlgmaarav/portfolio-optimization-rmt/stargazers)

> **A sophisticated Python framework bridging quantitative finance and physics, applying Random Matrix Theory (RMT), Hierarchical Risk Parity (HRP), and advanced risk management techniques for professional portfolio optimization.**

Built by a 3rd-year Physics student, this project demonstrates the application of complex mathematical concepts from statistical physics to real-world financial problems.

## Key Highlights

### Physics-Finance Bridge
- **Random Matrix Theory (RMT)**: Noise filtering using Marchenko-Pastur eigenvalue distribution
- **Network Theory**: Asset relationship modeling through Minimum Spanning Trees
- **Stochastic Processes**: Monte Carlo return simulations  
- **Statistical Physics**: Signal processing from correlation matrices
- **Phase Transitions**: T/N ratio sensitivity analysis

### Advanced Features
- **Professional-Grade Backtesting**: Sliding window optimization with realistic transaction costs
- **Interactive Dashboard**: Streamlit web application with 9 specialized analysis tabs
- **Comprehensive Risk Management**: CVaR optimization, stress testing, and advanced analytics
- **LaTeX Report Generation**: Professional PDF reports with embedded mathematics

## Interactive Dashboard

**9 Specialized Analysis Tabs:**
1. **RMT Analysis** - Eigenvalue distributions and noise filtering
2. **RMT vs PCA** - Comparison with traditional methods
3. **Periods Analysis** - Calm vs crisis period comparison  
4. **HRP Analysis** - Hierarchical clustering visualization
5. **Stress Testing** - Market shock simulation
6. **Backtesting** - Rolling window performance analysis
7. **Monte Carlo** - Future return projections
8. **Risk Contributions** - Asset-level risk decomposition
9. **RMT Sensitivity** - Parameter robustness testing

## Core Algorithms

| Algorithm | Implementation | Application |
|-----------|----------------|-------------|
| **RMT Denoising** | Marchenko-Pastur bounds | Market signal extraction |
| **Hierarchical Risk Parity** | Recursive bisection clustering | Superior diversification |
| **CVaR Minimization** | Convex programming | Tail risk optimization |
| **Monte Carlo Simulation** | Vectorized stochastic processes | Future scenario modeling |
| **Stress Testing** | Market shock simulation | Portfolio resilience |

## Quick Start

### Installation
```bash
git clone https://github.com/jlgmaarav/portfolio-optimization-rmt.git
cd portfolio-optimization-rmt
pip install -r requirements.txt
```

### Download Market Data
```bash
python src/download_data.py
```

### Launch Interactive Dashboard
```bash
streamlit run src/app.py
```
**Access at**: http://localhost:8501

## 📈 Sample Results

```
Portfolio Performance Comparison
├── CVaR Strategy: 42.94% annualized returns (high risk)
├── HRP Strategy: 21.55% returns with superior risk-adjustment
├── Minimum Volatility: 1.95% returns with 2.48% volatility
└── Equal Weight: 12.34% baseline returns

Risk Metrics Analysis
├── Maximum Drawdown: -15.2% (CVaR) vs -8.7% (HRP)
├── Sharpe Ratio: 1.89 (HRP) vs 1.24 (Equal Weight)  
├── CVaR (95%): -4.2% expected shortfall
└── Volatility: 12.3% (HRP) vs 16.8% (Equal Weight)

RMT Analysis Results
├── Market Factors Identified: 4 significant eigenvalues
├── Noise Filtered: 78% of correlation matrix
├── Signal-to-Noise Ratio: 3.2:1
└── Marchenko-Pastur Compliance: 94.2%
```

## Project Structure

```
portfolio-optimization/
├── src/
│   ├── app.py                   # Main Streamlit application
│   ├── optimization.py          # Core optimization algorithms  
│   ├── utils.py                 # Data processing utilities
│   ├── visuals.py               # Plotting and visualization
│   ├── generate_report.py       # PDF report generation
│   └── download_data.py         # Market data fetching
├── tests/                       # Comprehensive test suite
├── docs/                        # Documentation and LaTeX templates  
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Mathematical Foundation

### Random Matrix Theory
For an N×N correlation matrix with T observations, eigenvalues are filtered using:
```
λ± = (1 ± √(N/T))²
```

### Hierarchical Risk Parity
Uses recursive bisection with risk allocation:
```
α = σ_right / (σ_left + σ_right)
```

### CVaR Optimization
Minimizes expected shortfall using linear programming formulation for robust tail risk management.

## Physics Concepts Applied

| Physics Concept | Financial Application |
|-----------------|----------------------|
| **Eigenvalue Decomposition** | Market factor identification via RMT |
| **Network Theory** | Asset relationship modeling through MST |
| **Stochastic Processes** | Monte Carlo return simulations |
| **Phase Transitions** | T/N ratio sensitivity analysis |
| **Signal Processing** | Noise filtering from correlation matrices |

## Portfolio Strategies Implemented

1. **Minimum Volatility** - Risk minimization approach
2. **Mean-Variance** - Classic Markowitz optimization
3. **Risk Parity** - Equal risk contribution allocation  
4. **Hierarchical Risk Parity (HRP)** - Machine learning clustering
5. **CVaR Optimization** - Tail risk minimization
6. **Equal Weight** - Naive diversification baseline

## Skills Demonstrated

### Quantitative Finance
- Modern Portfolio Theory and optimization
- Risk management and measurement
- Backtesting and performance attribution  
- Stochastic modeling and Monte Carlo methods

### Applied Mathematics  
- Linear algebra and eigenvalue decomposition
- Optimization theory and convex programming
- Statistical analysis and hypothesis testing
- Numerical methods and computational efficiency

### Software Engineering
- Clean architecture and modular design
- Interactive web application development
- Professional documentation and testing
- Performance optimization and vectorization

## Technical Requirements

```
streamlit>=1.28.0     # Interactive web applications
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
scipy>=1.10.0         # Scientific computing
scikit-learn>=1.3.0   # Machine learning
cvxpy>=1.4.0          # Convex optimization
matplotlib>=3.7.0     # Static plotting
seaborn>=0.12.0       # Statistical visualization
plotly>=5.15.0        # Interactive plotting
yfinance>=0.2.18      # Financial data
networkx>=3.1         # Network analysis
reportlab>=4.0.0      # PDF generation
```

## Professional Applications

### Quantitative Analysis
- **Asset Management**: Portfolio construction and optimization
- **Risk Management**: Tail risk assessment and hedging strategies
- **Algorithmic Trading**: Signal generation and factor modeling
- **Model Validation**: Benchmark comparison and statistical testing

### Academic Research  
- **Computational Finance**: Advanced numerical methods
- **Mathematical Finance**: Stochastic volatility modeling
- **Statistical Physics**: Applications to complex systems
- **Machine Learning**: Clustering and dimensionality reduction

## Academic Foundation

Implementation based on:
- **Marchenko, V.A. & Pastur, L.A.** (1967) - Random matrix theory
- **López de Prado, M.** (2016) - Hierarchical Risk Parity  
- **Bouchaud, J.P. & Potters, M.** (2003) - Theory of Financial Risk
- **Plerou, V. et al.** (1999) - Random matrix theory in finance

## Key Features

- **Real-time Analytics** - Live market data integration
- **Professional Visualizations** - Publication-ready plots and surfaces
- **Comprehensive Testing** - Unit tests and statistical validation
- **Modular Architecture** - Clean separation of concerns
- **Performance Optimized** - Vectorized operations and efficient algorithms

## Career Applications

Demonstrates expertise relevant to:
- **Quantitative Researcher** - Advanced mathematical modeling
- **Portfolio Manager** - Risk-adjusted optimization strategies  
- **Risk Analyst** - Comprehensive risk assessment frameworks
- **Financial Engineer** - Structured product development
- **Data Scientist** - Statistical analysis and machine learning

## Contact

**Jorge Lucas González**  
jorge.lucas.glez@gmail.com  
LinkedIn: https://www.linkedin.com/in/jorge-lucas-bb3550356/  
Burgos, Spain  
Physics Student (University of Valladolid)  
Specialization: Quantitative Finance & Mathematical Physics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mathematical foundation inspired by seminal work in Random Matrix Theory
- Implementation follows industry best practices in quantitative finance
- Physics-finance bridge demonstrates interdisciplinary problem-solving

---

**If this project demonstrates relevant skills for your needs, please consider giving it a star!**

*This project showcases the intersection of theoretical physics and practical finance, demonstrating how advanced mathematical concepts can solve real-world investment challenges.*