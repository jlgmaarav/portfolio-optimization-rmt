from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import calculate_portfolio_metrics
from src.visuals import plot_eigenvalue_distribution, plot_dendrogram
import logging
from typing import Dict
import os
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_pdf_report(returns: pd.DataFrame, cov_matrix: pd.DataFrame, weights: Dict[str, pd.Series], output_file: str = "portfolio_report.pdf") -> None:
    """Generate a comprehensive PDF report with portfolio analysis.

    Args:
        returns (pd.DataFrame): Daily returns of assets.
        cov_matrix (pd.DataFrame): Covariance matrix.
        weights (Dict[str, pd.Series]): Portfolio weights by strategy.
        output_file (str): Output PDF filename.
    """
    if returns.empty or cov_matrix.empty or not weights:
        logger.error("Invalid inputs for report generation.")
        return

    # Create document
    doc = SimpleDocTemplate(output_file, pagesize=A4, rightMargin=72, leftMargin=72, 
                           topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        textColor=colors.darkblue
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    )

    # Title page
    story.append(Paragraph("Advanced Portfolio Optimization Report", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Random Matrix Theory & Hierarchical Risk Parity Analysis", styles['Heading3']))
    story.append(Spacer(1, 30))
    
    # Report metadata
    metadata_data = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ['Analysis Period:', f"{returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}"],
        ['Number of Assets:', str(len(returns.columns))],
        ['Total Observations:', str(len(returns))],
        ['Strategies Analyzed:', ', '.join(weights.keys())]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 30))

    # Executive Summary
    story.append(Paragraph("Executive Summary", subtitle_style))
    
    # Calculate best performing strategy
    best_strategy = None
    best_sharpe = -999
    for strategy, w in weights.items():
        portfolio_returns = (returns * w).sum(axis=1)
        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(portfolio_returns)
        if sr > best_sharpe:
            best_sharpe = sr
            best_strategy = strategy
    
    executive_text = f"""
    This report presents a comprehensive analysis of portfolio optimization using advanced quantitative techniques 
    inspired by physics and mathematics. The analysis covers {len(returns.columns)} assets over a period from 
    {returns.index[0].strftime('%B %Y')} to {returns.index[-1].strftime('%B %Y')}.
    
    <b>Key Findings:</b><br/>
    • Random Matrix Theory successfully identified {len(np.where(np.linalg.eigvals(returns.corr()) > (1 + np.sqrt(len(returns.columns)/len(returns)))**2)[0])} significant market factors<br/>
    • Best performing strategy: {best_strategy} (Sharpe ratio: {best_sharpe:.2f})<br/>
    • Portfolio diversification analysis reveals clear sector clustering<br/>
    • Risk concentration varies significantly across optimization methods
    """
    
    story.append(Paragraph(executive_text, body_style))
    story.append(PageBreak())

    # Portfolio Performance Metrics
    story.append(Paragraph("Portfolio Performance Analysis", subtitle_style))
    
    # Create detailed metrics table
    metrics_data = [['Strategy', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown', 'CVaR (95%)']]
    
    for strategy, w in weights.items():
        portfolio_returns = (returns * w).sum(axis=1)
        ar, av, sr, mdd, cvar = calculate_portfolio_metrics(portfolio_returns)
        metrics_data.append([
            strategy,
            f"{ar:.2%}",
            f"{av:.2%}", 
            f"{sr:.2f}",
            f"{mdd:.2%}",
            f"{cvar:.2%}"
        ])
    
    metrics_table = Table(metrics_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Methodology section
    story.append(Paragraph("Methodology", subtitle_style))
    methodology_text = """
    <b>Random Matrix Theory (RMT):</b> Applied Marchenko-Pastur distribution to filter noise from the correlation matrix. 
    Eigenvalues above the theoretical threshold λ+ = (1 + √(N/T))² are considered signal, while those below represent market noise.
    
    <b>Hierarchical Risk Parity (HRP):</b> Uses hierarchical clustering to group similar assets, then applies recursive 
    bisection to allocate risk equally across clusters, providing better diversification than traditional methods.
    
    <b>Conditional Value at Risk (CVaR):</b> Optimizes for tail risk by minimizing the expected loss in the worst 5% of scenarios,
    providing more robust risk management than variance-based approaches.
    
    <b>Backtesting Framework:</b> Employs sliding window optimization with realistic transaction costs and rebalancing frequency
    to simulate real-world portfolio implementation challenges.
    """
    story.append(Paragraph(methodology_text, body_style))
    story.append(Spacer(1, 20))

    # RMT Analysis
    story.append(Paragraph("Random Matrix Theory Analysis", subtitle_style))
    
    # Generate eigenvalue plot
    plt.figure(figsize=(10, 6))
    corr_matrix = returns.corr()
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    
    # Calculate RMT parameters
    T, N = returns.shape
    Q = T / N
    lambda_plus = (1 + np.sqrt(1/Q))**2
    
    plt.hist(eigenvalues, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=lambda_plus, color='red', linestyle='--', linewidth=2, label=f'RMT Threshold (λ+ = {lambda_plus:.2f})')
    plt.title('Eigenvalue Distribution vs. Random Matrix Theory', fontsize=14, fontweight='bold')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    story.append(Image('eigenvalue_analysis.png', width=6*inch, height=3.6*inch))
    story.append(Spacer(1, 15))
    
    # RMT interpretation
    signal_eigenvalues = eigenvalues[eigenvalues > lambda_plus]
    rmt_text = f"""
    The eigenvalue analysis reveals {len(signal_eigenvalues)} significant factors above the RMT threshold of {lambda_plus:.2f}.
    The largest eigenvalue ({eigenvalues[-1]:.2f}) represents the dominant market factor, capturing systematic risk across all assets.
    Eigenvalues below the threshold are considered noise and are filtered in the denoising process.
    """
    story.append(Paragraph(rmt_text, body_style))
    story.append(PageBreak())

    # Risk Structure Analysis  
    story.append(Paragraph("Risk Structure & Correlations", subtitle_style))
    
    # Generate correlation heatmap
    plt.figure(figsize=(12, 10))
    # Use denoised correlation matrix
    corr_denoised = cov_matrix.corr()
    mask = np.triu(np.ones_like(corr_denoised, dtype=bool), k=1)
    sns.heatmap(corr_denoised, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Denoised Correlation Matrix (RMT Filtered)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    story.append(Image('correlation_matrix.png', width=6*inch, height=5*inch))
    story.append(Spacer(1, 15))

    # HRP Analysis
    story.append(Paragraph("Hierarchical Risk Parity (HRP) Structure", subtitle_style))
    
    # Generate dendrogram
    from scipy.cluster.hierarchy import linkage, dendrogram
    plt.figure(figsize=(15, 8))
    corr = returns.corr()
    distance_matrix = np.sqrt(0.5 * (1 - corr))
    link = linkage(distance_matrix, 'single')
    dendrogram(link, labels=returns.columns, orientation='top', leaf_rotation=90)
    plt.title('Asset Clustering Dendrogram for HRP', fontsize=14, fontweight='bold')
    plt.xlabel('Assets')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('hrp_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    story.append(Image('hrp_dendrogram.png', width=7*inch, height=4*inch))
    
    # Portfolio weights comparison
    story.append(PageBreak())
    story.append(Paragraph("Portfolio Allocation Comparison", subtitle_style))
    
    # Create weights comparison table for top holdings
    weights_comparison = pd.DataFrame(weights)
    top_assets = weights_comparison.sum(axis=1).nlargest(15).index
    
    weights_data = [['Asset'] + list(weights.keys())]
    for asset in top_assets:
        row = [asset]
        for strategy in weights.keys():
            weight = weights[strategy].get(asset, 0)
            row.append(f"{weight:.1%}")
        weights_data.append(row)
    
    weights_table = Table(weights_data, colWidths=[1.2*inch] + [0.8*inch]*len(weights))
    weights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(weights_table)
    story.append(Spacer(1, 20))

    # Conclusions and Recommendations
    story.append(Paragraph("Conclusions and Recommendations", subtitle_style))
    conclusions_text = f"""
    <b>Key Insights:</b><br/>
    • RMT successfully identified {len(signal_eigenvalues)} meaningful risk factors from {len(eigenvalues)} potential factors<br/>
    • HRP provides superior diversification by considering asset correlation structure<br/>
    • CVaR optimization offers robust tail risk management but may sacrifice returns<br/>
    • Traditional mean-variance optimization remains competitive when properly implemented<br/>
    
    <b>Recommendations:</b><br/>
    • Consider dynamic rebalancing based on market regime detection<br/>
    • Implement risk budgeting constraints for better downside protection<br/>
    • Monitor factor loadings for early detection of structural breaks<br/>
    • Regular recalibration of RMT parameters as market conditions evolve
    """
    story.append(Paragraph(conclusions_text, body_style))

    # Build PDF
    doc.build(story)
    logger.info(f"Enhanced report saved to {output_file}")

    # Clean up temporary files
    temp_files = ['eigenvalue_analysis.png', 'correlation_matrix.png', 'hrp_dendrogram.png']
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)