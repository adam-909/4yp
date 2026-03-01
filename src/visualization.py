"""
Visualization Module for Options Momentum Trading

This module provides plotting utilities for analyzing straddle data,
returns distributions, feature correlations, and sector-level analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List, Dict, Tuple


# Default style settings
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})


def plot_straddle_prices(
    prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    title: str = 'Delta-Neutral Straddle Prices',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot straddle price time series.

    Args:
        prices: DataFrame of straddle prices
        tickers: List of tickers to plot (default: first 5)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if tickers is None:
        tickers = prices.columns[:5].tolist()

    fig, ax = plt.subplots(figsize=figsize)

    for ticker in tickers:
        if ticker in prices.columns:
            ax.plot(prices.index, prices[ticker], label=ticker, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Straddle Price ($)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_price_vs_vix(
    prices: pd.Series,
    vix: pd.Series,
    ticker: str,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot straddle price against VIX for comparison.

    Args:
        prices: Series of straddle prices
        vix: Series of VIX values
        ticker: Ticker symbol for title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Straddle Price ($)', color=color1)
    ax1.plot(prices.index, prices.values, color=color1, alpha=0.8, label='Straddle Price')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('VIX', color=color2)
    ax2.plot(vix.index, vix.values, color=color2, alpha=0.6, label='VIX')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title(f'{ticker} Delta-Neutral Straddle Price vs VIX')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig


def plot_return_distributions(
    returns: pd.DataFrame,
    horizons: Dict[str, pd.DataFrame],
    figsize: Tuple[int, int] = (14, 4)
) -> plt.Figure:
    """
    Plot histograms of return distributions at different horizons.

    Args:
        returns: DataFrame of daily returns
        horizons: Dictionary mapping horizon name to returns DataFrame
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_plots = len(horizons)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, (name, ret_df) in zip(axes, horizons.items()):
        data = ret_df.values.flatten()
        data = data[~np.isnan(data)]

        ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(data), color='green', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(data):.3f}')

        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Returns')
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_sector_returns(
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    title: str = 'Sector-Averaged Returns',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot returns grouped by sector.

    Args:
        returns: DataFrame of returns
        sector_map: Dictionary mapping ticker -> sector
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Group by sector
    sector_returns = {}

    for ticker in returns.columns:
        if ticker in sector_map:
            sector = sector_map[ticker]
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(returns[ticker])

    # Average within sectors
    sector_avg = {}
    for sector, series_list in sector_returns.items():
        sector_df = pd.concat(series_list, axis=1)
        sector_avg[sector] = sector_df.mean(axis=1)

    sector_df = pd.DataFrame(sector_avg)

    fig, ax = plt.subplots(figsize=figsize)

    for sector in sector_df.columns:
        ax.plot(sector_df.index, sector_df[sector], label=sector, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Return')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_sector_pie_chart(
    sector_counts: Dict[str, int],
    title: str = 'Assets by Sector',
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot pie chart of sector composition.

    Args:
        sector_counts: Dictionary mapping sector -> count
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sectors = list(sector_counts.keys())
    counts = list(sector_counts.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=sectors,
        autopct='%1.0f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.8
    )

    # Add counts in the middle of each wedge
    for i, (wedge, count) in enumerate(zip(wedges, counts)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = 0.6 * np.cos(np.deg2rad(ang))
        y = 0.6 * np.sin(np.deg2rad(ang))
        ax.annotate(str(count), xy=(x, y), ha='center', va='center', fontsize=10)

    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_return_statistics_table(
    returns: pd.DataFrame,
    horizons: Dict[str, int] = {'Daily': 1, 'Weekly': 5, 'Monthly': 20},
    figsize: Tuple[int, int] = (10, 3)
) -> plt.Figure:
    """
    Create a table figure showing return statistics.

    Args:
        returns: DataFrame of daily returns
        horizons: Dictionary mapping name -> horizon in days
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    stats = []

    for name, k in horizons.items():
        if k == 1:
            ret_data = returns
        else:
            ret_data = returns.rolling(window=k).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )

        data = ret_data.values.flatten()
        data = data[~np.isnan(data)]

        stats.append({
            'Returns': name,
            'Mean': f'{np.mean(data):.3f}',
            'Median': f'{np.median(data):.3f}',
            'Std Dev': f'{np.std(data):.3f}',
            'Skewness': f'{pd.Series(data).skew():.3f}',
            'Kurtosis': f'{pd.Series(data).kurtosis():.3f}'
        })

    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * len(stats_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title('Straddle Return Statistics', pad=20)
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = 'Return Correlation Matrix',
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot correlation heatmap of returns.

    Args:
        returns: DataFrame of returns
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation')

    # Set ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)

    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_feature_distributions(
    feature_tensor: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Plot distributions of all features.

    Args:
        feature_tensor: Feature tensor of shape (delta, F, N)
        feature_names: List of feature names
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        data = feature_tensor[:, idx, :].flatten()
        data = data[~np.isnan(data)]

        ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=np.mean(data), color='red', linestyle='--', alpha=0.7)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(len(feature_names), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=12)
    plt.tight_layout()

    return fig


def plot_cumulative_returns(
    returns: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    title: str = 'Cumulative Straddle Returns',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Plot cumulative returns over time.

    Args:
        returns: DataFrame of returns
        tickers: List of tickers to plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if tickers is None:
        tickers = returns.columns[:10].tolist()

    cum_returns = (1 + returns[tickers]).cumprod() - 1

    fig, ax = plt.subplots(figsize=figsize)

    for ticker in tickers:
        ax.plot(cum_returns.index, cum_returns[ticker], label=ticker, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_summary_dashboard(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: Dict[str, str],
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a summary dashboard with multiple panels.

    Args:
        prices: DataFrame of straddle prices
        returns: DataFrame of straddle returns
        sector_map: Dictionary mapping ticker -> sector
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Panel 1: Sample price series
    ax1 = fig.add_subplot(2, 2, 1)
    sample_tickers = prices.columns[:3].tolist()
    for ticker in sample_tickers:
        ax1.plot(prices.index, prices[ticker] / prices[ticker].iloc[0], label=ticker, alpha=0.8)
    ax1.set_title('Normalized Straddle Prices (Sample)')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Panel 2: Daily return distribution
    ax2 = fig.add_subplot(2, 2, 2)
    data = returns.values.flatten()
    data = data[~np.isnan(data)]
    ax2.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)
    ax2.axvline(x=np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.4f}')
    ax2.set_title('Daily Return Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Density')
    ax2.legend()

    # Panel 3: Sector composition
    ax3 = fig.add_subplot(2, 2, 3)
    sector_counts = {}
    for ticker in prices.columns:
        if ticker in sector_map:
            sector = sector_map[ticker]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    ax3.barh(list(sector_counts.keys()), list(sector_counts.values()), color='steelblue', alpha=0.7)
    ax3.set_title('Assets by Sector')
    ax3.set_xlabel('Count')

    # Panel 4: Average correlation by sector
    ax4 = fig.add_subplot(2, 2, 4)
    corr_matrix = returns.corr()
    avg_corr = corr_matrix.mean().mean()
    ax4.text(0.5, 0.5, f'Average Pairwise\nCorrelation:\n{avg_corr:.3f}',
             fontsize=20, ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Cross-Asset Correlation')
    ax4.axis('off')

    plt.suptitle('Straddle Data Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig
