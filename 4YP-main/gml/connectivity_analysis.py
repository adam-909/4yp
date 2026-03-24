"""
Connectivity vs Performance Analysis for Rolling Graph Models.

This module provides visualization tools for analyzing the relationship
between graph connectivity (edge counts) and model performance over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_connectivity_vs_performance(adjacencies, results_df, test_dates):
    """
    Plot connectivity metrics alongside performance metrics over time.

    Creates three plots:
    1. Dual-axis: Edge count vs cumulative returns over time
    2. Dual-axis: Edge count vs rolling Sharpe ratio over time
    3. Scatter plot: Edge count vs window returns with regression line

    Args:
        adjacencies: np.ndarray of shape (num_windows, num_tickers, num_tickers)
        results_df: DataFrame with columns ['time', 'captured_returns', ...]
        test_dates: Array of dates corresponding to each window
    """
    # Compute edge counts per window
    edge_counts = np.array([(adj > 0).sum() / 2 for adj in adjacencies])

    # Convert test_dates to datetime
    dates = pd.to_datetime(test_dates)

    # Compute per-window returns (aggregate across tickers)
    results_df = results_df.copy()
    results_df['time'] = pd.to_datetime(results_df['time'])
    daily_captured = results_df.groupby('time')['captured_returns'].mean()

    # Align dates - get returns for each window's last date
    window_returns = []
    for d in dates:
        if d in daily_captured.index:
            window_returns.append(daily_captured.loc[d])
        else:
            # Find closest date
            closest_idx = daily_captured.index.get_indexer([d], method='nearest')[0]
            window_returns.append(daily_captured.iloc[closest_idx])
    window_returns = np.array(window_returns)

    # Compute cumulative returns aligned to windows
    cumulative_returns = np.cumprod(1 + window_returns) - 1

    # Rolling 20-window Sharpe (approximate)
    rolling_window = 20
    rolling_sharpe = pd.Series(window_returns).rolling(rolling_window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    ).values

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ---- Plot 1: Dual-axis - Edge Count vs Cumulative Returns ----
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ln1 = ax1.plot(dates, edge_counts, 'b-', alpha=0.7, linewidth=1, label='Edge Count')
    ax1.fill_between(dates, edge_counts, alpha=0.2, color='blue')
    ax1.set_ylabel('Number of Edges', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ln2 = ax1_twin.plot(dates, cumulative_returns * 100, 'r-', linewidth=1.5, label='Cumulative Return')
    ax1_twin.set_ylabel('Cumulative Return (%)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.axhline(y=0, color='red', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Date')
    ax1.set_title('Graph Connectivity vs Cumulative Returns Over Time')

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- Plot 2: Dual-axis - Edge Count vs Rolling Sharpe ----
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ln1 = ax2.plot(dates, edge_counts, 'b-', alpha=0.7, linewidth=1, label='Edge Count')
    ax2.fill_between(dates, edge_counts, alpha=0.2, color='blue')
    ax2.set_ylabel('Number of Edges', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ln2 = ax2_twin.plot(dates, rolling_sharpe, 'g-', linewidth=1.5, label=f'Rolling {rolling_window}-Window Sharpe')
    ax2_twin.set_ylabel('Rolling Sharpe', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2_twin.axhline(y=0, color='green', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Date')
    ax2.set_title('Graph Connectivity vs Rolling Sharpe Ratio')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ---- Plot 3: Scatter Plot - Edges vs Window Returns ----
    ax3 = axes[2]

    # Color by time (earlier = lighter, later = darker)
    colors = np.linspace(0, 1, len(edge_counts))
    scatter = ax3.scatter(edge_counts, window_returns * 100, c=colors, cmap='viridis',
                          alpha=0.6, s=20, label='Windows')

    # Add regression line
    z = np.polyfit(edge_counts, window_returns * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(edge_counts.min(), edge_counts.max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')

    # Correlation
    corr = np.corrcoef(edge_counts, window_returns)[0, 1]

    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Number of Edges')
    ax3.set_ylabel('Window Return (%)')
    ax3.set_title(f'Edge Count vs Window Returns (Correlation: {corr:.3f})')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Colorbar for time
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Time (early -> late)')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nConnectivity-Performance Statistics:")
    print(f"  Correlation (edges vs returns): {corr:.4f}")
    print(f"  Mean edges: {edge_counts.mean():.1f}")
    print(f"  Mean window return: {window_returns.mean()*100:.4f}%")

    # Quartile analysis
    q1, q2, q3 = np.percentile(edge_counts, [25, 50, 75])
    low_mask = edge_counts <= q1
    high_mask = edge_counts >= q3

    print(f"\n  Low connectivity (<={q1:.0f} edges) mean return: {window_returns[low_mask].mean()*100:.4f}%")
    print(f"  High connectivity (>={q3:.0f} edges) mean return: {window_returns[high_mask].mean()*100:.4f}%")

    return {
        'edge_counts': edge_counts,
        'window_returns': window_returns,
        'correlation': corr,
        'dates': dates,
    }
