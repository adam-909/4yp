"""
Precompute and cache equity log returns for rolling Pearson correlation graphs.

This script downloads equity price data from yfinance and computes log returns,
saving them to a CSV file that can be used by lstm_gcn_rolling_pearson notebook.

Usage:
    python examples/create_equity_returns_cache.py

Output:
    data/graph_structure/equity_returns/log_returns.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from settings.default import ALL_TICKERS


def download_equity_returns(
    tickers: list,
    start_date: str = "2009-01-01",
    end_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Download equity prices and compute log returns.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data download
        end_date: End date for data download

    Returns:
        DataFrame with log returns, indexed by date, columns are tickers
    """
    # Map tickers that need different symbols for yfinance
    ticker_map = {'BRKB': 'BRK-B'}

    log_return_dict = {}
    failed = []

    print(f"Downloading equity data for {len(tickers)} tickers...")
    print(f"Date range: {start_date} to {end_date}")

    for i, ticker in enumerate(tickers):
        yf_ticker = ticker_map.get(ticker, ticker)
        try:
            stock = yf.Ticker(yf_ticker)
            hist = stock.history(start=start_date, end=end_date)
            if len(hist) > 0:
                log_returns = np.log(hist['Close']).diff()
                log_return_dict[ticker] = log_returns
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(tickers)} tickers...")
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"  Warning: Failed to load {ticker}: {e}")

    print(f"\nSuccessfully loaded: {len(log_return_dict)} tickers")
    if failed:
        print(f"Failed to load: {failed}")

    # Create DataFrame
    log_returns_df = pd.DataFrame(log_return_dict)
    log_returns_df.index = pd.to_datetime(log_returns_df.index).tz_localize(None)
    log_returns_df = log_returns_df.dropna(how='all')
    log_returns_df = log_returns_df.sort_index()

    return log_returns_df


def main():
    # Output directory
    output_dir = os.path.join("data", "graph_structure", "equity_returns")
    os.makedirs(output_dir, exist_ok=True)

    # Download and compute log returns
    log_returns_df = download_equity_returns(
        tickers=ALL_TICKERS,
        start_date="2009-01-01",
        end_date="2024-01-01",
    )

    # Save to CSV
    output_path = os.path.join(output_dir, "log_returns.csv")
    log_returns_df.to_csv(output_path)

    print(f"\nEquity log returns saved to: {output_path}")
    print(f"Shape: {log_returns_df.shape}")
    print(f"Date range: {log_returns_df.index.min().date()} to {log_returns_df.index.max().date()}")

    # Also compute and display some statistics
    print("\nCorrelation statistics (full period):")
    corr_matrix = log_returns_df.corr().values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    upper = corr_matrix[np.triu_indices(len(ALL_TICKERS), k=1)]
    print(f"  Mean correlation: {np.mean(upper):.3f}")
    print(f"  Std correlation: {np.std(upper):.3f}")

    # Edge counts at different thresholds
    print("\nEdge counts at different thresholds (full period):")
    for tau in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        np.fill_diagonal(corr_matrix, 0)
        edges = (np.abs(corr_matrix) >= tau).sum() // 2
        print(f"  tau={tau:.2f}: {edges} edges")


if __name__ == "__main__":
    main()
