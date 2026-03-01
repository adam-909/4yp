"""
Dataset building functions.

Functions for building complete straddle datasets and loading data efficiently.
"""

import gc
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path

from .portfolio import get_portfolio_formation_days
from .straddle import process_single_asset


def build_straddle_dataset(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    min_coverage: float = 0.95,
    return_metadata: bool = False,
    cfacpr_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Build complete straddle price and return matrices.

    Args:
        options_df: Options data with quotes and greeks
        equity_df: Underlying equity prices
        min_coverage: Minimum data coverage required
        return_metadata: If True, also return strike/expiry/spot DataFrames
        cfacpr_data: Optional DataFrame with cumulative price adjustment factors

    Returns:
        Tuple of (prices_matrix, returns_matrix, metadata) with shape (delta, N)
    """
    tickers = equity_df.columns.tolist()
    start_date = equity_df.index.min().strftime('%Y-%m-%d')
    end_date = equity_df.index.max().strftime('%Y-%m-%d')

    pfds = get_portfolio_formation_days(start_date, end_date)

    all_prices = {}
    all_returns = {}
    all_strikes = {}
    all_exdates = {}
    all_spots = {}

    for ticker in tickers:
        prices, returns, metadata = process_single_asset(
            options_df, equity_df, ticker, pfds, min_coverage,
            return_metadata=return_metadata,
            cfacpr_data=cfacpr_data
        )

        if prices is not None:
            all_prices[ticker] = prices
            all_returns[ticker] = returns
            if metadata is not None:
                all_strikes[ticker] = metadata['strike']
                all_exdates[ticker] = metadata['exdate']
                all_spots[ticker] = metadata['spot']

    if not all_prices:
        raise ValueError("No assets met the coverage requirement")

    prices_df = pd.DataFrame(all_prices)
    returns_df = pd.DataFrame(all_returns)

    result_metadata = None
    if return_metadata and all_strikes:
        result_metadata = {
            'strike': pd.DataFrame(all_strikes),
            'exdate': pd.DataFrame(all_exdates),
            'spot': pd.DataFrame(all_spots)
        }

    return prices_df, returns_df, result_metadata


def build_straddle_dataset_chunked(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    min_coverage: float = 0.95,
    chunk_size: int = 10,
    verbose: bool = True,
    return_metadata: bool = False,
    cfacpr_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Memory-efficient version of build_straddle_dataset.

    Processes tickers in chunks to reduce peak memory usage,
    with explicit garbage collection between chunks.

    Args:
        options_df: Options data with quotes and greeks
        equity_df: Underlying equity prices
        min_coverage: Minimum data coverage required
        chunk_size: Number of tickers to process per chunk
        verbose: Whether to print progress
        return_metadata: If True, also return strike/expiry/spot DataFrames
        cfacpr_data: Optional DataFrame with cumulative price adjustment factors

    Returns:
        Tuple of (prices_matrix, returns_matrix, metadata) with shape (delta, N)
    """
    tickers = equity_df.columns.tolist()
    start_date = equity_df.index.min().strftime('%Y-%m-%d')
    end_date = equity_df.index.max().strftime('%Y-%m-%d')

    pfds = get_portfolio_formation_days(start_date, end_date)

    all_prices = {}
    all_returns = {}
    all_strikes = {}
    all_exdates = {}
    all_spots = {}
    processed = 0
    total = len(tickers)

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_tickers = tickers[chunk_start:chunk_end]

        if verbose:
            print(f"  Processing tickers {chunk_start+1}-{chunk_end} of {total}...")

        for ticker in chunk_tickers:
            prices, returns, metadata = process_single_asset(
                options_df, equity_df, ticker, pfds, min_coverage,
                return_metadata=return_metadata,
                cfacpr_data=cfacpr_data
            )

            if prices is not None:
                all_prices[ticker] = prices
                all_returns[ticker] = returns
                if metadata is not None:
                    all_strikes[ticker] = metadata['strike']
                    all_exdates[ticker] = metadata['exdate']
                    all_spots[ticker] = metadata['spot']
                processed += 1

        gc.collect()

    if verbose:
        print(f"  Completed: {processed}/{total} tickers passed coverage filter")

    if not all_prices:
        raise ValueError("No assets met the coverage requirement")

    prices_df = pd.DataFrame(all_prices)
    returns_df = pd.DataFrame(all_returns)

    result_metadata = None
    if return_metadata and all_strikes:
        result_metadata = {
            'strike': pd.DataFrame(all_strikes),
            'exdate': pd.DataFrame(all_exdates),
            'spot': pd.DataFrame(all_spots)
        }

    return prices_df, returns_df, result_metadata


def load_options_year_by_year(
    data_dir: str,
    start_year: int,
    end_year: int,
    required_columns: Optional[List[str]] = None,
    dtype_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load options data year by year with memory optimization.

    Args:
        data_dir: Directory containing year-split parquet files
        start_year: First year to load
        end_year: Last year to load (inclusive)
        required_columns: Columns to load (None = all)
        dtype_map: Dict mapping column names to dtypes for conversion

    Returns:
        Concatenated DataFrame with all years
    """
    import pyarrow.parquet as pq

    if dtype_map is None:
        dtype_map = {
            'best_bid': 'float32',
            'best_offer': 'float32',
            'delta': 'float32',
            'spot_price': 'float32',
            'strike_price': 'float32',
            'strike': 'float32',
        }

    data_path = Path(data_dir)
    all_dfs = []

    for year in range(start_year, end_year + 1):
        for pattern in [f'options_filtered_{year}.parquet',
                        f'options_{year}.parquet']:
            fpath = data_path / pattern
            if fpath.exists():
                break
        else:
            continue

        table = pq.read_table(fpath, columns=required_columns)
        df = table.to_pandas()
        del table

        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        all_dfs.append(df)
        gc.collect()

    if not all_dfs:
        raise ValueError(f"No parquet files found in {data_dir}")

    result = pd.concat(all_dfs, ignore_index=True, copy=False)
    del all_dfs
    gc.collect()

    return result
