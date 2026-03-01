"""
Stock split adjustment functions.

Handles CRSP cfacpr adjustment factors for normalizing prices across splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime


def compute_split_adjustment_factors(
    equity_prices_path: str,
    security_names_path: str,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Compute cumulative split adjustment factors from CRSP equity data.

    The cfacpr column in CRSP data is a cumulative factor that accounts for
    all splits up to each date. To make prices comparable over time, we
    normalize to the most recent (latest) adjustment factor.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: List of tickers to compute adjustments for

    Returns:
        DataFrame with adjustment factors indexed by date, columns = tickers
        Multiply straddle prices by these factors to get split-adjusted prices.
    """
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    adjustment_factors = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date').set_index('date')

        latest_cfacpr = ticker_data['cfacpr'].iloc[-1]
        ticker_data['adj_factor'] = latest_cfacpr / ticker_data['cfacpr']

        adjustment_factors[ticker] = ticker_data['adj_factor']

    adj_df = pd.DataFrame(adjustment_factors)
    adj_df.index = pd.to_datetime(adj_df.index)

    return adj_df


def apply_split_adjustment(
    straddle_prices: pd.DataFrame,
    adjustment_factors: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply split adjustment factors to straddle prices.

    Args:
        straddle_prices: DataFrame of straddle prices (index=dates, columns=tickers)
        adjustment_factors: DataFrame of adjustment factors from compute_split_adjustment_factors

    Returns:
        Split-adjusted straddle prices
    """
    common_tickers = straddle_prices.columns.intersection(adjustment_factors.columns)

    adj_aligned = adjustment_factors[common_tickers].reindex(straddle_prices.index)
    adj_aligned = adj_aligned.ffill().bfill()

    adjusted_prices = straddle_prices.copy()
    adjusted_prices[common_tickers] = straddle_prices[common_tickers] * adj_aligned

    return adjusted_prices


def get_split_dates(
    equity_prices_path: str,
    security_names_path: str,
    tickers: Optional[List[str]] = None
) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Identify dates where stock splits occurred for each ticker.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: Optional list of tickers (None = all)

    Returns:
        Dict mapping ticker -> list of (date, split_ratio) tuples
    """
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    if tickers is None:
        tickers = list(ticker_to_secid.keys())

    split_dates = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date')

        ticker_data['cfacpr_prev'] = ticker_data['cfacpr'].shift(1)
        splits = ticker_data[ticker_data['cfacpr'] != ticker_data['cfacpr_prev']].dropna()

        if not splits.empty:
            split_info = []
            for _, row in splits.iterrows():
                ratio = row['cfacpr_prev'] / row['cfacpr']
                split_info.append((row['date'], ratio))
            split_dates[ticker] = split_info

    return split_dates


def load_cfacpr_data(
    equity_prices_path: str,
    security_names_path: str,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cumulative price adjustment factors as a DataFrame.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: Optional list of tickers (None = all)

    Returns:
        DataFrame with cfacpr values indexed by date, columns = tickers
    """
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    if tickers is None:
        tickers = list(ticker_to_secid.keys())

    cfacpr_dict = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date').set_index('date')
        cfacpr_dict[ticker] = ticker_data['cfacpr']

    cfacpr_df = pd.DataFrame(cfacpr_dict)
    cfacpr_df.index = pd.to_datetime(cfacpr_df.index)

    return cfacpr_df
