"""
Volatility and normalized returns computation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    min_periods: int = 5,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Compute rolling realized volatility from returns.

    Args:
        returns: DataFrame of returns
        window: Rolling window size
        min_periods: Minimum periods for valid calculation
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        DataFrame of volatility estimates
    """
    vol = returns.rolling(window=window, min_periods=min_periods).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def compute_volatility_normalized_returns(
    returns: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    vol_window: int = 20
) -> dict:
    """
    Compute volatility-normalized returns at multiple horizons.

    r^{(i,V)}_{t-k,t} / (sigma^{(i,V)}_t * sqrt(k))

    Args:
        returns: DataFrame of daily returns
        horizons: List of return horizons (lookback periods). Default: [1, 5, 10, 15, 20]
        vol_window: Window for volatility estimation

    Returns:
        Dictionary mapping horizon -> normalized returns DataFrame
    """
    if horizons is None:
        horizons = [1, 5, 10, 15, 20]

    daily_vol = compute_realized_volatility(returns, window=vol_window)

    normalized_returns = {}

    for k in horizons:
        if k == 1:
            k_returns = returns
        else:
            k_returns = returns.rolling(window=k).apply(
                lambda x: (1 + x).prod() - 1,
                raw=True
            )

        vol_scaled = daily_vol * np.sqrt(k)
        vol_scaled = vol_scaled.replace(0, np.nan)

        normalized = k_returns / vol_scaled
        normalized_returns[f'ret_norm_{k}d'] = normalized

    return normalized_returns
