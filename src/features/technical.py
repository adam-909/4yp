"""
Technical indicators (EMA, MACD).
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from .volatility import compute_realized_volatility


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute exponential moving average.

    Args:
        series: Input series
        span: EMA span

    Returns:
        EMA series
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    prices: pd.DataFrame,
    short_span: int,
    long_span: int
) -> pd.DataFrame:
    """
    Compute MACD indicator.

    MACD = EMA(short) - EMA(long)

    Args:
        prices: DataFrame of prices
        short_span: Short-term EMA span
        long_span: Long-term EMA span

    Returns:
        DataFrame of MACD values
    """
    ema_short = prices.ewm(span=short_span, adjust=False).mean()
    ema_long = prices.ewm(span=long_span, adjust=False).mean()

    return ema_short - ema_long


def compute_volatility_normalized_macd(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    macd_params: Optional[List[Tuple[int, int]]] = None,
    vol_window: int = 20
) -> dict:
    """
    Compute volatility-normalized MACD indicators.

    I^{(i,V)}_{MACD,t}(S,L) as defined in Equation (54).

    Args:
        prices: DataFrame of straddle prices
        returns: DataFrame of straddle returns
        macd_params: List of (short_span, long_span) tuples
        vol_window: Window for volatility estimation

    Returns:
        Dictionary mapping parameter pair -> normalized MACD DataFrame
    """
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    daily_vol = compute_realized_volatility(returns, window=vol_window)

    normalized_macd = {}

    for short_span, long_span in macd_params:
        macd = compute_macd(prices, short_span, long_span)

        price_vol = prices * daily_vol
        price_vol = price_vol.replace(0, np.nan)

        normalized = macd / price_vol
        normalized_macd[f'macd_{short_span}_{long_span}'] = normalized

    return normalized_macd
