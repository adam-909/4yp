"""
Option-specific features (moneyness, time-to-expiry).
"""

import numpy as np
import pandas as pd


def compute_log_moneyness(
    spot_prices: pd.DataFrame,
    strike_prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute log-moneyness: log(S/K).

    Args:
        spot_prices: DataFrame of underlying spot prices
        strike_prices: DataFrame of option strike prices

    Returns:
        DataFrame of log-moneyness values
    """
    moneyness = spot_prices / strike_prices
    moneyness = moneyness.replace(0, np.nan)
    moneyness = moneyness.clip(lower=1e-6)

    return np.log(moneyness)


def compute_time_to_expiry(
    dates: pd.DatetimeIndex,
    expiry_dates: pd.DataFrame,
    trading_days_per_year: int = 252
) -> pd.DataFrame:
    """
    Compute time-to-expiry in years.

    Args:
        dates: Index of observation dates
        expiry_dates: DataFrame of expiration dates for each asset
        trading_days_per_year: Number of trading days per year

    Returns:
        DataFrame of time-to-expiry values
    """
    tte = pd.DataFrame(index=dates, columns=expiry_dates.columns)

    for col in expiry_dates.columns:
        for date in dates:
            if date in expiry_dates.index:
                expiry = expiry_dates.loc[date, col]
                if pd.notna(expiry):
                    days = (expiry - date).days
                    tte.loc[date, col] = max(days, 0) / trading_days_per_year

    return tte.astype(float)
