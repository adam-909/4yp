"""
Utility functions for Algorithm 1 straddle pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime


def compute_delta_neutral_weights(
    delta_call: float,
    delta_put: float
) -> Tuple[float, float]:
    """
    Compute delta-neutral weights for straddle construction.

    Algorithm 1, Lines 5-6:
    w_call = delta_call / (delta_call - delta_put)
    w_put = -delta_put / (delta_call - delta_put)

    Args:
        delta_call: Call option delta (positive, typically 0.3-0.7)
        delta_put: Put option delta (negative, typically -0.7 to -0.3)

    Returns:
        Tuple of (w_call, w_put) weights
    """
    if pd.isna(delta_call) or pd.isna(delta_put):
        return np.nan, np.nan

    denom = delta_call - delta_put
    if abs(denom) < 1e-10:
        return 0.5, 0.5

    w_call = delta_call / denom
    w_put = -delta_put / denom

    return w_call, w_put


def select_atm_strike(
    options_df: pd.DataFrame,
    spot_price: float,
    expiration_date: datetime
) -> Optional[float]:
    """
    Select the at-the-money strike price closest to spot.

    Args:
        options_df: DataFrame with option data (must have 'strike', 'exdate')
        spot_price: Current underlying spot price
        expiration_date: Target expiration date

    Returns:
        ATM strike price or None if no valid options
    """
    mask = options_df['exdate'] == expiration_date
    filtered = options_df[mask]

    if filtered.empty:
        return None

    strikes = filtered['strike'].unique()
    if len(strikes) == 0:
        return None

    atm_strike = strikes[np.argmin(np.abs(strikes - spot_price))]
    return atm_strike


def stitch_price_series(
    front_prices: pd.Series,
    back_prices: pd.Series,
    roll_date: datetime,
    debug: bool = False
) -> pd.Series:
    """
    Stitch front and back month price series to avoid PnL jumps.

    Uses ratio adjustment to maintain continuity at the roll date.

    Args:
        front_prices: Front month price series
        back_prices: Back month price series
        roll_date: Date of the roll
        debug: If True, print debug info

    Returns:
        Stitched price series
    """
    roll_date_norm = pd.Timestamp(roll_date).normalize()
    front_index_norm = pd.to_datetime(front_prices.index).normalize()
    back_index_norm = pd.to_datetime(back_prices.index).normalize()

    front_at_roll = front_prices[front_index_norm == roll_date_norm]
    back_at_roll = back_prices[back_index_norm == roll_date_norm]

    front_roll = front_at_roll.iloc[0] if len(front_at_roll) > 0 else np.nan
    back_roll = back_at_roll.iloc[0] if len(back_at_roll) > 0 else np.nan

    front_before = front_prices[front_index_norm < roll_date_norm]
    back_after = back_prices[back_index_norm >= roll_date_norm]

    if debug:
        print(f"    stitch: front={len(front_prices)}, back={len(back_prices)}, "
              f"front_before={len(front_before)}, back_after={len(back_after)}")

    if np.isnan(front_roll) or np.isnan(back_roll) or back_roll == 0:
        return pd.concat([front_before, back_after])

    ratio = front_roll / back_roll
    adjusted_back = back_prices * ratio
    adjusted_back_after = adjusted_back[back_index_norm >= roll_date_norm]

    stitched = pd.concat([front_before, adjusted_back_after])
    return stitched.sort_index()


def backfill_missing_ema(prices: pd.Series, span: int = 5) -> pd.Series:
    """
    Fill missing prices using forward fill then EMA.

    Args:
        prices: Price series with potential NaN values
        span: EMA span in days

    Returns:
        Series with NaN values filled
    """
    filled = prices.copy()

    # First forward fill
    filled = filled.ffill()

    # Then backward fill any remaining leading NaNs
    filled = filled.bfill()

    # If still any NaN, use EMA
    if filled.isna().any():
        ema = filled.ewm(span=span, adjust=False).mean()
        filled = filled.fillna(ema)

    return filled
