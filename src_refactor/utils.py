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
    Stitch using a SAME-DAY anchor (avoid Fri-vs-Mon scaling that compounds to ~0).

    1) Prefer using roll_date itself if both sides have a price.
    2) Otherwise, use the earliest date >= roll_date where BOTH have non-NaN.
    3) Scale back from that anchor date onward, and keep front up to that date inclusive.
    """
    roll = pd.Timestamp(roll_date).normalize()

    # ensure normalized datetime index
    f = front_prices.copy()
    b = back_prices.copy()
    f.index = pd.to_datetime(f.index).normalize()
    b.index = pd.to_datetime(b.index).normalize()

    # candidate dates where both are available on/after roll
    common = f.loc[f.index >= roll].dropna().index.intersection(b.loc[b.index >= roll].dropna().index)

    if len(common) == 0:
        # no same-day anchor available -> just concatenate without scaling
        return pd.concat([f.loc[f.index < roll], b.loc[b.index >= roll]]).sort_index()

    anchor_date = common[0]
    front_anchor = f.loc[anchor_date]
    back_anchor = b.loc[anchor_date]

    if pd.isna(front_anchor) or pd.isna(back_anchor) or back_anchor == 0:
        return pd.concat([f.loc[f.index < roll], b.loc[b.index >= roll]]).sort_index()

    ratio = front_anchor / back_anchor

    if debug:
        print(f"stitch roll={roll.date()} anchor={anchor_date.date()} front={front_anchor:.6g} back={back_anchor:.6g} ratio={ratio:.6g}")

    # keep front THROUGH anchor_date
    front_keep = f.loc[f.index <= anchor_date]

    # scale back FROM anchor_date (and then drop the duplicate anchor point from back to avoid double-counting)
    back_scaled = b.loc[b.index >= anchor_date] * ratio
    back_scaled = back_scaled.loc[back_scaled.index > anchor_date]

    return pd.concat([front_keep, back_scaled]).sort_index()

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
