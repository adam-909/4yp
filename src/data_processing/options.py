"""
Option filtering and selection functions.

Handles ATM strike selection and moneyness filtering.
"""

import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime


def select_atm_strike(
    options_df: pd.DataFrame,
    spot_price: float,
    expiration_date: datetime
) -> Optional[float]:
    """
    Select the at-the-money strike price closest to the spot price.

    Args:
        options_df: DataFrame with option data
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

    atm_strike = strikes[np.argmin(np.abs(strikes - spot_price))]

    return atm_strike


def filter_by_moneyness(
    options_df: pd.DataFrame,
    spot_price: float,
    min_moneyness: float = 0.9,
    max_moneyness: float = 1.1
) -> pd.DataFrame:
    """
    Filter options by moneyness to exclude illiquid contracts.

    Moneyness is S/K for calls and K/S for puts.

    Args:
        options_df: DataFrame with option data
        spot_price: Current spot price
        min_moneyness: Minimum moneyness threshold
        max_moneyness: Maximum moneyness threshold

    Returns:
        Filtered DataFrame
    """
    df = options_df.copy()

    call_mask = df['cp_flag'] == 'C'
    put_mask = df['cp_flag'] == 'P'

    df.loc[call_mask, 'moneyness'] = spot_price / df.loc[call_mask, 'strike']
    df.loc[put_mask, 'moneyness'] = df.loc[put_mask, 'strike'] / spot_price

    mask = (df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)

    return df[mask]
