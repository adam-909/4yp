"""
Portfolio formation and delta-neutral weighting functions.

Implements PFD (Portfolio Formation Day) calculation and delta-neutral straddle weighting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_portfolio_formation_days(start_date: str, end_date: str) -> List[datetime]:
    """
    Get the second Monday of each month (Portfolio Formation Day - PFD).

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        List of PFD dates
    """
    pfds = []
    current = pd.Timestamp(start_date).to_pydatetime()
    end = pd.Timestamp(end_date).to_pydatetime()

    while current <= end:
        first_of_month = current.replace(day=1)

        days_until_monday = (7 - first_of_month.weekday()) % 7
        first_monday = first_of_month + pd.Timedelta(days=days_until_monday)

        second_monday = first_monday + pd.Timedelta(days=7)

        if second_monday <= end:
            pfds.append(second_monday)

        current = (current.replace(day=1) + relativedelta(months=1))

    return pfds


def compute_delta_neutral_weights(delta_call: float, delta_put: float) -> Tuple[float, float]:
    """
    Compute normalized weights for delta-neutral straddle.

    The weights ensure the combined position has zero delta exposure.

    Args:
        delta_call: Delta of the call option
        delta_put: Delta of the put option (typically negative)

    Returns:
        Tuple of (call_weight, put_weight)
    """
    denominator = delta_call - delta_put
    if abs(denominator) < 1e-8:
        return 0.5, 0.5

    w_call = delta_call / denominator
    w_put = -delta_put / denominator

    return w_call, w_put


def compute_straddle_price(
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    delta_call: float,
    delta_put: float
) -> float:
    """
    Compute delta-neutral straddle price using mid prices.

    Args:
        call_bid: Call option bid price
        call_ask: Call option ask price
        put_bid: Put option bid price
        put_ask: Put option ask price
        delta_call: Call option delta
        delta_put: Put option delta

    Returns:
        Delta-neutral straddle price
    """
    call_mid = (call_bid + call_ask) / 2
    put_mid = (put_bid + put_ask) / 2

    w_call, w_put = compute_delta_neutral_weights(delta_call, delta_put)

    return w_call * call_mid + w_put * put_mid
