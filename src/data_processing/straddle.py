"""
Straddle construction and processing functions.

Implements Algorithm 1 for building delta-neutral straddle price series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .portfolio import compute_delta_neutral_weights
from .options import select_atm_strike, filter_by_moneyness


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

    if debug and len(front_prices) < 50:
        print(f"        stitch: front={len(front_prices)}, back={len(back_prices)}, "
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
    Backfill missing prices using exponential moving average.

    Args:
        prices: Price series with potential missing values
        span: EMA span in days

    Returns:
        Series with missing values filled
    """
    filled = prices.copy()
    filled = filled.ffill()

    if filled.isna().any():
        ema = filled.ewm(span=span, adjust=False).mean()
        filled = filled.fillna(ema)

    return filled


def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute simple returns from price series.

    r_{t+1,t} = (p_{t+1} - p_t) / p_t

    Args:
        prices: Price series

    Returns:
        Return series
    """
    returns = prices.pct_change()
    return returns


def process_single_asset(
    options_data: pd.DataFrame,
    equity_prices: pd.DataFrame,
    asset_id: str,
    pfds: List[datetime],
    min_coverage: float = 0.95,
    return_metadata: bool = False,
    cfacpr_data: Optional[pd.DataFrame] = None
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[dict]]:
    """
    Process options data for a single underlying asset.

    Implements Algorithm 1 from the paper, including price stitching
    at PFD boundaries to avoid incorrect PnL jumps during rolls.

    Args:
        options_data: DataFrame with option quotes and greeks
        equity_prices: DataFrame with underlying equity prices
        asset_id: Identifier for the underlying asset
        pfds: List of portfolio formation dates
        min_coverage: Minimum data coverage required (default 95%)
        return_metadata: If True, also return strike/expiry/spot data for features
        cfacpr_data: Optional DataFrame with cumulative price adjustment factors

    Returns:
        Tuple of (straddle_prices, straddle_returns, metadata) or (None, None, None)
    """
    period_prices = {}
    all_metadata = []

    asset_options_all = options_data[options_data['ticker'] == asset_id].copy()

    if asset_options_all.empty:
        print(f"  DEBUG {asset_id}: No options found for this ticker")
        return None, None, None

    equity_index_norm = pd.to_datetime(equity_prices.index).normalize()

    debug_pfd_count = 0
    debug_no_spot = 0
    debug_no_options = 0
    debug_no_expiry = 0
    debug_no_strike = 0
    debug_no_calls_puts = 0
    debug_no_data = 0

    for i, pfd in enumerate(pfds[:-1]):
        next_pfd = pfds[i + 1]

        pfd_normalized = pd.Timestamp(pfd).normalize()

        pfd_match = equity_index_norm == pfd_normalized
        if not pfd_match.any():
            debug_no_spot += 1
            continue
        pfd_idx = equity_prices.index[pfd_match][0]
        spot = equity_prices.loc[pfd_idx, asset_id]

        if pd.isna(spot):
            debug_no_spot += 1
            continue

        debug_pfd_count += 1

        pfd_cfacpr = None
        if cfacpr_data is not None and asset_id in cfacpr_data.columns:
            cfacpr_index_norm = pd.to_datetime(cfacpr_data.index).normalize()
            cfacpr_match = cfacpr_index_norm == pfd_normalized
            if cfacpr_match.any():
                cfacpr_idx = cfacpr_data.index[cfacpr_match][0]
                pfd_cfacpr = cfacpr_data.loc[cfacpr_idx, asset_id]

        asset_options = filter_by_moneyness(asset_options_all, spot)

        if asset_options.empty:
            debug_no_options += 1
            continue

        next_month = pfd + relativedelta(months=1)
        target_month_start = next_month.replace(day=1)
        target_month_end = (target_month_start + relativedelta(months=1)) - pd.Timedelta(days=1)

        target_expirations = asset_options[
            (asset_options['exdate'] >= target_month_start) &
            (asset_options['exdate'] <= target_month_end)
        ]['exdate'].unique()

        if len(target_expirations) == 0:
            target_expirations = asset_options[
                (asset_options['exdate'] > pfd) &
                (asset_options['exdate'] <= next_pfd + pd.Timedelta(days=10))
            ]['exdate'].unique()
            if len(target_expirations) == 0:
                debug_no_expiry += 1
                continue

        exdate = min(target_expirations)
        atm_strike = select_atm_strike(asset_options, spot, exdate)

        if atm_strike is None:
            debug_no_strike += 1
            continue

        pfd_options = asset_options[asset_options['date'] == pfd]

        call_pfd = pfd_options[
            (pfd_options['cp_flag'] == 'C') &
            (pfd_options['strike'] == atm_strike) &
            (pfd_options['exdate'] == exdate)
        ]
        put_pfd = pfd_options[
            (pfd_options['cp_flag'] == 'P') &
            (pfd_options['strike'] == atm_strike) &
            (pfd_options['exdate'] == exdate)
        ]

        if call_pfd.empty or put_pfd.empty:
            debug_no_calls_puts += 1
            continue

        delta_call = call_pfd.iloc[0]['delta']
        delta_put = put_pfd.iloc[0]['delta']
        w_call, w_put = compute_delta_neutral_weights(delta_call, delta_put)

        trading_days = pd.date_range(start=pfd, end=next_pfd, freq='B')
        period_data = []

        for day in trading_days:
            day_options = asset_options[asset_options['date'] == day]

            if day_options.empty:
                continue

            call_day = day_options[
                (day_options['cp_flag'] == 'C') &
                (day_options['strike'] == atm_strike) &
                (day_options['exdate'] == exdate)
            ]
            put_day = day_options[
                (day_options['cp_flag'] == 'P') &
                (day_options['strike'] == atm_strike) &
                (day_options['exdate'] == exdate)
            ]

            if call_day.empty or put_day.empty:
                continue

            if day in equity_prices.index:
                current_spot = equity_prices.loc[day, asset_id]
            else:
                current_spot = spot

            call_mid = (call_day.iloc[0]['best_bid'] + call_day.iloc[0]['best_offer']) / 2
            put_mid = (put_day.iloc[0]['best_bid'] + put_day.iloc[0]['best_offer']) / 2
            straddle_price = w_call * call_mid + w_put * put_mid

            period_data.append({'date': day, 'price': straddle_price})

            if return_metadata:
                all_metadata.append({
                    'date': day,
                    'strike': atm_strike,
                    'exdate': exdate,
                    'spot': current_spot
                })

        if period_data:
            period_df = pd.DataFrame(period_data).set_index('date')
            period_prices[pfd] = period_df['price']
        else:
            debug_no_data += 1

    total_pfds = len(pfds) - 1
    print(f"    {asset_id}: PFDs={debug_pfd_count}/{total_pfds}, "
          f"no_spot={debug_no_spot}, no_opts={debug_no_options}, "
          f"no_exp={debug_no_expiry}, no_strike={debug_no_strike}, "
          f"no_cp={debug_no_calls_puts}, no_data={debug_no_data}, "
          f"periods={len(period_prices)}")

    if not period_prices:
        return None, None, None

    sorted_pfds = sorted(period_prices.keys())

    if len(sorted_pfds) == 1:
        stitched_prices = period_prices[sorted_pfds[0]]
    else:
        stitched_prices = period_prices[sorted_pfds[0]].copy()

        for i in range(1, len(sorted_pfds)):
            roll_date = sorted_pfds[i]
            back_prices = period_prices[roll_date]

            stitched_prices = stitch_price_series(
                front_prices=stitched_prices,
                back_prices=back_prices,
                roll_date=roll_date,
                debug=(i < 3 and asset_id == 'AAPL')
            )

    prices = stitched_prices.sort_index()

    expected_days = len(pd.date_range(start=pfds[0], end=pfds[-1], freq='B'))
    actual_coverage = len(prices) / expected_days

    print(f"      -> {asset_id} coverage: {len(prices)}/{expected_days} = {actual_coverage:.1%} "
          f"(need {min_coverage:.0%}), pfds[0]={pfds[0].date()}, pfds[-1]={pfds[-1].date()}")

    if actual_coverage < min_coverage:
        return None, None, None

    prices = backfill_missing_ema(prices, span=5)
    returns = compute_returns(prices)

    metadata = None
    if return_metadata and all_metadata:
        meta_df = pd.DataFrame(all_metadata).set_index('date')
        metadata = {
            'strike': meta_df['strike'],
            'exdate': meta_df['exdate'],
            'spot': meta_df['spot']
        }

    return prices, returns, metadata
