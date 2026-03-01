"""
Refactored Algorithm 1: Straddle Prices & Returns Pipeline

FIXES APPLIED:
1. Missing-day handling: Reindex to full trading calendar before coverage check
2. Trading calendar: Use equity_prices.index instead of freq='B'
3. Split adjustment: REMOVED - option prices already split-adjusted
4. PFD flexibility: Search ±3 days for call/put pair if missing on exact PFD
5. Delta filtering: Only require delta at PFD for weights, not globally
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .utils import (
    compute_delta_neutral_weights,
    select_atm_strike,
    stitch_price_series,
    backfill_missing_ema,
)


def get_portfolio_formation_days(
    start_date: str,
    end_date: str
) -> List[datetime]:
    """
    Generate Portfolio Formation Days (2nd Monday of each month).

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        List of PFD datetime objects
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    pfds = []
    current = start.replace(day=1)

    while current <= end:
        # Find first day of month
        first = current.replace(day=1)

        # Find first Monday
        days_until_monday = (7 - first.weekday()) % 7
        if first.weekday() == 0:
            days_until_monday = 0
        first_monday = first + pd.Timedelta(days=days_until_monday)

        # Second Monday is 7 days later
        second_monday = first_monday + pd.Timedelta(days=7)

        if start <= second_monday <= end:
            pfds.append(second_monday.to_pydatetime())

        # Move to next month
        current = current + relativedelta(months=1)

    return pfds


def find_valid_pfd_data(
    options_df: pd.DataFrame,
    pfd: datetime,
    atm_strike: float,
    exdate: datetime,
    max_offset: int = 3
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], datetime]:
    """
    Find valid call/put pair on PFD or nearby days (±max_offset).

    FIX FOR CONCERN 4: Instead of failing when no data on exact PFD,
    search nearby days.

    Args:
        options_df: Filtered options for this asset
        pfd: Target PFD date
        atm_strike: Selected ATM strike
        exdate: Expiration date
        max_offset: Maximum days to search (±)

    Returns:
        Tuple of (call_df, put_df, actual_date) or (None, None, pfd) if not found
    """
    # Try exact PFD first
    offsets = [0] + [i for j in range(1, max_offset + 1) for i in (j, -j)]

    for offset in offsets:
        check_date = pfd + pd.Timedelta(days=offset)
        day_options = options_df[options_df['date'] == check_date]

        if day_options.empty:
            continue

        call_pfd = day_options[
            (day_options['cp_flag'] == 'C') &
            (day_options['strike'] == atm_strike) &
            (day_options['exdate'] == exdate)
        ]
        put_pfd = day_options[
            (day_options['cp_flag'] == 'P') &
            (day_options['strike'] == atm_strike) &
            (day_options['exdate'] == exdate)
        ]

        # FIX FOR CONCERN 5: Only check delta here, not globally
        if not call_pfd.empty and not put_pfd.empty:
            # Check if delta is available for weight computation
            call_delta = call_pfd.iloc[0]['delta']
            put_delta = put_pfd.iloc[0]['delta']
            if pd.notna(call_delta) and pd.notna(put_delta):
                return call_pfd, put_pfd, check_date

    return None, None, pfd


def process_single_asset_v2(
    options_data: pd.DataFrame,
    equity_prices: pd.DataFrame,
    asset_id: str,
    pfds: List[datetime],
    cfacpr_series: Optional[pd.Series] = None,
    min_coverage: float = 0.95,
    return_metadata: bool = False,
    debug: bool = False
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[dict]]:
    """
    Process options data for a single underlying asset.

    REFACTORED VERSION with all fixes applied:
    1. Reindex to trading calendar (not freq='B')
    2. Use equity_prices.index as trading calendar
    3. Split adjustment using cfacpr (divide prices by cfacpr)
    4. ±3 day lookahead for PFD data
    5. Delta required only at PFD

    Args:
        options_data: DataFrame with option quotes (must have 'ticker' column)
        equity_prices: DataFrame with underlying equity prices (index=dates, columns=tickers)
        asset_id: Ticker symbol for the underlying asset
        pfds: List of portfolio formation dates
        min_coverage: Minimum data coverage required (default 95%)
        return_metadata: If True, also return strike/expiry/spot data
        debug: If True, print detailed debug info

    Returns:
        Tuple of (straddle_prices, straddle_returns, metadata) or (None, None, None)
    """
    period_prices = {}
    all_metadata = []

    # Filter options for this asset
    asset_options_all = options_data[options_data['ticker'] == asset_id].copy()

    if asset_options_all.empty:
        if debug:
            print(f"  {asset_id}: No options found")
        return None, None, None

    # FIX FOR CONCERN 2: Use equity index as trading calendar
    trading_calendar = equity_prices.index

    # Debug counters
    debug_pfd_count = 0
    debug_no_spot = 0
    debug_no_options = 0
    debug_no_expiry = 0
    debug_no_strike = 0
    debug_no_calls_puts = 0
    debug_no_data = 0
    debug_used_offset = 0

    for i, pfd in enumerate(pfds[:-1]):
        next_pfd = pfds[i + 1]
        pfd_ts = pd.Timestamp(pfd)

        # Get spot price at PFD
        # FIX: Use trading calendar for lookup
        pfd_idx = trading_calendar[trading_calendar >= pfd_ts]
        if len(pfd_idx) == 0:
            debug_no_spot += 1
            continue
        pfd_trading_day = pfd_idx[0]

        if pfd_trading_day not in equity_prices.index:
            debug_no_spot += 1
            continue

        spot = equity_prices.loc[pfd_trading_day, asset_id]
        if pd.isna(spot):
            debug_no_spot += 1
            continue

        debug_pfd_count += 1

        # NEW APPROACH: Find best (expiry, strike) combination from PFD-day quotes
        # This ensures we select options that actually have quotes on PFD day

        # Step 1: Get options quoted on or near PFD (±3 days)
        offsets = [0] + [i for j in range(1, 4) for i in (j, -j)]
        pfd_day_options = None
        actual_pfd_date = pfd

        for offset in offsets:
            check_date = pfd + pd.Timedelta(days=offset)
            day_opts = asset_options_all[asset_options_all['date'] == check_date]
            if not day_opts.empty:
                pfd_day_options = day_opts
                actual_pfd_date = check_date
                break

        if pfd_day_options is None or pfd_day_options.empty:
            debug_no_options += 1
            continue

        if actual_pfd_date != pfd:
            debug_used_offset += 1

        # Step 2: Filter by moneyness using spot
        call_mask = pfd_day_options['cp_flag'] == 'C'
        put_mask = pfd_day_options['cp_flag'] == 'P'

        moneyness = pd.Series(index=pfd_day_options.index, dtype=float)
        moneyness[call_mask] = spot / pfd_day_options.loc[call_mask, 'strike']
        moneyness[put_mask] = pfd_day_options.loc[put_mask, 'strike'] / spot

        pfd_options_atm = pfd_day_options[
            (moneyness >= 0.9) & (moneyness <= 1.1)
        ]

        if pfd_options_atm.empty:
            debug_no_options += 1
            continue

        # Step 3: Find (expiry, strike) pairs that have BOTH call and put with delta
        next_month = pfd + relativedelta(months=1)
        target_month_start = next_month.replace(day=1)
        target_month_end = (target_month_start + relativedelta(months=1)) - pd.Timedelta(days=1)
        next_pfd_ts = pd.Timestamp(next_pfd)

        # Filter to expirations in target window
        pfd_options_exp = pfd_options_atm[
            (pfd_options_atm['exdate'] >= target_month_start) &
            (pfd_options_atm['exdate'] <= target_month_end)
        ]

        if pfd_options_exp.empty:
            # Fallback: broader window
            pfd_options_exp = pfd_options_atm[
                (pfd_options_atm['exdate'] > pfd) &
                (pfd_options_atm['exdate'] <= next_pfd + pd.Timedelta(days=30))
            ]
            if pfd_options_exp.empty:
                debug_no_expiry += 1
                continue

        # Find valid (expiry, strike) combinations with call+put and delta
        valid_combos = []
        for exdate_candidate in pfd_options_exp['exdate'].unique():
            exp_opts = pfd_options_exp[pfd_options_exp['exdate'] == exdate_candidate]
            for strike_candidate in exp_opts['strike'].unique():
                strike_opts = exp_opts[exp_opts['strike'] == strike_candidate]
                calls = strike_opts[strike_opts['cp_flag'] == 'C']
                puts = strike_opts[strike_opts['cp_flag'] == 'P']

                if not calls.empty and not puts.empty:
                    call_delta = calls.iloc[0]['delta']
                    put_delta = puts.iloc[0]['delta']
                    if pd.notna(call_delta) and pd.notna(put_delta):
                        valid_combos.append({
                            'exdate': exdate_candidate,
                            'strike': strike_candidate,
                            'call': calls.iloc[0],
                            'put': puts.iloc[0],
                            'dist_to_spot': abs(strike_candidate - spot)
                        })

        if not valid_combos:
            debug_no_calls_puts += 1
            continue

        # Step 4: Select best combo - prefer expiry >= next_pfd, then closest to ATM
        combos_after_next_pfd = [c for c in valid_combos if pd.Timestamp(c['exdate']) >= next_pfd_ts]

        if combos_after_next_pfd:
            # Among expiries >= next_pfd, take earliest expiry, then closest strike
            best = min(combos_after_next_pfd, key=lambda c: (c['exdate'], c['dist_to_spot']))
        else:
            # No expiry >= next_pfd, take latest expiry, then closest strike
            best = max(valid_combos, key=lambda c: (c['exdate'], -c['dist_to_spot']))

        exdate = best['exdate']
        atm_strike = best['strike']
        call_pfd = pd.DataFrame([best['call']])
        put_pfd = pd.DataFrame([best['put']])

        # Compute delta-neutral weights (delta required here only)
        delta_call = call_pfd.iloc[0]['delta']
        delta_put = put_pfd.iloc[0]['delta']
        w_call, w_put = compute_delta_neutral_weights(delta_call, delta_put)

        if pd.isna(w_call) or pd.isna(w_put):
            debug_no_calls_puts += 1
            continue

        # FIX FOR CONCERN 2: Use actual trading days from equity calendar
        # Now that we select expiry >= next_pfd, the period covers PFD to next_pfd
        period_start = pfd_ts
        period_end = pd.Timestamp(next_pfd)
        trading_days = trading_calendar[
            (trading_calendar >= period_start) &
            (trading_calendar < period_end)  # Up to but not including next PFD
        ]

        period_data = {}

        for day in trading_days:
            day_options = asset_options_all[asset_options_all['date'] == day]

            # FIX FOR CONCERN 1 & 5: Don't skip - we'll reindex later
            # Just try to get prices if available
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
                # FIX FOR CONCERN 1: Keep day in period but with NaN
                period_data[day] = np.nan
                continue

            call_mid = (call_day.iloc[0]['best_bid'] + call_day.iloc[0]['best_offer']) / 2
            put_mid = (put_day.iloc[0]['best_bid'] + put_day.iloc[0]['best_offer']) / 2
            straddle_price = w_call * call_mid + w_put * put_mid

            # SPLIT ADJUSTMENT: Divide by cfacpr to get split-adjusted price
            if cfacpr_series is not None and day in cfacpr_series.index:
                cfacpr = cfacpr_series.loc[day]
                if pd.notna(cfacpr) and cfacpr > 0:
                    straddle_price = straddle_price / cfacpr

            period_data[day] = straddle_price

            if return_metadata:
                current_spot = equity_prices.loc[day, asset_id] if day in equity_prices.index else spot
                all_metadata.append({
                    'date': day,
                    'strike': atm_strike,
                    'exdate': exdate,
                    'spot': current_spot
                })

        if period_data:
            period_series = pd.Series(period_data)
            # FIX FOR CONCERN 1: Reindex to full trading calendar for this period
            period_series = period_series.reindex(trading_days)
            period_prices[pfd] = period_series
        else:
            debug_no_data += 1

    # Print debug summary
    total_pfds = len(pfds) - 1
    if debug:
        print(f"    {asset_id}: PFDs={debug_pfd_count}/{total_pfds}, "
              f"no_spot={debug_no_spot}, no_opts={debug_no_options}, "
              f"no_exp={debug_no_expiry}, no_strike={debug_no_strike}, "
              f"no_cp={debug_no_calls_puts}, no_data={debug_no_data}, "
              f"used_offset={debug_used_offset}, periods={len(period_prices)}")

    if not period_prices:
        return None, None, None

    # Stitch price series
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
                debug=debug and i < 3
            )

    prices = stitched_prices.sort_index()

    # FIX FOR CONCERN 2: Use actual trading calendar for expected days
    full_calendar = trading_calendar[
        (trading_calendar >= pd.Timestamp(pfds[0])) &
        (trading_calendar <= pd.Timestamp(pfds[-1]))
    ]

    # FIX FOR CONCERN 1: Reindex to full calendar BEFORE coverage check
    prices = prices.reindex(full_calendar)

    expected_days = len(full_calendar)
    actual_days = prices.notna().sum()
    actual_coverage = actual_days / expected_days

    if debug:
        print(f"      -> {asset_id} coverage: {actual_days}/{expected_days} = {actual_coverage:.1%} "
              f"(need {min_coverage:.0%})")

    if actual_coverage < min_coverage:
        return None, None, None

    # Fill missing with EMA (Algorithm 1, Line 18)
    prices = backfill_missing_ema(prices, span=5)

    # Compute returns
    returns = prices.pct_change()

    metadata = None
    if return_metadata and all_metadata:
        meta_df = pd.DataFrame(all_metadata).set_index('date')
        metadata = {
            'strike': meta_df['strike'],
            'exdate': meta_df['exdate'],
            'spot': meta_df['spot']
        }

    return prices, returns, metadata


def build_straddle_dataset_v2(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    cfacpr_df: Optional[pd.DataFrame] = None,
    min_coverage: float = 0.95,
    verbose: bool = True,
    return_metadata: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Build complete straddle price and return matrices.

    REFACTORED VERSION with split adjustment using cfacpr.

    Args:
        options_df: Options data with quotes and greeks
        equity_df: Underlying equity prices (index=dates, columns=tickers)
        cfacpr_df: Cumulative split adjustment factors (index=dates, columns=tickers)
                   If provided, straddle prices are divided by cfacpr to get split-adjusted prices
        min_coverage: Minimum data coverage required
        verbose: Print progress
        return_metadata: If True, also return strike/expiry/spot DataFrames
        debug: If True, print detailed debug info

    Returns:
        Tuple of (prices_matrix, returns_matrix, metadata)
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
    passed = 0
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{total}...")

        # Get cfacpr series for this ticker if available
        cfacpr_series = None
        if cfacpr_df is not None and ticker in cfacpr_df.columns:
            cfacpr_series = cfacpr_df[ticker]

        prices, returns, metadata = process_single_asset_v2(
            options_df, equity_df, ticker, pfds,
            cfacpr_series=cfacpr_series,
            min_coverage=min_coverage,
            return_metadata=return_metadata,
            debug=debug
        )

        if prices is not None:
            all_prices[ticker] = prices
            all_returns[ticker] = returns
            if metadata is not None:
                all_strikes[ticker] = metadata['strike']
                all_exdates[ticker] = metadata['exdate']
                all_spots[ticker] = metadata['spot']
            passed += 1

    if verbose:
        print(f"  Completed: {passed}/{total} tickers passed {min_coverage:.0%} coverage filter")

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
