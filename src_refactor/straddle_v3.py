"""
Algorithm 1 v3: Matches original implementation more closely.

Key difference from v2:
- Builds a continuous daily series with (price, exdate) for each day
- Stitches using exdate-based roll detection (like original)
- Normalizes to final price = 1
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .utils import compute_delta_neutral_weights


def get_portfolio_formation_days(start_date: str, end_date: str) -> List[datetime]:
    """Generate Portfolio Formation Days (2nd Monday of each month)."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    pfds = []
    current = start.replace(day=1)

    while current <= end:
        first = current.replace(day=1)
        days_until_monday = (7 - first.weekday()) % 7
        if first.weekday() == 0:
            days_until_monday = 0
        first_monday = first + pd.Timedelta(days=days_until_monday)
        second_monday = first_monday + pd.Timedelta(days=7)

        if start <= second_monday <= end:
            pfds.append(second_monday.to_pydatetime())
        current = current + relativedelta(months=1)

    return pfds


def stitch_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stitch contracts using exdate-based roll detection.

    This is a direct port of the original stitch_contracts function.

    Args:
        df: DataFrame with columns ['straddle_price', 'exdate'], indexed by date

    Returns:
        DataFrame with 'stitched_price' column, normalized to final=1
    """
    df = df.sort_index().reset_index()
    df = df.rename(columns={'index': 'date'}) if 'date' not in df.columns else df

    stitched_rows = []
    current_factor = 1.0

    # Process first row
    if len(df) == 0:
        return pd.DataFrame()

    first_row = df.iloc[0].copy()
    first_row['stitched_price'] = first_row['straddle_price'] * current_factor
    stitched_rows.append(first_row)

    # Process remaining rows
    i = 1
    while i < len(df):
        # Check for rollover: exdate changes
        if df.iloc[i]['exdate'] != df.iloc[i-1]['exdate']:
            # Compute new scaling factor
            prev_stitched = stitched_rows[-1]['stitched_price']
            curr_raw = df.iloc[i]['straddle_price']

            if curr_raw != 0 and not pd.isna(curr_raw):
                current_factor = prev_stitched / curr_raw

            # SKIP the rollover row (like original)
            i += 1
            continue
        else:
            new_row = df.iloc[i].copy()
            new_row['stitched_price'] = new_row['straddle_price'] * current_factor
            stitched_rows.append(new_row)
        i += 1

    if not stitched_rows:
        return pd.DataFrame()

    stitched_df = pd.DataFrame(stitched_rows)
    stitched_df['date'] = pd.to_datetime(stitched_df['date'])
    stitched_df = stitched_df.set_index('date')

    # Normalize so final price = 1
    if len(stitched_df) > 0:
        final_price = stitched_df['stitched_price'].iloc[-1]
        if final_price != 0 and not pd.isna(final_price):
            stitched_df['stitched_price'] = stitched_df['stitched_price'] / final_price

    return stitched_df


def process_single_asset_v3(
    options_data: pd.DataFrame,
    equity_prices: pd.DataFrame,
    asset_id: str,
    pfds: List[datetime],
    min_coverage: float = 0.95,
    debug: bool = False
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Process a single asset using the original's approach:
    1. Build continuous daily series with (price, exdate)
    2. Stitch using exdate-based roll detection
    3. Normalize to final = 1
    """
    asset_options = options_data[options_data['ticker'] == asset_id].copy()

    if asset_options.empty:
        if debug:
            print(f"  {asset_id}: No options found")
        return None, None

    trading_calendar = equity_prices.index

    # Build a mapping: for each PFD, what contract (strike, exdate, weights) to use
    contract_specs = {}  # pfd -> {strike, exdate, w_call, w_put}

    for i, pfd in enumerate(pfds[:-1]):
        next_pfd = pfds[i + 1]
        pfd_ts = pd.Timestamp(pfd)

        # Get spot at PFD
        pfd_idx = trading_calendar[trading_calendar >= pfd_ts]
        if len(pfd_idx) == 0:
            continue
        pfd_day = pfd_idx[0]

        if pfd_day not in equity_prices.index:
            continue
        spot = equity_prices.loc[pfd_day, asset_id]
        if pd.isna(spot):
            continue

        # Find options on or near PFD
        pfd_options = None
        for offset in [0, 1, -1, 2, -2, 3, -3]:
            check_date = pfd + pd.Timedelta(days=offset)
            day_opts = asset_options[asset_options['date'] == check_date]
            if not day_opts.empty:
                pfd_options = day_opts
                break

        if pfd_options is None:
            continue

        # Filter by moneyness
        call_mask = pfd_options['cp_flag'] == 'C'
        put_mask = pfd_options['cp_flag'] == 'P'
        moneyness = pd.Series(index=pfd_options.index, dtype=float)
        moneyness[call_mask] = spot / pfd_options.loc[call_mask, 'strike']
        moneyness[put_mask] = pfd_options.loc[put_mask, 'strike'] / spot

        atm_options = pfd_options[(moneyness >= 0.9) & (moneyness <= 1.1)]
        if atm_options.empty:
            continue

        # Find expiration in next month
        next_month = pfd + relativedelta(months=1)
        target_start = next_month.replace(day=1)
        target_end = (target_start + relativedelta(months=1)) - pd.Timedelta(days=1)

        exp_options = atm_options[
            (atm_options['exdate'] >= target_start) &
            (atm_options['exdate'] <= target_end)
        ]

        if exp_options.empty:
            # Fallback: any expiration after PFD
            exp_options = atm_options[atm_options['exdate'] > pfd]

        if exp_options.empty:
            continue

        # Find best (exdate, strike) combo with both call and put with delta
        best_combo = None
        for exdate in exp_options['exdate'].unique():
            exp_opts = exp_options[exp_options['exdate'] == exdate]
            for strike in exp_opts['strike'].unique():
                strike_opts = exp_opts[exp_opts['strike'] == strike]
                calls = strike_opts[strike_opts['cp_flag'] == 'C']
                puts = strike_opts[strike_opts['cp_flag'] == 'P']

                if calls.empty or puts.empty:
                    continue

                call_delta = calls.iloc[0]['delta']
                put_delta = puts.iloc[0]['delta']

                if pd.isna(call_delta) or pd.isna(put_delta):
                    continue

                dist = abs(strike - spot)
                if best_combo is None or dist < best_combo['dist']:
                    best_combo = {
                        'exdate': exdate,
                        'strike': strike,
                        'w_call': call_delta / (call_delta - put_delta),
                        'w_put': -put_delta / (call_delta - put_delta),
                        'dist': dist
                    }

        if best_combo:
            contract_specs[pfd] = best_combo

    if not contract_specs:
        if debug:
            print(f"  {asset_id}: No valid contracts found")
        return None, None

    # Now build continuous daily series
    daily_data = []
    sorted_pfds = sorted(contract_specs.keys())

    for i, pfd in enumerate(sorted_pfds):
        spec = contract_specs[pfd]

        # Determine period end
        if i + 1 < len(sorted_pfds):
            period_end = sorted_pfds[i + 1]
        else:
            # Last period: go until data ends or next month
            period_end = pfd + relativedelta(months=1)

        # Get trading days in this period
        period_days = trading_calendar[
            (trading_calendar >= pd.Timestamp(pfd)) &
            (trading_calendar < pd.Timestamp(period_end))
        ]

        for day in period_days:
            day_options = asset_options[
                (asset_options['date'] == day) &
                (asset_options['exdate'] == spec['exdate'])
            ]

            if day_options.empty:
                continue

            # Find strike (allow small tolerance for rounding)
            available_strikes = day_options['strike'].unique()
            closest = min(available_strikes, key=lambda x: abs(x - spec['strike']))
            if abs(closest - spec['strike']) / spec['strike'] > 0.02:
                continue

            calls = day_options[(day_options['cp_flag'] == 'C') & (day_options['strike'] == closest)]
            puts = day_options[(day_options['cp_flag'] == 'P') & (day_options['strike'] == closest)]

            if calls.empty or puts.empty:
                continue

            call_mid = (calls.iloc[0]['best_bid'] + calls.iloc[0]['best_offer']) / 2
            put_mid = (puts.iloc[0]['best_bid'] + puts.iloc[0]['best_offer']) / 2

            straddle_price = spec['w_call'] * call_mid + spec['w_put'] * put_mid

            daily_data.append({
                'date': day,
                'straddle_price': straddle_price,
                'exdate': spec['exdate'],
                'strike': closest
            })

    if not daily_data:
        if debug:
            print(f"  {asset_id}: No daily data generated")
        return None, None

    daily_df = pd.DataFrame(daily_data).set_index('date')

    if debug:
        print(f"  {asset_id}: {len(daily_df)} daily prices, {len(contract_specs)} contracts")

    # Apply stitch_contracts (exdate-based, like original)
    stitched_df = stitch_contracts(daily_df)

    if stitched_df.empty:
        return None, None

    # Check coverage
    full_calendar = trading_calendar[
        (trading_calendar >= pd.Timestamp(sorted_pfds[0])) &
        (trading_calendar <= pd.Timestamp(sorted_pfds[-1]))
    ]

    prices = stitched_df['stitched_price'].reindex(full_calendar)

    actual_coverage = prices.notna().sum() / len(full_calendar)

    if debug:
        print(f"    {asset_id}: coverage = {actual_coverage:.1%} (need {min_coverage:.0%})")

    if actual_coverage < min_coverage:
        return None, None

    # Fill missing with interpolation (like original)
    prices = prices.interpolate(method='linear')
    prices = prices.ffill().bfill()

    # Compute returns
    returns = prices.pct_change()

    return prices, returns


def build_straddle_dataset_v3(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    min_coverage: float = 0.95,
    verbose: bool = True,
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build straddle dataset using v3 approach (matches original).
    """
    tickers = equity_df.columns.tolist()
    start_date = equity_df.index.min().strftime('%Y-%m-%d')
    end_date = equity_df.index.max().strftime('%Y-%m-%d')

    pfds = get_portfolio_formation_days(start_date, end_date)

    all_prices = {}
    all_returns = {}
    passed = 0

    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(tickers)}...")

        prices, returns = process_single_asset_v3(
            options_df, equity_df, ticker, pfds,
            min_coverage=min_coverage,
            debug=debug
        )

        if prices is not None:
            all_prices[ticker] = prices
            all_returns[ticker] = returns
            passed += 1

    if verbose:
        print(f"  Completed: {passed}/{len(tickers)} passed {min_coverage:.0%} coverage")

    if not all_prices:
        raise ValueError("No assets met coverage requirement")

    return pd.DataFrame(all_prices), pd.DataFrame(all_returns)
