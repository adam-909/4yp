"""
Data Processing Module for Options Momentum Trading

This module implements the straddle formation and data processing pipeline
as described in Section 4 of the paper "A Geometric Deep Learning Approach
to Momentum Options Trading".

Key components:
- Delta-neutral straddle formation
- Price stitching across contract rolls
- Missing data handling with EMA backfill
- Return computation
- Stock split adjustment
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta


def compute_split_adjustment_factors(
    equity_prices_path: str,
    security_names_path: str,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Compute cumulative split adjustment factors from CRSP equity data.

    The cfacpr column in CRSP data is a cumulative factor that accounts for
    all splits up to each date. To make prices comparable over time, we
    normalize to the most recent (latest) adjustment factor.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: List of tickers to compute adjustments for

    Returns:
        DataFrame with adjustment factors indexed by date, columns = tickers
        Multiply straddle prices by these factors to get split-adjusted prices.
    """
    # Load data
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    # Build secid -> ticker mapping
    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    adjustment_factors = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date').set_index('date')

        # Get the latest (most recent) cfacpr as the base
        latest_cfacpr = ticker_data['cfacpr'].iloc[-1]

        # Adjustment factor: multiply by this to normalize to latest basis
        # If cfacpr was 28 (pre-split) and latest is 4 (post-split),
        # factor = 4/28 = 1/7, so pre-split prices get divided by 7
        ticker_data['adj_factor'] = latest_cfacpr / ticker_data['cfacpr']

        adjustment_factors[ticker] = ticker_data['adj_factor']

    # Combine into DataFrame
    adj_df = pd.DataFrame(adjustment_factors)
    adj_df.index = pd.to_datetime(adj_df.index)

    return adj_df


def apply_split_adjustment(
    straddle_prices: pd.DataFrame,
    adjustment_factors: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply split adjustment factors to straddle prices.

    Args:
        straddle_prices: DataFrame of straddle prices (index=dates, columns=tickers)
        adjustment_factors: DataFrame of adjustment factors from compute_split_adjustment_factors

    Returns:
        Split-adjusted straddle prices
    """
    # Align adjustment factors to straddle price dates
    common_tickers = straddle_prices.columns.intersection(adjustment_factors.columns)

    # Reindex adjustment factors to match straddle dates
    adj_aligned = adjustment_factors[common_tickers].reindex(straddle_prices.index)

    # Forward fill adjustment factors (they only change on split dates)
    adj_aligned = adj_aligned.ffill().bfill()

    # Apply adjustment
    adjusted_prices = straddle_prices.copy()
    adjusted_prices[common_tickers] = straddle_prices[common_tickers] * adj_aligned

    return adjusted_prices


def get_split_dates(
    equity_prices_path: str,
    security_names_path: str,
    tickers: Optional[List[str]] = None
) -> Dict[str, List[Tuple[datetime, float]]]:
    """
    Identify dates where stock splits occurred for each ticker.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: Optional list of tickers (None = all)

    Returns:
        Dict mapping ticker -> list of (date, split_ratio) tuples
    """
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    if tickers is None:
        tickers = list(ticker_to_secid.keys())

    split_dates = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date')

        # Find where cfacpr changes
        ticker_data['cfacpr_prev'] = ticker_data['cfacpr'].shift(1)
        splits = ticker_data[ticker_data['cfacpr'] != ticker_data['cfacpr_prev']].dropna()

        if not splits.empty:
            split_info = []
            for _, row in splits.iterrows():
                ratio = row['cfacpr_prev'] / row['cfacpr']
                split_info.append((row['date'], ratio))
            split_dates[ticker] = split_info

    return split_dates


def load_cfacpr_data(
    equity_prices_path: str,
    security_names_path: str,
    tickers: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cumulative price adjustment factors as a DataFrame.

    This creates a DataFrame that can be passed to build_straddle_dataset
    to handle stock splits during straddle construction.

    Args:
        equity_prices_path: Path to equity_prices_raw.parquet
        security_names_path: Path to security_names.parquet
        tickers: Optional list of tickers (None = all)

    Returns:
        DataFrame with cfacpr values indexed by date, columns = tickers
    """
    equity_df = pd.read_parquet(equity_prices_path)
    secnames_df = pd.read_parquet(security_names_path)

    ticker_to_secid = dict(zip(secnames_df['ticker'], secnames_df['secid'].astype(int)))

    if tickers is None:
        tickers = list(ticker_to_secid.keys())

    cfacpr_dict = {}

    for ticker in tickers:
        if ticker not in ticker_to_secid:
            continue

        secid = ticker_to_secid[ticker]
        ticker_data = equity_df[equity_df['secid'] == secid].copy()

        if ticker_data.empty:
            continue

        ticker_data = ticker_data.sort_values('date').set_index('date')
        cfacpr_dict[ticker] = ticker_data['cfacpr']

    cfacpr_df = pd.DataFrame(cfacpr_dict)
    cfacpr_df.index = pd.to_datetime(cfacpr_df.index)

    return cfacpr_df


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
        # Find first day of month
        first_of_month = current.replace(day=1)

        # Find first Monday
        days_until_monday = (7 - first_of_month.weekday()) % 7
        first_monday = first_of_month + pd.Timedelta(days=days_until_monday)

        # Second Monday is 7 days later
        second_monday = first_monday + pd.Timedelta(days=7)

        if second_monday <= end:
            pfds.append(second_monday)

        # Move to next month
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
    # Filter by expiration
    mask = options_df['exdate'] == expiration_date
    filtered = options_df[mask]

    if filtered.empty:
        return None

    # Get unique strikes
    strikes = filtered['strike'].unique()

    # Find closest to spot
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

    # Compute moneyness based on option type
    call_mask = df['cp_flag'] == 'C'
    put_mask = df['cp_flag'] == 'P'

    df.loc[call_mask, 'moneyness'] = spot_price / df.loc[call_mask, 'strike']
    df.loc[put_mask, 'moneyness'] = df.loc[put_mask, 'strike'] / spot_price

    # Filter
    mask = (df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)

    return df[mask]


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
    # Normalize roll_date for comparison
    roll_date_norm = pd.Timestamp(roll_date).normalize()
    front_index_norm = pd.to_datetime(front_prices.index).normalize()
    back_index_norm = pd.to_datetime(back_prices.index).normalize()

    # Get prices at roll date using normalized comparison
    front_at_roll = front_prices[front_index_norm == roll_date_norm]
    back_at_roll = back_prices[back_index_norm == roll_date_norm]

    front_roll = front_at_roll.iloc[0] if len(front_at_roll) > 0 else np.nan
    back_roll = back_at_roll.iloc[0] if len(back_at_roll) > 0 else np.nan

    # Use normalized comparison for filtering
    front_before = front_prices[front_index_norm < roll_date_norm]
    back_after = back_prices[back_index_norm >= roll_date_norm]

    if debug and len(front_prices) < 50:  # Only debug first few
        print(f"        stitch: front={len(front_prices)}, back={len(back_prices)}, "
              f"front_before={len(front_before)}, back_after={len(back_after)}")

    if np.isnan(front_roll) or np.isnan(back_roll) or back_roll == 0:
        # Can't compute ratio, just concatenate
        return pd.concat([front_before, back_after])

    # Compute adjustment ratio
    ratio = front_roll / back_roll

    # Adjust back month prices
    adjusted_back = back_prices * ratio

    # Use the already-computed normalized filtering
    adjusted_back_after = adjusted_back[back_index_norm >= roll_date_norm]

    # Concatenate
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

    # Forward fill first, then use EMA for remaining
    filled = filled.ffill()

    # For any remaining NaNs at the start, use backward EMA
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
                     (index=date, columns=tickers). Used to handle stock splits.

    Returns:
        Tuple of (straddle_prices, straddle_returns, metadata) or (None, None, None)
        where metadata is a dict with 'strike', 'exdate', 'spot' Series (if return_metadata=True)
    """
    # Collect prices by PFD period for stitching
    period_prices = {}  # pfd -> pd.Series of prices for that period
    # Also collect metadata for computing real moneyness and TTE
    all_metadata = []  # List of dicts with date, strike, exdate, spot

    # Filter options for this asset ONCE (doesn't depend on PFD)
    asset_options_all = options_data[options_data['ticker'] == asset_id].copy()

    if asset_options_all.empty:
        print(f"  DEBUG {asset_id}: No options found for this ticker")
        return None, None, None

    # Pre-normalize equity_prices index for efficient lookup
    equity_index_norm = pd.to_datetime(equity_prices.index).normalize()

    # DEBUG: Track why PFDs fail
    debug_pfd_count = 0
    debug_no_spot = 0
    debug_no_options = 0
    debug_no_expiry = 0
    debug_no_strike = 0
    debug_no_calls_puts = 0
    debug_no_data = 0

    for i, pfd in enumerate(pfds[:-1]):
        next_pfd = pfds[i + 1]

        # Normalize PFD for date comparison
        pfd_normalized = pd.Timestamp(pfd).normalize()

        # Get spot price at PFD (using normalized date lookup)
        pfd_match = equity_index_norm == pfd_normalized
        if not pfd_match.any():
            debug_no_spot += 1
            continue
        pfd_idx = equity_prices.index[pfd_match][0]
        spot = equity_prices.loc[pfd_idx, asset_id]

        # Check for NaN spot
        if pd.isna(spot):
            debug_no_spot += 1
            continue

        debug_pfd_count += 1

        # Get cfacpr at PFD if available (for split detection)
        pfd_cfacpr = None
        if cfacpr_data is not None and asset_id in cfacpr_data.columns:
            cfacpr_index_norm = pd.to_datetime(cfacpr_data.index).normalize()
            cfacpr_match = cfacpr_index_norm == pfd_normalized
            if cfacpr_match.any():
                cfacpr_idx = cfacpr_data.index[cfacpr_match][0]
                pfd_cfacpr = cfacpr_data.loc[cfacpr_idx, asset_id]

        # Filter by moneyness (depends on spot which changes per PFD)
        asset_options = filter_by_moneyness(asset_options_all, spot)

        if asset_options.empty:
            debug_no_options += 1
            continue

        # Select ATM strike
        # Get expiration dates specifically in the FOLLOWING calendar month
        # Paper specifies: "strike closest to the money which expires in the following month"
        next_month = pfd + relativedelta(months=1)
        target_month_start = next_month.replace(day=1)
        target_month_end = (target_month_start + relativedelta(months=1)) - pd.Timedelta(days=1)

        target_expirations = asset_options[
            (asset_options['exdate'] >= target_month_start) &
            (asset_options['exdate'] <= target_month_end)
        ]['exdate'].unique()

        if len(target_expirations) == 0:
            # Fallback: try the original broader window if no options in target month
            target_expirations = asset_options[
                (asset_options['exdate'] > pfd) &
                (asset_options['exdate'] <= next_pfd + pd.Timedelta(days=10))
            ]['exdate'].unique()
            if len(target_expirations) == 0:
                debug_no_expiry += 1
                continue

        # Select the earliest expiration in the target month
        exdate = min(target_expirations)
        atm_strike = select_atm_strike(asset_options, spot, exdate)

        if atm_strike is None:
            debug_no_strike += 1
            continue

        # Get delta at PFD for weight computation (Algorithm 1, Lines 5-6)
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

        # Compute straddle prices for each trading day (Algorithm 1, Lines 7-10)
        trading_days = pd.date_range(start=pfd, end=next_pfd, freq='B')
        period_data = []

        for day in trading_days:
            # Filter options for this specific day, strike, and expiration
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

            # Get current spot
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

    # Print debug summary for ALL tickers (debug mode)
    total_pfds = len(pfds) - 1
    print(f"    {asset_id}: PFDs={debug_pfd_count}/{total_pfds}, "
          f"no_spot={debug_no_spot}, no_opts={debug_no_options}, "
          f"no_exp={debug_no_expiry}, no_strike={debug_no_strike}, "
          f"no_cp={debug_no_calls_puts}, no_data={debug_no_data}, "
          f"periods={len(period_prices)}")

    if not period_prices:
        return None, None, None

    # Stitch price series together at PFD boundaries (Algorithm 1, Line 12)
    # Sort PFDs to ensure chronological order
    sorted_pfds = sorted(period_prices.keys())

    if len(sorted_pfds) == 1:
        # Only one period, no stitching needed
        stitched_prices = period_prices[sorted_pfds[0]]
    else:
        # Start with the first period
        stitched_prices = period_prices[sorted_pfds[0]].copy()

        # Stitch each subsequent period
        for i in range(1, len(sorted_pfds)):
            roll_date = sorted_pfds[i]
            back_prices = period_prices[roll_date]

            # Use stitch_price_series to maintain continuity
            stitched_prices = stitch_price_series(
                front_prices=stitched_prices,
                back_prices=back_prices,
                roll_date=roll_date,
                debug=(i < 3 and asset_id == 'AAPL')  # Debug first 3 stitches for AAPL
            )

    # Convert to proper Series with sorted index
    prices = stitched_prices.sort_index()

    # Check coverage
    expected_days = len(pd.date_range(start=pfds[0], end=pfds[-1], freq='B'))
    actual_coverage = len(prices) / expected_days

    # DEBUG: Print coverage details
    print(f"      -> {asset_id} coverage: {len(prices)}/{expected_days} = {actual_coverage:.1%} "
          f"(need {min_coverage:.0%}), pfds[0]={pfds[0].date()}, pfds[-1]={pfds[-1].date()}")

    if actual_coverage < min_coverage:
        return None, None, None

    # Backfill missing with EMA (Algorithm 1, Line 18)
    prices = backfill_missing_ema(prices, span=5)

    # Compute returns (Algorithm 1, Line 19)
    returns = compute_returns(prices)

    # Build metadata dict if requested
    metadata = None
    if return_metadata and all_metadata:
        meta_df = pd.DataFrame(all_metadata).set_index('date')
        metadata = {
            'strike': meta_df['strike'],
            'exdate': meta_df['exdate'],
            'spot': meta_df['spot']
        }

    return prices, returns, metadata


def create_synthetic_options_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic options and equity data for testing.

    Generates realistic option prices and greeks based on simplified
    Black-Scholes assumptions with stochastic volatility.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        seed: Random seed for reproducibility

    Returns:
        Tuple of (options_df, equity_df)
    """
    np.random.seed(seed)

    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(trading_days)
    n_tickers = len(tickers)

    # Generate correlated equity returns
    # Create correlation matrix with sector structure
    base_corr = 0.4  # Higher base correlation (market factor)
    sector_corr = 0.7  # Higher within-sector correlation

    # Simple sector assignment (roughly equal groups)
    n_sectors = 6  # Fewer sectors = more tickers per sector
    sector_size = max(n_tickers // n_sectors, 1)

    corr_matrix = np.full((n_tickers, n_tickers), base_corr)
    for s in range(n_sectors):
        start_idx = s * sector_size
        end_idx = min((s + 1) * sector_size, n_tickers)
        corr_matrix[start_idx:end_idx, start_idx:end_idx] = sector_corr
    np.fill_diagonal(corr_matrix, 1.0)

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(corr_matrix)

    # Generate daily returns with varying volatility
    base_vol = 0.02  # ~32% annualized
    returns = np.zeros((n_days, n_tickers))
    vol_process = np.ones(n_tickers) * base_vol

    for t in range(n_days):
        # Update volatility (mean-reverting)
        vol_shock = np.random.normal(0, 0.001, n_tickers)
        vol_process = 0.98 * vol_process + 0.02 * base_vol + vol_shock
        vol_process = np.clip(vol_process, 0.005, 0.08)

        # Generate correlated returns
        z = np.random.normal(0, 1, n_tickers)
        corr_z = L @ z
        returns[t] = corr_z * vol_process

    # Convert to prices (starting at 100)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    equity_df = pd.DataFrame(prices, index=trading_days, columns=tickers)

    # Generate options data
    options_records = []

    pfds = get_portfolio_formation_days(start_date, end_date)

    for pfd in pfds:
        if pfd not in trading_days:
            continue

        # Find next month expiration (approximate as 30 days out)
        exdate = pfd + pd.Timedelta(days=30)
        # Adjust to Friday
        days_to_friday = (4 - exdate.weekday()) % 7
        exdate = exdate + pd.Timedelta(days=days_to_friday)

        pfd_idx = trading_days.get_loc(pfd)

        for ticker_idx, ticker in enumerate(tickers):
            spot = equity_df.loc[pfd, ticker]

            # Generate strikes around ATM
            strikes = spot * np.array([0.95, 0.975, 1.0, 1.025, 1.05])

            for strike in strikes:
                # Simplified Black-Scholes for option prices
                T = 30 / 252  # Time to expiry in years
                sigma = vol_process[ticker_idx] * np.sqrt(252)  # Annualized vol
                r = 0.02  # Risk-free rate

                d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)

                from scipy.stats import norm

                # Call price and delta
                call_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
                call_delta = norm.cdf(d1)

                # Put price and delta
                put_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
                put_delta = norm.cdf(d1) - 1

                # Add bid-ask spread (tighter for ATM)
                moneyness = spot / strike
                spread_pct = 0.02 + 0.03 * abs(moneyness - 1)

                # Generate daily option data from PFD to expiration
                option_days = trading_days[(trading_days >= pfd) & (trading_days <= exdate)]

                for day in option_days:
                    day_idx = trading_days.get_loc(day)
                    current_spot = equity_df.loc[day, ticker]
                    T_remaining = max((exdate - day).days / 252, 1/252)

                    # Recalculate prices
                    d1 = (np.log(current_spot / strike) + (r + 0.5 * sigma**2) * T_remaining) / (sigma * np.sqrt(T_remaining))
                    d2 = d1 - sigma * np.sqrt(T_remaining)

                    call_price = max(current_spot * norm.cdf(d1) - strike * np.exp(-r * T_remaining) * norm.cdf(d2), 0.01)
                    put_price = max(strike * np.exp(-r * T_remaining) * norm.cdf(-d2) - current_spot * norm.cdf(-d1), 0.01)

                    call_delta = norm.cdf(d1)
                    put_delta = norm.cdf(d1) - 1

                    # Call option record
                    options_records.append({
                        'date': day,
                        'ticker': ticker,
                        'exdate': exdate,
                        'strike': strike,
                        'cp_flag': 'C',
                        'best_bid': call_price * (1 - spread_pct/2),
                        'best_offer': call_price * (1 + spread_pct/2),
                        'delta': call_delta,
                        'impl_volatility': sigma,
                        'open_interest': np.random.randint(100, 10000)
                    })

                    # Put option record
                    options_records.append({
                        'date': day,
                        'ticker': ticker,
                        'exdate': exdate,
                        'strike': strike,
                        'cp_flag': 'P',
                        'best_bid': put_price * (1 - spread_pct/2),
                        'best_offer': put_price * (1 + spread_pct/2),
                        'delta': put_delta,
                        'impl_volatility': sigma,
                        'open_interest': np.random.randint(100, 10000)
                    })

    options_df = pd.DataFrame(options_records)

    return options_df, equity_df


def build_straddle_dataset(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    min_coverage: float = 0.95,
    return_metadata: bool = False,
    cfacpr_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Build complete straddle price and return matrices.

    Args:
        options_df: Options data with quotes and greeks
        equity_df: Underlying equity prices
        min_coverage: Minimum data coverage required
        return_metadata: If True, also return strike/expiry/spot DataFrames
        cfacpr_data: Optional DataFrame with cumulative price adjustment factors
                     (index=date, columns=tickers). Used to handle stock splits.

    Returns:
        Tuple of (prices_matrix, returns_matrix, metadata) with shape (delta, N)
        metadata is a dict with 'strike', 'exdate', 'spot' DataFrames if return_metadata=True
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

    for ticker in tickers:
        prices, returns, metadata = process_single_asset(
            options_df, equity_df, ticker, pfds, min_coverage,
            return_metadata=return_metadata,
            cfacpr_data=cfacpr_data
        )

        if prices is not None:
            all_prices[ticker] = prices
            all_returns[ticker] = returns
            if metadata is not None:
                all_strikes[ticker] = metadata['strike']
                all_exdates[ticker] = metadata['exdate']
                all_spots[ticker] = metadata['spot']

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


def build_straddle_dataset_chunked(
    options_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    min_coverage: float = 0.95,
    chunk_size: int = 10,
    verbose: bool = True,
    return_metadata: bool = False,
    cfacpr_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """
    Memory-efficient version of build_straddle_dataset.

    Processes tickers in chunks to reduce peak memory usage,
    with explicit garbage collection between chunks.

    Args:
        options_df: Options data with quotes and greeks
        equity_df: Underlying equity prices
        min_coverage: Minimum data coverage required
        chunk_size: Number of tickers to process per chunk
        verbose: Whether to print progress
        return_metadata: If True, also return strike/expiry/spot DataFrames
        cfacpr_data: Optional DataFrame with cumulative price adjustment factors
                     (index=date, columns=tickers). Used to handle stock splits.

    Returns:
        Tuple of (prices_matrix, returns_matrix, metadata) with shape (delta, N)
        metadata is a dict with 'strike', 'exdate', 'spot' DataFrames if return_metadata=True
    """
    import gc

    tickers = equity_df.columns.tolist()
    start_date = equity_df.index.min().strftime('%Y-%m-%d')
    end_date = equity_df.index.max().strftime('%Y-%m-%d')

    pfds = get_portfolio_formation_days(start_date, end_date)

    all_prices = {}
    all_returns = {}
    all_strikes = {}
    all_exdates = {}
    all_spots = {}
    processed = 0
    total = len(tickers)

    # Process in chunks
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk_tickers = tickers[chunk_start:chunk_end]

        if verbose:
            print(f"  Processing tickers {chunk_start+1}-{chunk_end} of {total}...")

        for ticker in chunk_tickers:
            prices, returns, metadata = process_single_asset(
                options_df, equity_df, ticker, pfds, min_coverage,
                return_metadata=return_metadata,
                cfacpr_data=cfacpr_data
            )

            if prices is not None:
                all_prices[ticker] = prices
                all_returns[ticker] = returns
                if metadata is not None:
                    all_strikes[ticker] = metadata['strike']
                    all_exdates[ticker] = metadata['exdate']
                    all_spots[ticker] = metadata['spot']
                processed += 1

        # Force garbage collection after each chunk
        gc.collect()

    if verbose:
        print(f"  Completed: {processed}/{total} tickers passed coverage filter")

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


def load_options_year_by_year(
    data_dir: str,
    start_year: int,
    end_year: int,
    required_columns: Optional[List[str]] = None,
    dtype_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load options data year by year with memory optimization.

    Loads each year's parquet file, converts to efficient dtypes,
    and concatenates with explicit cleanup between years.

    Args:
        data_dir: Directory containing year-split parquet files
        start_year: First year to load
        end_year: Last year to load (inclusive)
        required_columns: Columns to load (None = all)
        dtype_map: Dict mapping column names to dtypes for conversion

    Returns:
        Concatenated DataFrame with all years
    """
    import gc
    from pathlib import Path
    import pyarrow.parquet as pq

    if dtype_map is None:
        dtype_map = {
            'best_bid': 'float32',
            'best_offer': 'float32',
            'delta': 'float32',
            'spot_price': 'float32',
            'strike_price': 'float32',
            'strike': 'float32',
        }

    data_path = Path(data_dir)
    all_dfs = []

    for year in range(start_year, end_year + 1):
        # Try different naming conventions
        for pattern in [f'options_filtered_{year}.parquet',
                        f'options_{year}.parquet']:
            fpath = data_path / pattern
            if fpath.exists():
                break
        else:
            continue  # No file found for this year

        # Load with PyArrow for efficiency
        table = pq.read_table(fpath, columns=required_columns)
        df = table.to_pandas()
        del table

        # Convert dtypes
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        all_dfs.append(df)
        gc.collect()

    if not all_dfs:
        raise ValueError(f"No parquet files found in {data_dir}")

    # Concatenate all years
    result = pd.concat(all_dfs, ignore_index=True, copy=False)
    del all_dfs
    gc.collect()

    return result
