"""
Feature Engineering Module for Options Momentum Trading

This module implements the feature computation pipeline as described in
Section 4.3.1 of the paper. Features include:
- Volatility-normalized returns at multiple horizons
- MACD momentum indicators
- Option-specific features (moneyness, time-to-expiry)
- Winsorization using exponentially weighted statistics
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    min_periods: int = 5,
    annualize: bool = False
) -> pd.DataFrame:
    """
    Compute rolling realized volatility from returns.

    Args:
        returns: DataFrame of returns
        window: Rolling window size
        min_periods: Minimum periods for valid calculation
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        DataFrame of volatility estimates
    """
    vol = returns.rolling(window=window, min_periods=min_periods).std()

    if annualize:
        vol = vol * np.sqrt(252)

    return vol


def compute_volatility_normalized_returns(
    returns: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    vol_window: int = 20
) -> dict:
    """
    Compute volatility-normalized returns at multiple horizons.

    r^{(i,V)}_{t-k,t} / (sigma^{(i,V)}_t * sqrt(k))

    Args:
        returns: DataFrame of daily returns
        horizons: List of return horizons (lookback periods). Default: [1, 5, 10, 15, 20]
        vol_window: Window for volatility estimation

    Returns:
        Dictionary mapping horizon -> normalized returns DataFrame
    """
    if horizons is None:
        horizons = [1, 5, 10, 15, 20]

    # Compute daily volatility
    daily_vol = compute_realized_volatility(returns, window=vol_window)

    normalized_returns = {}

    for k in horizons:
        # Compute k-period returns (cumulative)
        if k == 1:
            k_returns = returns
        else:
            k_returns = returns.rolling(window=k).apply(
                lambda x: (1 + x).prod() - 1,
                raw=True
            )

        # Normalize by volatility scaled by sqrt(k)
        vol_scaled = daily_vol * np.sqrt(k)
        vol_scaled = vol_scaled.replace(0, np.nan)  # Avoid division by zero

        normalized = k_returns / vol_scaled
        normalized_returns[f'ret_norm_{k}d'] = normalized

    return normalized_returns


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

    # Compute daily volatility
    daily_vol = compute_realized_volatility(returns, window=vol_window)

    normalized_macd = {}

    for short_span, long_span in macd_params:
        macd = compute_macd(prices, short_span, long_span)

        # Normalize by price * volatility
        price_vol = prices * daily_vol
        price_vol = price_vol.replace(0, np.nan)

        normalized = macd / price_vol
        normalized_macd[f'macd_{short_span}_{long_span}'] = normalized

    return normalized_macd


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
    moneyness = moneyness.clip(lower=1e-6)  # Avoid log(0)

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


def winsorize_ewm(
    data: pd.DataFrame,
    half_life: int = 252,
    n_std: float = 5.0
) -> pd.DataFrame:
    """
    Winsorize data using exponentially weighted mean and std.

    Applies floor/cap at mu_EWM +/- n_std * sigma_EWM.

    Args:
        data: Input DataFrame
        half_life: Half-life for EWM calculation (in days)
        n_std: Number of standard deviations for floor/cap

    Returns:
        Winsorized DataFrame
    """
    # Compute EWM mean and std
    ewm_mean = data.ewm(halflife=half_life, min_periods=1).mean()
    ewm_std = data.ewm(halflife=half_life, min_periods=1).std()

    # Compute bounds
    lower = ewm_mean - n_std * ewm_std
    upper = ewm_mean + n_std * ewm_std

    # Clip values
    winsorized = data.clip(lower=lower, upper=upper)

    return winsorized


def build_feature_tensor(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    spot_prices: Optional[pd.DataFrame] = None,
    strike_prices: Optional[pd.DataFrame] = None,
    expiry_dates: Optional[pd.DataFrame] = None,
    return_horizons: Optional[List[int]] = None,
    macd_params: Optional[List[Tuple[int, int]]] = None,
    vol_window: int = 20,
    winsorize: bool = True,
    half_life: int = 252,
    n_std: float = 5.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the complete feature tensor X of shape (delta, F, N).

    Features (F=10 by default):
    - 5 volatility-normalized returns at different horizons
    - 3 volatility-normalized MACD indicators
    - Log-moneyness (if spot/strike provided)
    - Time-to-expiry (if expiry dates provided)

    Args:
        prices: DataFrame of straddle prices (delta x N)
        returns: DataFrame of straddle returns (delta x N)
        spot_prices: DataFrame of underlying prices (optional)
        strike_prices: DataFrame of strike prices (optional)
        expiry_dates: DataFrame of expiration dates (optional)
        return_horizons: List of return horizons
        macd_params: List of MACD parameter tuples
        vol_window: Window for volatility estimation
        winsorize: Whether to apply EWM winsorization
        half_life: Half-life for winsorization
        n_std: Number of std devs for winsorization bounds

    Returns:
        Tuple of (feature_tensor, feature_names)
    """
    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    features = {}
    feature_names = []

    # Volatility-normalized returns
    norm_returns = compute_volatility_normalized_returns(
        returns, return_horizons, vol_window
    )
    for name, feat in norm_returns.items():
        features[name] = feat
        feature_names.append(name)

    # Volatility-normalized MACD
    norm_macd = compute_volatility_normalized_macd(
        prices, returns, macd_params, vol_window
    )
    for name, feat in norm_macd.items():
        features[name] = feat
        feature_names.append(name)

    # Log-moneyness (if data available)
    if spot_prices is not None and strike_prices is not None:
        log_m = compute_log_moneyness(spot_prices, strike_prices)
        features['log_moneyness'] = log_m
        feature_names.append('log_moneyness')

    # Time-to-expiry (if data available)
    if expiry_dates is not None:
        tte = compute_time_to_expiry(prices.index, expiry_dates)
        features['time_to_expiry'] = tte
        feature_names.append('time_to_expiry')

    # Apply winsorization
    if winsorize:
        for name in features:
            features[name] = winsorize_ewm(features[name], half_life, n_std)

    # Align indices
    common_index = prices.index
    common_columns = prices.columns

    # Stack into tensor (delta, F, N)
    delta = len(common_index)
    F = len(feature_names)
    N = len(common_columns)

    tensor = np.zeros((delta, F, N))

    for f_idx, name in enumerate(feature_names):
        feat_df = features[name].reindex(index=common_index, columns=common_columns)
        tensor[:, f_idx, :] = feat_df.values

    return tensor, feature_names


def create_synthetic_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    return_horizons: Optional[List[int]] = None,
    macd_params: Optional[List[Tuple[int, int]]] = None,
    include_moneyness: bool = True,
    include_tte: bool = True
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Create synthetic feature data for testing when full option data unavailable.

    Generates approximate moneyness and time-to-expiry features using
    simple assumptions.

    Args:
        prices: DataFrame of straddle prices
        returns: DataFrame of straddle returns
        return_horizons: Return horizon list. Default: [1, 5, 10, 15, 20]
        macd_params: MACD parameter list. Default: [(2, 8), (4, 16), (8, 32)]
        include_moneyness: Whether to include synthetic moneyness
        include_tte: Whether to include synthetic time-to-expiry

    Returns:
        Tuple of (feature_tensor, feature_names, feature_dataframe)
    """
    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    features = {}
    feature_names = []

    # Volatility-normalized returns
    norm_returns = compute_volatility_normalized_returns(
        returns, return_horizons, vol_window=20
    )
    for name, feat in norm_returns.items():
        features[name] = feat
        feature_names.append(name)

    # Volatility-normalized MACD
    norm_macd = compute_volatility_normalized_macd(
        prices, returns, macd_params, vol_window=20
    )
    for name, feat in norm_macd.items():
        features[name] = feat
        feature_names.append(name)

    # Synthetic log-moneyness (assume ATM, small random perturbations)
    if include_moneyness:
        np.random.seed(42)
        log_m = pd.DataFrame(
            np.random.normal(0, 0.02, size=prices.shape),
            index=prices.index,
            columns=prices.columns
        )
        features['log_moneyness'] = log_m
        feature_names.append('log_moneyness')

    # Synthetic time-to-expiry (cycles monthly from ~0.08 to 0.003)
    if include_tte:
        n_days = len(prices)
        tte_cycle = np.linspace(30/252, 1/252, 21)  # 21 trading days per month
        full_tte = np.tile(tte_cycle, n_days // 21 + 1)[:n_days]

        tte_df = pd.DataFrame(
            np.tile(full_tte.reshape(-1, 1), (1, len(prices.columns))),
            index=prices.index,
            columns=prices.columns
        )
        features['time_to_expiry'] = tte_df
        feature_names.append('time_to_expiry')

    # Winsorize
    for name in features:
        features[name] = winsorize_ewm(features[name], half_life=252, n_std=5.0)

    # Build tensor
    delta = len(prices)
    F = len(feature_names)
    N = len(prices.columns)

    tensor = np.zeros((delta, F, N))

    for f_idx, name in enumerate(feature_names):
        tensor[:, f_idx, :] = features[name].values

    # Also return as DataFrame for analysis
    feature_df = pd.concat(features.values(), axis=1, keys=features.keys())

    return tensor, feature_names, feature_df


def create_synthetic_features_efficient(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    return_horizons: Optional[List[int]] = None,
    macd_params: Optional[List[Tuple[int, int]]] = None,
    include_moneyness: bool = True,
    include_tte: bool = True,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, List[str]]:
    """
    Memory-efficient version of create_synthetic_features.

    Builds the feature tensor incrementally, freeing intermediate DataFrames
    immediately after use. Uses float32 by default to halve memory usage.

    Args:
        prices: DataFrame of straddle prices
        returns: DataFrame of straddle returns
        return_horizons: Return horizon list. Default: [1, 5, 10, 15, 20]
        macd_params: MACD parameter list. Default: [(2, 8), (4, 16), (8, 32)]
        include_moneyness: Whether to include synthetic moneyness
        include_tte: Whether to include synthetic time-to-expiry
        dtype: NumPy dtype for tensor (default: float32 for memory efficiency)

    Returns:
        Tuple of (feature_tensor, feature_names) - does NOT return feature_df
    """
    import gc

    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    # Calculate dimensions
    delta = len(prices)
    N = len(prices.columns)
    F = len(return_horizons) + len(macd_params)
    if include_moneyness:
        F += 1
    if include_tte:
        F += 1

    # Pre-allocate tensor with specified dtype
    tensor = np.zeros((delta, F, N), dtype=dtype)
    feature_names = []
    f_idx = 0

    # Compute daily volatility once (reused for all features)
    daily_vol = compute_realized_volatility(returns, window=20)

    # --- Volatility-normalized returns (computed one at a time) ---
    for k in return_horizons:
        # Compute k-period returns
        if k == 1:
            k_returns = returns.copy()
        else:
            k_returns = returns.rolling(window=k).apply(
                lambda x: (1 + x).prod() - 1,
                raw=True
            )

        # Normalize by volatility scaled by sqrt(k)
        vol_scaled = daily_vol * np.sqrt(k)
        vol_scaled = vol_scaled.replace(0, np.nan)
        normalized = k_returns / vol_scaled

        # Winsorize
        normalized = winsorize_ewm(normalized, half_life=252, n_std=5.0)

        # Store in tensor and free memory
        tensor[:, f_idx, :] = normalized.values.astype(dtype)
        feature_names.append(f'ret_norm_{k}d')
        f_idx += 1

        # Explicit cleanup
        del k_returns, normalized
        gc.collect()

    # --- Volatility-normalized MACD (computed one at a time) ---
    for short_span, long_span in macd_params:
        # Compute MACD
        ema_short = prices.ewm(span=short_span, adjust=False).mean()
        ema_long = prices.ewm(span=long_span, adjust=False).mean()
        macd = ema_short - ema_long
        del ema_short, ema_long

        # Normalize by price * volatility
        price_vol = prices * daily_vol
        price_vol = price_vol.replace(0, np.nan)
        normalized = macd / price_vol
        del macd, price_vol

        # Winsorize
        normalized = winsorize_ewm(normalized, half_life=252, n_std=5.0)

        # Store in tensor and free memory
        tensor[:, f_idx, :] = normalized.values.astype(dtype)
        feature_names.append(f'macd_{short_span}_{long_span}')
        f_idx += 1

        del normalized
        gc.collect()

    # Free daily_vol now that we're done with it
    del daily_vol
    gc.collect()

    # --- Synthetic log-moneyness ---
    if include_moneyness:
        np.random.seed(42)
        log_m = np.random.normal(0, 0.02, size=(delta, N)).astype(dtype)
        # Apply simple winsorization inline
        log_m = np.clip(log_m, -0.1, 0.1)
        tensor[:, f_idx, :] = log_m
        feature_names.append('log_moneyness')
        f_idx += 1
        del log_m

    # --- Synthetic time-to-expiry ---
    if include_tte:
        tte_cycle = np.linspace(30/252, 1/252, 21).astype(dtype)
        full_tte = np.tile(tte_cycle, delta // 21 + 1)[:delta]
        tte_matrix = np.tile(full_tte.reshape(-1, 1), (1, N))
        tensor[:, f_idx, :] = tte_matrix
        feature_names.append('time_to_expiry')
        f_idx += 1
        del tte_cycle, full_tte, tte_matrix

    gc.collect()
    return tensor, feature_names


def create_features_with_real_metadata(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    metadata: dict,
    return_horizons: Optional[List[int]] = None,
    macd_params: Optional[List[Tuple[int, int]]] = None,
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, List[str]]:
    """
    Create feature tensor using real option metadata (strike, expiry, spot).

    Computes actual log-moneyness and time-to-expiry from the provided metadata
    instead of synthetic approximations.

    Args:
        prices: DataFrame of straddle prices (delta x N)
        returns: DataFrame of straddle returns (delta x N)
        metadata: Dict with 'strike', 'exdate', 'spot' DataFrames
        return_horizons: Return horizon list. Default: [1, 5, 10, 15, 20]
        macd_params: MACD parameter list. Default: [(2, 8), (4, 16), (8, 32)]
        dtype: NumPy dtype for tensor (default: float32)

    Returns:
        Tuple of (feature_tensor, feature_names)
    """
    import gc

    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    # Calculate dimensions
    delta = len(prices)
    N = len(prices.columns)
    F = len(return_horizons) + len(macd_params) + 2  # +2 for moneyness and TTE

    # Pre-allocate tensor
    tensor = np.zeros((delta, F, N), dtype=dtype)
    feature_names = []
    f_idx = 0

    # Compute daily volatility once
    daily_vol = compute_realized_volatility(returns, window=20)

    # --- Volatility-normalized returns ---
    for k in return_horizons:
        if k == 1:
            k_returns = returns.copy()
        else:
            k_returns = returns.rolling(window=k).apply(
                lambda x: (1 + x).prod() - 1,
                raw=True
            )

        vol_scaled = daily_vol * np.sqrt(k)
        vol_scaled = vol_scaled.replace(0, np.nan)
        normalized = k_returns / vol_scaled
        normalized = winsorize_ewm(normalized, half_life=252, n_std=5.0)

        tensor[:, f_idx, :] = normalized.reindex(
            index=prices.index, columns=prices.columns
        ).values.astype(dtype)
        feature_names.append(f'ret_norm_{k}d')
        f_idx += 1

        del k_returns, normalized
        gc.collect()

    # --- Volatility-normalized MACD ---
    for short_span, long_span in macd_params:
        ema_short = prices.ewm(span=short_span, adjust=False).mean()
        ema_long = prices.ewm(span=long_span, adjust=False).mean()
        macd = ema_short - ema_long
        del ema_short, ema_long

        price_vol = prices * daily_vol
        price_vol = price_vol.replace(0, np.nan)
        normalized = macd / price_vol
        del macd, price_vol

        normalized = winsorize_ewm(normalized, half_life=252, n_std=5.0)

        tensor[:, f_idx, :] = normalized.reindex(
            index=prices.index, columns=prices.columns
        ).values.astype(dtype)
        feature_names.append(f'macd_{short_span}_{long_span}')
        f_idx += 1

        del normalized
        gc.collect()

    del daily_vol
    gc.collect()

    # --- Real log-moneyness: log(S/K) ---
    spot_df = metadata['spot'].reindex(index=prices.index, columns=prices.columns)
    strike_df = metadata['strike'].reindex(index=prices.index, columns=prices.columns)

    # Compute log-moneyness
    moneyness = spot_df / strike_df
    moneyness = moneyness.replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
    moneyness = moneyness.clip(lower=1e-6)
    log_moneyness = np.log(moneyness)

    # Winsorize
    log_moneyness = winsorize_ewm(log_moneyness, half_life=252, n_std=5.0)

    tensor[:, f_idx, :] = log_moneyness.values.astype(dtype)
    feature_names.append('log_moneyness')
    f_idx += 1

    del moneyness, log_moneyness
    gc.collect()

    # --- Real time-to-expiry (in years) ---
    exdate_df = metadata['exdate'].reindex(index=prices.index, columns=prices.columns)

    # Compute TTE: (exdate - current_date) / 252
    tte = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        for idx in prices.index:
            if pd.notna(exdate_df.loc[idx, col]):
                days_to_expiry = (exdate_df.loc[idx, col] - idx).days
                tte.loc[idx, col] = max(days_to_expiry, 0) / 252.0
            else:
                tte.loc[idx, col] = np.nan

    # Forward fill missing TTE values
    tte = tte.ffill().bfill()

    tensor[:, f_idx, :] = tte.values.astype(dtype)
    feature_names.append('time_to_expiry')
    f_idx += 1

    del tte, exdate_df, spot_df, strike_df
    gc.collect()

    return tensor, feature_names


def compute_feature_statistics(
    feature_tensor: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute summary statistics for each feature.

    Args:
        feature_tensor: Feature tensor of shape (delta, F, N)
        feature_names: List of feature names

    Returns:
        DataFrame with statistics for each feature
    """
    stats = []

    for f_idx, name in enumerate(feature_names):
        feat_data = feature_tensor[:, f_idx, :].flatten()
        feat_data = feat_data[~np.isnan(feat_data)]

        stats.append({
            'feature': name,
            'mean': np.mean(feat_data),
            'median': np.median(feat_data),
            'std': np.std(feat_data),
            'skewness': pd.Series(feat_data).skew(),
            'kurtosis': pd.Series(feat_data).kurtosis(),
            'min': np.min(feat_data),
            'max': np.max(feat_data),
            'pct_nan': np.isnan(feature_tensor[:, f_idx, :]).mean() * 100
        })

    return pd.DataFrame(stats).set_index('feature')
