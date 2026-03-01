"""
Feature tensor building functions.
"""

import gc
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from .volatility import compute_realized_volatility, compute_volatility_normalized_returns
from .technical import compute_volatility_normalized_macd
from .options_features import compute_log_moneyness, compute_time_to_expiry
from .preprocessing import winsorize_ewm


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

    # Synthetic log-moneyness
    if include_moneyness:
        np.random.seed(42)
        log_m = pd.DataFrame(
            np.random.normal(0, 0.02, size=prices.shape),
            index=prices.index,
            columns=prices.columns
        )
        features['log_moneyness'] = log_m
        feature_names.append('log_moneyness')

    # Synthetic time-to-expiry
    if include_tte:
        n_days = len(prices)
        tte_cycle = np.linspace(30/252, 1/252, 21)
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

    Args:
        prices: DataFrame of straddle prices
        returns: DataFrame of straddle returns
        return_horizons: Return horizon list. Default: [1, 5, 10, 15, 20]
        macd_params: MACD parameter list. Default: [(2, 8), (4, 16), (8, 32)]
        include_moneyness: Whether to include synthetic moneyness
        include_tte: Whether to include synthetic time-to-expiry
        dtype: NumPy dtype for tensor (default: float32)

    Returns:
        Tuple of (feature_tensor, feature_names)
    """
    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    delta = len(prices)
    N = len(prices.columns)
    F = len(return_horizons) + len(macd_params)
    if include_moneyness:
        F += 1
    if include_tte:
        F += 1

    tensor = np.zeros((delta, F, N), dtype=dtype)
    feature_names = []
    f_idx = 0

    daily_vol = compute_realized_volatility(returns, window=20)

    # Volatility-normalized returns
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

        tensor[:, f_idx, :] = normalized.values.astype(dtype)
        feature_names.append(f'ret_norm_{k}d')
        f_idx += 1

        del k_returns, normalized
        gc.collect()

    # Volatility-normalized MACD
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

        tensor[:, f_idx, :] = normalized.values.astype(dtype)
        feature_names.append(f'macd_{short_span}_{long_span}')
        f_idx += 1

        del normalized
        gc.collect()

    del daily_vol
    gc.collect()

    # Synthetic log-moneyness
    if include_moneyness:
        np.random.seed(42)
        log_m = np.random.normal(0, 0.02, size=(delta, N)).astype(dtype)
        log_m = np.clip(log_m, -0.1, 0.1)
        tensor[:, f_idx, :] = log_m
        feature_names.append('log_moneyness')
        f_idx += 1
        del log_m

    # Synthetic time-to-expiry
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
    if return_horizons is None:
        return_horizons = [1, 5, 10, 15, 20]
    if macd_params is None:
        macd_params = [(2, 8), (4, 16), (8, 32)]

    delta = len(prices)
    N = len(prices.columns)
    F = len(return_horizons) + len(macd_params) + 2

    tensor = np.zeros((delta, F, N), dtype=dtype)
    feature_names = []
    f_idx = 0

    daily_vol = compute_realized_volatility(returns, window=20)

    # Volatility-normalized returns
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

    # Volatility-normalized MACD
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

    # Real log-moneyness
    spot_df = metadata['spot'].reindex(index=prices.index, columns=prices.columns)
    strike_df = metadata['strike'].reindex(index=prices.index, columns=prices.columns)

    moneyness = spot_df / strike_df
    moneyness = moneyness.replace(0, np.nan).replace([np.inf, -np.inf], np.nan)
    moneyness = moneyness.clip(lower=1e-6)
    log_moneyness = np.log(moneyness)
    log_moneyness = winsorize_ewm(log_moneyness, half_life=252, n_std=5.0)

    tensor[:, f_idx, :] = log_moneyness.values.astype(dtype)
    feature_names.append('log_moneyness')
    f_idx += 1

    del moneyness, log_moneyness
    gc.collect()

    # Real time-to-expiry
    exdate_df = metadata['exdate'].reindex(index=prices.index, columns=prices.columns)

    tte = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        for idx in prices.index:
            if pd.notna(exdate_df.loc[idx, col]):
                days_to_expiry = (exdate_df.loc[idx, col] - idx).days
                tte.loc[idx, col] = max(days_to_expiry, 0) / 252.0
            else:
                tte.loc[idx, col] = np.nan

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
