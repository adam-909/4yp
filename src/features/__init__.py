"""
Feature Engineering Package for Options Momentum Trading

This package implements the feature computation pipeline as described in
Section 4.3.1 of the paper.
"""

from .volatility import (
    compute_realized_volatility,
    compute_volatility_normalized_returns,
)

from .technical import (
    compute_ema,
    compute_macd,
    compute_volatility_normalized_macd,
)

from .options_features import (
    compute_log_moneyness,
    compute_time_to_expiry,
)

from .preprocessing import (
    winsorize_ewm,
)

from .tensor import (
    build_feature_tensor,
    create_synthetic_features,
    create_synthetic_features_efficient,
    create_features_with_real_metadata,
    compute_feature_statistics,
)

__all__ = [
    # Volatility
    'compute_realized_volatility',
    'compute_volatility_normalized_returns',
    # Technical
    'compute_ema',
    'compute_macd',
    'compute_volatility_normalized_macd',
    # Options features
    'compute_log_moneyness',
    'compute_time_to_expiry',
    # Preprocessing
    'winsorize_ewm',
    # Tensor building
    'build_feature_tensor',
    'create_synthetic_features',
    'create_synthetic_features_efficient',
    'create_features_with_real_metadata',
    'compute_feature_statistics',
]
