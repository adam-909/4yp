"""
Data Processing Package for Options Momentum Trading

This package implements the straddle formation and data processing pipeline
as described in Section 4 of the paper "A Geometric Deep Learning Approach
to Momentum Options Trading".
"""

from .splits import (
    compute_split_adjustment_factors,
    apply_split_adjustment,
    get_split_dates,
    load_cfacpr_data,
    get_cfacpr_for_date,
    compute_split_adjusted_strike,
)

from .portfolio import (
    get_portfolio_formation_days,
    compute_delta_neutral_weights,
    compute_straddle_price,
)

from .options import (
    select_atm_strike,
    filter_by_moneyness,
)

from .straddle import (
    stitch_price_series,
    backfill_missing_ema,
    compute_returns,
    process_single_asset,
)

from .synthetic import (
    create_synthetic_options_data,
)

from .dataset import (
    build_straddle_dataset,
    build_straddle_dataset_chunked,
    load_options_year_by_year,
)

__all__ = [
    # Splits
    'compute_split_adjustment_factors',
    'apply_split_adjustment',
    'get_split_dates',
    'load_cfacpr_data',
    'get_cfacpr_for_date',
    'compute_split_adjusted_strike',
    # Portfolio
    'get_portfolio_formation_days',
    'compute_delta_neutral_weights',
    'compute_straddle_price',
    # Options
    'select_atm_strike',
    'filter_by_moneyness',
    # Straddle
    'stitch_price_series',
    'backfill_missing_ema',
    'compute_returns',
    'process_single_asset',
    # Synthetic
    'create_synthetic_options_data',
    # Dataset
    'build_straddle_dataset',
    'build_straddle_dataset_chunked',
    'load_options_year_by_year',
]
