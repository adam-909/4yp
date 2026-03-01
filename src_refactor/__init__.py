"""
Refactored Algorithm 1 Implementation

This module contains the corrected straddle pipeline with fixes for:
1. Missing-day handling (reindex to trading calendar)
2. Correct trading calendar (use equity index, not freq='B')
3. No split adjustment on option prices
4. Flexible PFD date matching (±3 day lookahead)
5. Delta required only at PFD, not globally
"""

from .straddle import (
    process_single_asset_v2,
    build_straddle_dataset_v2,
    get_portfolio_formation_days,
)

from .utils import (
    compute_delta_neutral_weights,
    select_atm_strike,
    stitch_price_series,
    backfill_missing_ema,
)

__all__ = [
    'process_single_asset_v2',
    'build_straddle_dataset_v2',
    'get_portfolio_formation_days',
    'compute_delta_neutral_weights',
    'select_atm_strike',
    'stitch_price_series',
    'backfill_missing_ema',
]
