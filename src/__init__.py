"""
Options Momentum Trading - Data Processing Package

This package provides utilities for processing options data and building
features for geometric deep learning momentum trading strategies.
"""

from .data_processing import (
    get_portfolio_formation_days,
    compute_delta_neutral_weights,
    compute_straddle_price,
    select_atm_strike,
    filter_by_moneyness,
    stitch_price_series,
    backfill_missing_ema,
    compute_returns,
    process_single_asset,
    create_synthetic_options_data,
    build_straddle_dataset,
    # Split adjustment
    compute_split_adjustment_factors,
    apply_split_adjustment,
    get_split_dates,
    load_cfacpr_data,
)

from .features import (
    compute_realized_volatility,
    compute_volatility_normalized_returns,
    compute_macd,
    compute_volatility_normalized_macd,
    compute_log_moneyness,
    compute_time_to_expiry,
    winsorize_ewm,
    build_feature_tensor,
    create_synthetic_features,
    compute_feature_statistics,
)

from .visualization import (
    plot_straddle_prices,
    plot_price_vs_vix,
    plot_return_distributions,
    plot_sector_returns,
    plot_sector_pie_chart,
    plot_return_statistics_table,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_cumulative_returns,
    create_summary_dashboard,
)

from .graph_construction import (
    # Pearson correlation method
    compute_log_returns,
    compute_correlation_matrix,
    build_pearson_adjacency,
    sweep_pearson_thresholds,
    # Convex optimization method
    learn_adjacency_convex,
    learn_adjacency_convex_numpy,
    learn_adjacency_convex_torch,
    grid_search_convex,
    build_graph_ensemble,
    normalize_adjacency,
    # Evaluation metrics
    compute_connectivity,
    compute_edge_homophily,
    compute_louvain_modularity,
    compute_graph_metrics,
    # Utilities
    reshape_feature_tensor_for_graph,
    create_sector_labels,
)

__all__ = [
    # Data processing
    'get_portfolio_formation_days',
    'compute_delta_neutral_weights',
    'compute_straddle_price',
    'select_atm_strike',
    'filter_by_moneyness',
    'stitch_price_series',
    'backfill_missing_ema',
    'compute_returns',
    'process_single_asset',
    'create_synthetic_options_data',
    'build_straddle_dataset',
    # Split adjustment
    'compute_split_adjustment_factors',
    'apply_split_adjustment',
    'get_split_dates',
    'load_cfacpr_data',
    # Features
    'compute_realized_volatility',
    'compute_volatility_normalized_returns',
    'compute_macd',
    'compute_volatility_normalized_macd',
    'compute_log_moneyness',
    'compute_time_to_expiry',
    'winsorize_ewm',
    'build_feature_tensor',
    'create_synthetic_features',
    'compute_feature_statistics',
    # Visualization
    'plot_straddle_prices',
    'plot_price_vs_vix',
    'plot_return_distributions',
    'plot_sector_returns',
    'plot_sector_pie_chart',
    'plot_return_statistics_table',
    'plot_correlation_heatmap',
    'plot_feature_distributions',
    'plot_cumulative_returns',
    'create_summary_dashboard',
    # Graph construction
    'compute_log_returns',
    'compute_correlation_matrix',
    'build_pearson_adjacency',
    'sweep_pearson_thresholds',
    'learn_adjacency_convex',
    'learn_adjacency_convex_numpy',
    'learn_adjacency_convex_torch',
    'grid_search_convex',
    'build_graph_ensemble',
    'normalize_adjacency',
    'compute_connectivity',
    'compute_edge_homophily',
    'compute_louvain_modularity',
    'compute_graph_metrics',
    'reshape_feature_tensor_for_graph',
    'create_sector_labels',
]
