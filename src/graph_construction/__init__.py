"""
Graph Construction Package for LSTM-GCN Options Trading

This package implements graph construction methods as described in Section 5 of the paper:
- Pearson correlation-based adjacency matrix (Section 5.2.1)
- Convex optimization-based adjacency matrix (Section 5.2.2)
- Graph evaluation metrics (Section 5.3)
"""

from .pearson import (
    compute_log_returns,
    compute_correlation_matrix,
    build_pearson_adjacency,
    sweep_pearson_thresholds,
)

from .convex import (
    learn_adjacency_convex_numpy,
    learn_adjacency_convex_torch,
    learn_adjacency_convex,
    grid_search_convex,
    build_graph_ensemble,
)

from .normalization import (
    normalize_adjacency,
)

from .metrics import (
    compute_connectivity,
    compute_edge_homophily,
    compute_louvain_modularity,
    compute_graph_metrics,
)

from .utils import (
    reshape_feature_tensor_for_graph,
    create_sector_labels,
)

__all__ = [
    # Pearson correlation
    'compute_log_returns',
    'compute_correlation_matrix',
    'build_pearson_adjacency',
    'sweep_pearson_thresholds',
    # Convex optimization
    'learn_adjacency_convex_numpy',
    'learn_adjacency_convex_torch',
    'learn_adjacency_convex',
    'grid_search_convex',
    'build_graph_ensemble',
    # Normalization
    'normalize_adjacency',
    # Metrics
    'compute_connectivity',
    'compute_edge_homophily',
    'compute_louvain_modularity',
    'compute_graph_metrics',
    # Utils
    'reshape_feature_tensor_for_graph',
    'create_sector_labels',
]
