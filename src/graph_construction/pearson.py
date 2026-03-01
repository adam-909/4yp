"""
Pearson correlation-based graph construction (Section 5.2.1).
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log-returns from price series.

    Args:
        prices: DataFrame of prices (dates x assets)

    Returns:
        DataFrame of log-returns
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute Pearson correlation matrix from returns.

    Equation (40):
    ρ_ij = Σ(r_i - r̄_i)(r_j - r̄_j) / sqrt(Σ(r_i - r̄_i)² · Σ(r_j - r̄_j)²)

    Args:
        returns: DataFrame of returns (dates x assets)

    Returns:
        Correlation matrix (N x N)
    """
    corr = returns.corr().values
    # Handle NaN values
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def build_pearson_adjacency(
    corr_matrix: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Build adjacency matrix from correlation using threshold.

    Equation (41):
    A_ij = τ if |ρ_ij| ≥ τ and i ≠ j, else 0

    Args:
        corr_matrix: Pearson correlation matrix (N x N)
        threshold: Correlation threshold τ

    Returns:
        Adjacency matrix (N x N)
    """
    N = corr_matrix.shape[0]
    A = np.zeros((N, N))

    # Apply threshold
    mask = np.abs(corr_matrix) >= threshold
    A[mask] = threshold

    # Zero out diagonal
    np.fill_diagonal(A, 0)

    return A


def sweep_pearson_thresholds(
    corr_matrix: np.ndarray,
    thresholds: Optional[List[float]] = None,
    sector_labels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Sweep correlation thresholds and compute metrics for each.

    Args:
        corr_matrix: Pearson correlation matrix
        thresholds: List of thresholds to try
        sector_labels: Optional sector labels for homophily

    Returns:
        DataFrame with metrics for each threshold
    """
    from .metrics import compute_connectivity, compute_edge_homophily, compute_louvain_modularity

    if thresholds is None:
        thresholds = np.arange(0.3, 0.75, 0.05)

    results = []
    for tau in thresholds:
        A = build_pearson_adjacency(corr_matrix, tau)

        metrics = {
            'threshold': tau,
            'connectivity': compute_connectivity(A),
            'num_edges': int(np.sum(A > 0) / 2),
            'density': np.sum(A > 0) / (A.shape[0] * (A.shape[0] - 1))
        }

        if sector_labels is not None:
            metrics['homophily'] = compute_edge_homophily(A, sector_labels)

        # Compute modularity (requires community detection)
        try:
            modularity, _ = compute_louvain_modularity(A)
            metrics['modularity'] = modularity
        except Exception:
            metrics['modularity'] = np.nan

        results.append(metrics)

    return pd.DataFrame(results)
