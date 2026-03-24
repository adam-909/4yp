"""
Pearson correlation graph computation for static LSTM-GCN.

This module provides functions to compute adjacency matrices from:
- Straddle returns (matching rolling Pearson implementation)
- Equity log returns (matching original precomputed pearson CSVs but with training data only)
"""

import numpy as np
import pandas as pd
import os


def compute_equity_pearson_adjacency(
    threshold,
    train_end_year=None,
    normalize=True,
):
    """
    Compute Pearson correlation adjacency matrix from equity log returns.

    Uses cached equity returns from data/graph_structure/equity_returns/log_returns.csv.
    If cache doesn't exist, downloads from yfinance.

    Args:
        threshold: Correlation threshold (tau) for creating edges
        train_end_year: If provided, only use data before this year (avoids look-ahead bias)
        normalize: If True, apply symmetric normalization (D^{-1/2} A D^{-1/2})

    Returns:
        adjacency: Normalized adjacency matrix of shape (num_tickers, num_tickers)
        tickers: List of tickers in order
    """
    cache_path = os.path.join("data", "graph_structure", "equity_returns", "log_returns.csv")

    # Load equity returns
    if os.path.exists(cache_path):
        print(f"Loading equity returns from cache: {cache_path}")
        log_returns_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            f"Equity returns cache not found at {cache_path}. "
            "Run 'python examples/create_equity_returns_cache.py' to create it."
        )

    log_returns_df = log_returns_df.sort_index()
    tickers = sorted(log_returns_df.columns.tolist())
    num_tickers = len(tickers)

    # Filter to training data only if specified
    if train_end_year is not None:
        log_returns_df = log_returns_df[log_returns_df.index.year < train_end_year]
        print(f"Using data before {train_end_year} for correlation computation")

    print(f"Computing correlation from {len(log_returns_df)} days of equity log returns")
    print(f"Date range: {log_returns_df.index.min().date()} to {log_returns_df.index.max().date()}")

    # Compute Pearson correlation matrix
    corr_matrix = log_returns_df[tickers].corr().values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Build adjacency from correlation using threshold
    A = np.zeros((num_tickers, num_tickers))
    mask = np.abs(corr_matrix) >= threshold
    A[mask] = threshold  # Use threshold value (per spec: A_{i,j} = τ)
    np.fill_diagonal(A, 0)  # No self-loops initially

    # Count edges
    num_edges = (A > 0).sum() // 2
    density = (A > 0).sum() / (num_tickers * num_tickers)
    print(f"Edges (before normalization): {num_edges}")
    print(f"Density: {density:.4f}")

    if normalize:
        # Symmetric normalization: A_norm = D^{-1/2} (A + I) D^{-1/2}
        A_tilde = A + np.eye(num_tickers)
        d = A_tilde.sum(axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[d == 0] = 0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        A = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        print("Applied symmetric normalization")

    return A, tickers


def compute_straddle_pearson_adjacency(
    df,
    threshold,
    train_end_year=None,
    normalize=True,
    returns_column="daily_returns",
):
    """
    Compute Pearson correlation adjacency matrix from straddle returns.

    This matches the methodology used in the rolling Pearson implementation,
    but computes a single static graph.

    Args:
        df: DataFrame with straddle features (must have 'ticker', 'date', and returns_column)
        threshold: Correlation threshold (tau) for creating edges
        train_end_year: If provided, only use data before this year (avoids look-ahead bias)
        normalize: If True, apply symmetric normalization (D^{-1/2} A D^{-1/2})
        returns_column: Column name containing returns

    Returns:
        adjacency: Normalized adjacency matrix of shape (num_tickers, num_tickers)
        tickers: List of tickers in order
    """
    # Filter to training data only if specified
    if train_end_year is not None:
        df = df[df['date'].dt.year < train_end_year].copy()
        print(f"Using data before {train_end_year} for correlation computation")

    # Get sorted tickers
    tickers = sorted(df['ticker'].unique())
    num_tickers = len(tickers)

    # Pivot to get returns per ticker per date
    returns_pivot = df.pivot_table(
        index='date',
        columns='ticker',
        values=returns_column,
        aggfunc='first'
    )
    returns_pivot = returns_pivot[tickers]  # Ensure consistent ordering
    returns_pivot = returns_pivot.sort_index()

    print(f"Computing correlation from {len(returns_pivot)} days of straddle returns")
    print(f"Date range: {returns_pivot.index.min()} to {returns_pivot.index.max()}")

    # Compute Pearson correlation matrix
    corr_matrix = returns_pivot.corr().values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Build adjacency from correlation using threshold
    A = np.zeros((num_tickers, num_tickers))
    mask = np.abs(corr_matrix) >= threshold
    A[mask] = threshold  # Use threshold value (like rolling implementation)
    np.fill_diagonal(A, 0)  # No self-loops initially

    # Count edges
    num_edges = (A > 0).sum() // 2  # Divide by 2 for undirected
    density = (A > 0).sum() / (num_tickers * num_tickers)
    print(f"Edges (before normalization): {num_edges}")
    print(f"Density: {density:.4f}")

    if normalize:
        # Symmetric normalization: A_norm = D^{-1/2} (A + I) D^{-1/2}
        A_tilde = A + np.eye(num_tickers)  # Add self-loops
        d = A_tilde.sum(axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[d == 0] = 0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        A = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        print("Applied symmetric normalization")

    return A, tickers


def load_or_compute_adjacency(
    graph_type,
    df=None,
    alpha=None,
    beta=None,
    tau=None,
    tickers=None,
    train_end_year=None,
    normalize=True,
):
    """
    Load precomputed adjacency or compute Pearson-based adjacency on-the-fly.

    Args:
        graph_type: "cvx", "pearson", "straddle_pearson", or "equity_pearson"
        df: DataFrame with features (required for straddle_pearson)
        alpha, beta: CVX parameters
        tau: Correlation threshold
        tickers: List of tickers
        train_end_year: End year for training data (for straddle_pearson/equity_pearson)
        normalize: Whether to normalize (for straddle_pearson/equity_pearson)

    Returns:
        adjacency: Adjacency matrix
    """
    if graph_type == "straddle_pearson":
        if df is None:
            raise ValueError("df is required for straddle_pearson graph type")
        adjacency, _ = compute_straddle_pearson_adjacency(
            df=df,
            threshold=tau,
            train_end_year=train_end_year,
            normalize=normalize,
        )
        return adjacency
    elif graph_type == "equity_pearson":
        adjacency, _ = compute_equity_pearson_adjacency(
            threshold=tau,
            train_end_year=train_end_year,
            normalize=normalize,
        )
        return adjacency
    else:
        # Load precomputed adjacency ("cvx" or "pearson")
        from gml.graph_model_2_v2 import load_adjacency_matrix
        return load_adjacency_matrix(
            graph_type=graph_type,
            alpha=alpha,
            beta=beta,
            tau=tau,
            tickers=tickers,
        )
