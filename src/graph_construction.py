"""
Graph Construction Module for LSTM-GCN Options Trading

This module implements graph construction methods as described in Section 5 of the paper:
- Pearson correlation-based adjacency matrix (Section 5.2.1)
- Convex optimization-based adjacency matrix (Section 5.2.2)
- Graph evaluation metrics (Section 5.3)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import warnings


# =============================================================================
# Pearson Correlation Method (Section 5.2.1)
# =============================================================================

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


# =============================================================================
# Convex Optimization Method (Section 5.2.2)
# =============================================================================

def learn_adjacency_convex_numpy(
    X: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-2,
    lr: float = 0.01,
    verbose: bool = False
) -> np.ndarray:
    """
    Learn adjacency matrix via convex optimization (NumPy fallback).

    Minimizes Equation (42):
    min_A  tr(X^T(D-A)X) - α·1^T·log(A·1) + β·||A||_F²
    s.t.   A = A^T, A_ij ≥ 0 ∀i≠j

    Args:
        X: Feature matrix (N x F*delta), each row is a node
        alpha: Connectivity regularizer
        beta: Sparsity regularizer
        max_iter: Maximum iterations
        tol: Convergence tolerance
        lr: Learning rate
        verbose: Print progress

    Returns:
        Adjacency matrix (N x N)
    """
    N = X.shape[0]

    # Initialize A with small positive values
    A = np.random.uniform(0.01, 0.1, (N, N))
    A = (A + A.T) / 2  # Symmetric
    np.fill_diagonal(A, 0)

    # Precompute X @ X.T for efficiency
    XXt = X @ X.T

    prev_loss = np.inf

    for iteration in range(max_iter):
        # Compute degree matrix D
        d = A.sum(axis=1)
        D = np.diag(d)

        # Compute Laplacian term: tr(X^T L X) = tr(X^T (D-A) X)
        L = D - A
        laplacian_term = np.trace(X.T @ L @ X)

        # Connectivity term: -α * 1^T log(A·1)
        Ad1 = A.sum(axis=1)
        Ad1_safe = np.maximum(Ad1, 1e-10)
        connectivity_term = -alpha * np.sum(np.log(Ad1_safe))

        # Sparsity term: β * ||A||_F²
        sparsity_term = beta * np.sum(A ** 2)

        loss = laplacian_term + connectivity_term + sparsity_term

        # Check convergence
        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
        prev_loss = loss

        # Compute gradient
        # ∂/∂A [tr(X^T(D-A)X)] = diag(X @ X^T) @ 1^T + 1 @ diag(X @ X^T)^T - 2 * X @ X^T
        diag_XXt = np.diag(XXt)
        grad_laplacian = np.outer(diag_XXt, np.ones(N)) + np.outer(np.ones(N), diag_XXt) - 2 * XXt

        # ∂/∂A [-α * 1^T log(A·1)] = -α / (A·1)
        grad_connectivity = -alpha / Ad1_safe.reshape(-1, 1)
        grad_connectivity = np.tile(grad_connectivity, (1, N))

        # ∂/∂A [β * ||A||_F²] = 2β * A
        grad_sparsity = 2 * beta * A

        grad = grad_laplacian + grad_connectivity + grad_sparsity

        # Make gradient symmetric
        grad = (grad + grad.T) / 2

        # Update A
        A = A - lr * grad

        # Project: A >= 0 and symmetric
        A = np.maximum(A, 0)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        if verbose and iteration % 1000 == 0:
            print(f"Iter {iteration}: loss = {loss:.4f}")

    return A


def learn_adjacency_convex_torch(
    X: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-2,
    lr: float = 0.01,
    verbose: bool = False,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Learn adjacency matrix via convex optimization (PyTorch).

    Minimizes Equation (42):
    min_A  tr(X^T(D-A)X) - α·1^T·log(A·1) + β·||A||_F²
    s.t.   A = A^T, A_ij ≥ 0 ∀i≠j

    Args:
        X: Feature matrix (N x F*delta), each row is a node
        alpha: Connectivity regularizer
        beta: Sparsity regularizer
        max_iter: Maximum iterations
        tol: Convergence tolerance (for early stopping)
        lr: Learning rate
        verbose: Print progress
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        Adjacency matrix (N x N)
    """
    try:
        import torch
    except ImportError:
        warnings.warn("PyTorch not available, falling back to NumPy implementation")
        return learn_adjacency_convex_numpy(X, alpha, beta, max_iter, tol, lr, verbose)

    N = X.shape[0]

    # Convert to torch
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # Initialize A as learnable parameter (upper triangular to enforce symmetry)
    A_upper = torch.nn.Parameter(
        torch.rand(N, N, device=device) * 0.1
    )

    optimizer = torch.optim.Adam([A_upper], lr=lr)

    prev_loss = float('inf')
    patience = 100
    patience_counter = 0

    for iteration in range(max_iter):
        optimizer.zero_grad()

        # Construct symmetric A from upper triangular
        A = torch.triu(A_upper, diagonal=1)
        A = A + A.T
        A = torch.relu(A)  # Enforce non-negativity

        # Compute degree
        d = A.sum(dim=1)

        # Laplacian term: tr(X^T(D-A)X)
        # = sum_i d_i * ||x_i||² - 2 * sum_{i<j} A_ij * x_i^T x_j
        XXt = X_t @ X_t.T
        laplacian_term = (d * torch.diag(XXt)).sum() - (A * XXt).sum()

        # Connectivity term: -α * 1^T log(A·1)
        d_safe = torch.clamp(d, min=1e-10)
        connectivity_term = -alpha * torch.log(d_safe).sum()

        # Sparsity term: β * ||A||_F²
        sparsity_term = beta * (A ** 2).sum()

        loss = laplacian_term + connectivity_term + sparsity_term

        # Backprop
        loss.backward()
        optimizer.step()

        # Early stopping check
        loss_val = loss.item()
        if abs(prev_loss - loss_val) < tol:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
        else:
            patience_counter = 0
        prev_loss = loss_val

        if verbose and iteration % 1000 == 0:
            print(f"Iter {iteration}: loss = {loss_val:.4f}")

    # Extract final A
    with torch.no_grad():
        A = torch.triu(A_upper, diagonal=1)
        A = A + A.T
        A = torch.relu(A)
        A_np = A.cpu().numpy()

    return A_np


def learn_adjacency_convex(
    X: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    max_iter: int = 10000,
    tol: float = 1e-2,
    lr: float = 0.01,
    verbose: bool = False,
    use_torch: bool = True,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Learn adjacency matrix via convex optimization.

    Wrapper that uses PyTorch if available, else falls back to NumPy.

    Args:
        X: Feature matrix (N x F*delta)
        alpha: Connectivity regularizer
        beta: Sparsity regularizer
        max_iter: Maximum iterations
        tol: Convergence tolerance
        lr: Learning rate
        verbose: Print progress
        use_torch: Try to use PyTorch
        device: PyTorch device

    Returns:
        Adjacency matrix (N x N)
    """
    if use_torch:
        try:
            import torch
            return learn_adjacency_convex_torch(
                X, alpha, beta, max_iter, tol, lr, verbose, device
            )
        except ImportError:
            pass

    return learn_adjacency_convex_numpy(
        X, alpha, beta, max_iter, tol, lr, verbose
    )


def grid_search_convex(
    X: np.ndarray,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    sector_labels: Optional[np.ndarray] = None,
    max_iter: int = 5000,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Grid search over alpha and beta parameters.

    Args:
        X: Feature matrix (N x F*delta)
        alphas: List of alpha values
        betas: List of beta values
        sector_labels: Optional sector labels for homophily
        max_iter: Max iterations per run
        verbose: Print progress

    Returns:
        DataFrame with metrics for each (alpha, beta) pair
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    if betas is None:
        betas = [0.01, 0.1, 1.0, 10.0]

    results = []

    for alpha in alphas:
        for beta in betas:
            if verbose:
                print(f"Running alpha={alpha}, beta={beta}")

            A = learn_adjacency_convex(
                X, alpha=alpha, beta=beta, max_iter=max_iter, verbose=False
            )

            metrics = {
                'alpha': alpha,
                'beta': beta,
                'connectivity': compute_connectivity(A),
                'num_edges': int(np.sum(A > 0) / 2),
                'density': np.sum(A > 0) / (A.shape[0] * (A.shape[0] - 1))
            }

            if sector_labels is not None:
                metrics['homophily'] = compute_edge_homophily(A, sector_labels)

            try:
                modularity, _ = compute_louvain_modularity(A)
                metrics['modularity'] = modularity
            except Exception:
                metrics['modularity'] = np.nan

            results.append(metrics)

    return pd.DataFrame(results)


def build_graph_ensemble(
    feature_tensor: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    K: int = 5,
    periods: Optional[List[int]] = None,
    max_iter: int = 5000,
    verbose: bool = False
) -> np.ndarray:
    """
    Build ensemble adjacency matrix from multiple time periods.

    Equation (43):
    Ã = (1/K) Σ_k A^(k)

    Args:
        feature_tensor: Feature tensor (delta x F x N)
        alpha: Connectivity regularizer
        beta: Sparsity regularizer
        K: Number of ensemble members
        periods: List of period lengths (in days). Default: [252, 504, 756, 1008, 1260]
        max_iter: Max iterations per period
        verbose: Print progress

    Returns:
        Ensemble adjacency matrix (N x N)
    """
    if periods is None:
        periods = [252, 504, 756, 1008, 1260]  # 1-5 years

    delta, F, N = feature_tensor.shape

    # Ensure we have enough periods
    periods = [p for p in periods if p <= delta]
    K = min(K, len(periods))

    if K == 0:
        # Not enough data, use full period
        X = feature_tensor.transpose(2, 1, 0).reshape(N, -1)  # N x (F*delta)
        return learn_adjacency_convex(X, alpha, beta, max_iter, verbose=verbose)

    A_ensemble = np.zeros((N, N))

    for k, period in enumerate(periods[:K]):
        if verbose:
            print(f"Building graph for period {k+1}/{K} ({period} days)")

        # Use last 'period' days
        start_idx = max(0, delta - period)
        X_period = feature_tensor[start_idx:, :, :]

        # Reshape to N x (F*period)
        X = X_period.transpose(2, 1, 0).reshape(N, -1)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        A_k = learn_adjacency_convex(X, alpha, beta, max_iter, verbose=False)
        A_ensemble += A_k

    A_ensemble /= K

    return A_ensemble


def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Normalize adjacency matrix using symmetric normalization.

    Â = D̃^{-1/2} Ã D̃^{-1/2}

    where Ã = A + I (if add_self_loops)

    Args:
        A: Adjacency matrix (N x N)
        add_self_loops: Whether to add self-loops

    Returns:
        Normalized adjacency matrix
    """
    if add_self_loops:
        A_tilde = A + np.eye(A.shape[0])
    else:
        A_tilde = A.copy()

    # Compute degree
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[d == 0] = 0

    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Symmetric normalization
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    return A_norm


# =============================================================================
# Evaluation Metrics (Section 5.3)
# =============================================================================

def compute_connectivity(A: np.ndarray) -> float:
    """
    Compute graph connectivity (algebraic connectivity).

    Uses second smallest eigenvalue of Laplacian.

    Args:
        A: Adjacency matrix (N x N)

    Returns:
        Algebraic connectivity (λ_2)
    """
    # Compute Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.sort(eigenvalues)

    # Return second smallest (first is always ~0)
    if len(eigenvalues) > 1:
        return float(eigenvalues[1])
    return 0.0


def compute_edge_homophily(A: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute edge homophily ratio.

    Equation (4):
    h = |{(u,v) : (u,v) ∈ E ∧ y_u = y_v}| / |E|

    Args:
        A: Adjacency matrix (N x N)
        labels: Node labels (N,)

    Returns:
        Homophily ratio in [0, 1]
    """
    # Find edges
    edges = np.where(A > 0)

    if len(edges[0]) == 0:
        return 0.0

    # Count same-label edges
    same_label = 0
    total_edges = 0

    for i, j in zip(edges[0], edges[1]):
        if i < j:  # Avoid double counting
            total_edges += 1
            if labels[i] == labels[j]:
                same_label += 1

    if total_edges == 0:
        return 0.0

    return same_label / total_edges


def compute_louvain_modularity(
    A: np.ndarray,
    resolution: float = 1.0
) -> Tuple[float, List]:
    """
    Compute Louvain modularity and detect communities.

    Args:
        A: Adjacency matrix (N x N)
        resolution: Resolution parameter

    Returns:
        Tuple of (modularity, community_labels)
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity
    except ImportError:
        warnings.warn("networkx not available for community detection")
        return np.nan, []

    # Create graph from adjacency
    G = nx.from_numpy_array(A)

    if G.number_of_edges() == 0:
        return 0.0, [set(range(A.shape[0]))]

    # Run Louvain
    communities = louvain_communities(G, resolution=resolution, seed=42)
    communities = list(communities)

    # Compute modularity
    mod = modularity(G, communities)

    # Convert to label array
    labels = np.zeros(A.shape[0], dtype=int)
    for comm_idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = comm_idx

    return mod, labels.tolist()


def compute_graph_metrics(
    A: np.ndarray,
    sector_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute comprehensive metrics for a graph.

    Args:
        A: Adjacency matrix
        sector_labels: Optional sector labels

    Returns:
        Dictionary of metrics
    """
    N = A.shape[0]
    num_edges = int(np.sum(A > 0) / 2)

    metrics = {
        'num_nodes': N,
        'num_edges': num_edges,
        'density': num_edges / (N * (N - 1) / 2) if N > 1 else 0,
        'avg_degree': A.sum() / N if N > 0 else 0,
        'connectivity': compute_connectivity(A),
        'avg_weight': A[A > 0].mean() if np.any(A > 0) else 0,
        'max_weight': A.max()
    }

    if sector_labels is not None:
        metrics['edge_homophily'] = compute_edge_homophily(A, sector_labels)

    try:
        modularity, communities = compute_louvain_modularity(A)
        metrics['louvain_modularity'] = modularity
        metrics['num_communities'] = len(set(communities))
    except Exception:
        metrics['louvain_modularity'] = np.nan
        metrics['num_communities'] = np.nan

    return metrics


# =============================================================================
# Utility Functions
# =============================================================================

def reshape_feature_tensor_for_graph(
    feature_tensor: np.ndarray
) -> np.ndarray:
    """
    Reshape feature tensor from (delta, F, N) to (N, F*delta) for graph learning.

    Args:
        feature_tensor: Shape (delta, F, N)

    Returns:
        Reshaped matrix (N, F*delta)
    """
    delta, F, N = feature_tensor.shape
    # Transpose to (N, F, delta) then reshape to (N, F*delta)
    X = feature_tensor.transpose(2, 1, 0).reshape(N, -1)
    return X


def create_sector_labels(
    ticker_list: List[str],
    sector_map: Dict[str, str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create numeric sector labels from ticker list and sector mapping.

    Args:
        ticker_list: List of ticker symbols
        sector_map: Dictionary mapping ticker -> sector name

    Returns:
        Tuple of (numeric_labels, sector_to_id mapping)
    """
    # Get unique sectors
    sectors = sorted(set(sector_map.get(t, 'Unknown') for t in ticker_list))
    sector_to_id = {s: i for i, s in enumerate(sectors)}

    # Create label array
    labels = np.array([
        sector_to_id[sector_map.get(t, 'Unknown')]
        for t in ticker_list
    ])

    return labels, sector_to_id
