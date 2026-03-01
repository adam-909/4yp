"""
Convex optimization-based graph construction (Section 5.2.2).
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import warnings


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
    from .metrics import compute_connectivity, compute_edge_homophily, compute_louvain_modularity

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
