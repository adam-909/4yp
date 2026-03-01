"""
Graph evaluation metrics (Section 5.3).
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings


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
