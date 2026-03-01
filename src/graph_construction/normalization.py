"""
Adjacency matrix normalization.
"""

import numpy as np


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
