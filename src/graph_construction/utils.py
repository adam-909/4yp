"""
Utility functions for graph construction.
"""

import numpy as np
from typing import List, Dict, Tuple


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
