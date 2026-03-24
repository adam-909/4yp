"""
Graph visualization utilities for LSTM-GCN adjacency matrices.

Visualizes the graph structure with:
- Nodes colored by sector
- Ticker labels on each node
- Edge weights shown by line thickness/opacity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Optional, Dict, List

from settings.default import ALL_TICKERS, BBG_SECTORS


# Sector color palette (colorblind-friendly)
SECTOR_COLORS = {
    "Information Technology": "#1f77b4",  # blue
    "Healthcare": "#2ca02c",              # green
    "Financials": "#ff7f0e",              # orange
    "Consumer Discretionary": "#d62728",  # red
    "Consumer Staples": "#9467bd",        # purple
    "Industrials": "#8c564b",             # brown
    "Communication Services": "#e377c2",  # pink
    "Energy": "#7f7f7f",                  # gray
    "Utilities": "#bcbd22",               # olive
    "Real Estate": "#17becf",             # cyan
}


def plot_adjacency_graph(
    adjacency: np.ndarray,
    tickers: List[str] = None,
    sectors: Dict[str, str] = None,
    title: str = "Adjacency Graph",
    figsize: tuple = (16, 14),
    node_size: int = 800,
    font_size: int = 8,
    edge_threshold: float = 0.0,
    layout: str = "spring",
    show_edge_weights: bool = False,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Plot the adjacency matrix as a network graph with sector-colored nodes.

    Args:
        adjacency: NxN adjacency matrix
        tickers: List of ticker symbols (default: ALL_TICKERS)
        sectors: Dict mapping ticker -> sector (default: BBG_SECTORS)
        title: Plot title
        figsize: Figure size
        node_size: Size of nodes
        font_size: Font size for labels
        edge_threshold: Minimum edge weight to display (filters weak connections)
        layout: Layout algorithm ("spring", "circular", "kamada_kawai", "spectral", "shell")
        show_edge_weights: If True, show edge weight labels
        save_path: If provided, save figure to this path
        seed: Random seed for reproducible layouts

    Returns:
        matplotlib Figure object
    """
    if tickers is None:
        tickers = ALL_TICKERS
    if sectors is None:
        sectors = BBG_SECTORS

    n = len(tickers)
    assert adjacency.shape == (n, n), f"Adjacency shape {adjacency.shape} doesn't match {n} tickers"

    # Create networkx graph
    G = nx.Graph()

    # Add nodes with sector attributes
    for i, ticker in enumerate(tickers):
        sector = sectors.get(ticker, "Unknown")
        G.add_node(ticker, sector=sector)

    # Add edges (only above threshold)
    for i in range(n):
        for j in range(i + 1, n):
            weight = adjacency[i, j]
            if abs(weight) > edge_threshold:
                G.add_edge(tickers[i], tickers[j], weight=weight)

    # Get node colors based on sector
    node_colors = [SECTOR_COLORS.get(sectors.get(ticker, "Unknown"), "#cccccc")
                   for ticker in tickers]

    # Choose layout
    np.random.seed(seed)
    if layout == "spring":
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "shell":
        # Group by sector for shell layout
        sector_groups = {}
        for ticker in tickers:
            sector = sectors.get(ticker, "Unknown")
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(ticker)
        shells = list(sector_groups.values())
        pos = nx.shell_layout(G, nlist=shells)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get edge weights for styling
    edges = G.edges(data=True)
    if len(edges) > 0:
        weights = [d.get('weight', 1.0) for _, _, d in edges]
        max_weight = max(abs(w) for w in weights) if weights else 1.0

        # Normalize edge widths and alphas
        edge_widths = [2.0 * abs(w) / max_weight for w in weights]
        edge_alphas = [0.3 + 0.7 * abs(w) / max_weight for w in weights]

        # Draw edges
        for (u, v, d), width, alpha in zip(edges, edge_widths, edge_alphas):
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, color='gray', linewidth=width, alpha=alpha, zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_weight='bold',
        ax=ax,
    )

    # Create legend for sectors
    unique_sectors = sorted(set(sectors.get(t, "Unknown") for t in tickers))
    legend_patches = [
        mpatches.Patch(color=SECTOR_COLORS.get(s, "#cccccc"), label=s)
        for s in unique_sectors
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        title="Sectors",
        title_fontsize=12,
    )

    # Stats
    num_edges = G.number_of_edges()
    max_possible = n * (n - 1) // 2
    density = num_edges / max_possible if max_possible > 0 else 0

    ax.set_title(f"{title}\n({num_edges} edges, {density:.1%} density)", fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def plot_adjacency_heatmap_with_sectors(
    adjacency: np.ndarray,
    tickers: List[str] = None,
    sectors: Dict[str, str] = None,
    title: str = "Adjacency Matrix by Sector",
    figsize: tuple = (14, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot adjacency matrix as a heatmap, sorted by sector.

    Args:
        adjacency: NxN adjacency matrix
        tickers: List of ticker symbols (default: ALL_TICKERS)
        sectors: Dict mapping ticker -> sector (default: BBG_SECTORS)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    if tickers is None:
        tickers = ALL_TICKERS
    if sectors is None:
        sectors = BBG_SECTORS

    # Sort tickers by sector
    sorted_tickers = sorted(tickers, key=lambda t: (sectors.get(t, "ZZZ"), t))

    # Reorder adjacency matrix
    idx_map = {t: i for i, t in enumerate(tickers)}
    sorted_indices = [idx_map[t] for t in sorted_tickers]
    sorted_adj = adjacency[np.ix_(sorted_indices, sorted_indices)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(sorted_adj, cmap='RdYlBu_r', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Edge Weight', fontsize=12)

    # Add tick labels
    ax.set_xticks(range(len(sorted_tickers)))
    ax.set_yticks(range(len(sorted_tickers)))
    ax.set_xticklabels(sorted_tickers, rotation=90, fontsize=7)
    ax.set_yticklabels(sorted_tickers, fontsize=7)

    # Add sector dividers
    current_sector = None
    sector_boundaries = []
    for i, ticker in enumerate(sorted_tickers):
        sector = sectors.get(ticker, "Unknown")
        if sector != current_sector:
            if current_sector is not None:
                sector_boundaries.append(i - 0.5)
            current_sector = sector

    for boundary in sector_boundaries:
        ax.axhline(y=boundary, color='black', linewidth=1.5)
        ax.axvline(x=boundary, color='black', linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


def compare_adjacency_graphs(
    adj1: np.ndarray,
    adj2: np.ndarray,
    name1: str = "Graph 1",
    name2: str = "Graph 2",
    tickers: List[str] = None,
    sectors: Dict[str, str] = None,
    figsize: tuple = (20, 8),
    edge_threshold: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare two adjacency matrices side by side.

    Args:
        adj1, adj2: Adjacency matrices to compare
        name1, name2: Names for each graph
        tickers: List of ticker symbols
        sectors: Dict mapping ticker -> sector
        figsize: Figure size
        edge_threshold: Minimum edge weight to display
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    if tickers is None:
        tickers = ALL_TICKERS
    if sectors is None:
        sectors = BBG_SECTORS

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, adj, name in zip(axes, [adj1, adj2], [name1, name2]):
        n = len(tickers)
        G = nx.Graph()

        for i, ticker in enumerate(tickers):
            sector = sectors.get(ticker, "Unknown")
            G.add_node(ticker, sector=sector)

        for i in range(n):
            for j in range(i + 1, n):
                weight = adj[i, j]
                if abs(weight) > edge_threshold:
                    G.add_edge(tickers[i], tickers[j], weight=weight)

        node_colors = [SECTOR_COLORS.get(sectors.get(ticker, "Unknown"), "#cccccc")
                       for ticker in tickers]

        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

        # Draw edges
        edges = G.edges(data=True)
        if len(edges) > 0:
            weights = [d.get('weight', 1.0) for _, _, d in edges]
            max_weight = max(abs(w) for w in weights) if weights else 1.0
            edge_widths = [2.0 * abs(w) / max_weight for w in weights]
            edge_alphas = [0.3 + 0.7 * abs(w) / max_weight for w in weights]

            for (u, v, d), width, alpha in zip(edges, edge_widths, edge_alphas):
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(x, y, color='gray', linewidth=width, alpha=alpha, zorder=1)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold', ax=ax)

        num_edges = G.number_of_edges()
        ax.set_title(f"{name}\n({num_edges} edges)", fontsize=12, fontweight='bold')
        ax.axis('off')

    # Shared legend
    unique_sectors = sorted(set(sectors.get(t, "Unknown") for t in tickers))
    legend_patches = [
        mpatches.Patch(color=SECTOR_COLORS.get(s, "#cccccc"), label=s)
        for s in unique_sectors
    ]
    fig.legend(
        handles=legend_patches,
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
        fontsize=9,
        title="Sectors",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    return fig


# Convenience function for quick visualization
def visualize_graph(
    graph_type: str = "pearson",
    tau: float = 0.45,
    alpha: float = 100,
    beta: float = 0.1,
    layout: str = "spring",
    edge_threshold: float = 0.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Convenience function to quickly visualize a saved adjacency matrix.

    Args:
        graph_type: "pearson" or "cvx"
        tau: Threshold for pearson graph
        alpha, beta: Parameters for CVX graph
        layout: Layout algorithm
        edge_threshold: Minimum edge weight to display
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    import os

    if graph_type == "pearson":
        graph_file = os.path.join("data", "graph_structure", "pearson", f"{tau}.csv")
        title = f"Pearson Correlation Graph (τ={tau})"
    elif graph_type == "cvx":
        graph_file = os.path.join("data", "graph_structure", "cvx_opt", f"{alpha}_{beta}_cvx.csv")
        title = f"CVX Optimization Graph (α={alpha}, β={beta})"
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    adjacency_df = pd.read_csv(graph_file, index_col=0)
    adjacency_df = adjacency_df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS)
    adjacency = adjacency_df.values

    return plot_adjacency_graph(
        adjacency,
        title=title,
        layout=layout,
        edge_threshold=edge_threshold,
        save_path=save_path,
    )
