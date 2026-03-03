import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from collections import defaultdict
from settings.default import ALL_TICKERS, BBG_SECTORS
# (Other imports such as networkx, torch, etc. omitted if not needed.)

# Parameters used in the file name.
alpha = 100
beta = 0.01

# Build the full file path.
path = r"C:\Users\Sean\Documents\gml-master\data\graph_structure\cvx_opt"
csv_file = os.path.join(path, f"{alpha}_{beta}_cvx.csv")

# Read the CSV file. Assumes that the first column in the CSV contains ticker names.
A_norm = pd.read_csv(csv_file, index_col=0)

# Create groups of tickers by sector.
sector_groups = defaultdict(list)
# Here we assume ALL_TICKERS are expected to appear in the CSV.
for ticker in ALL_TICKERS:
    sector = BBG_SECTORS.get(ticker, "Unknown")
    sector_groups[sector].append(ticker)

# Sort tickers within each sector.
for sector in sector_groups:
    sector_groups[sector].sort()

# Get a sorted list of all sectors.
ordered_sectors = sorted(sector_groups.keys())

# Build the ordered tickers list and also record the boundaries per sector.
ordered_tickers = []
sector_boundaries = []
current_index = 0
for sector in ordered_sectors:
    tickers_in_sector = sector_groups[sector]
    # Optionally, filter to keep only tickers that exist in the DataFrame.
    tickers_in_sector = [t for t in tickers_in_sector if t in A_norm.index]
    ordered_tickers.extend(tickers_in_sector)
    sector_boundaries.append((current_index, current_index + len(tickers_in_sector)))
    current_index += len(tickers_in_sector)

# Build a mapping from ticker to its integer index in the DataFrame.
ticker_to_index = {ticker: i for i, ticker in enumerate(A_norm.index)}

# Get the indices corresponding to the ordered tickers.
new_indices = [ticker_to_index[ticker] for ticker in ordered_tickers]

# Reorder the DataFrame. We use .iloc with the new indices.
A_ordered = A_norm.iloc[new_indices, new_indices]

# =============================================================================
# Define function to draw a curly brace with label.
# =============================================================================
def draw_curly_brace(ax, x, y0, y1, label, color='black', lw=2, fontsize=12):
    t = np.linspace(0, np.pi, 100)
    dx = 0.3  
    brace_x = x + dx * np.sin(t)
    brace_y = y0 + (y1 - y0) * (1 - np.cos(t)) / 2.0
    ax.plot(brace_x, brace_y, color=color, lw=lw)
    ax.text(x - 0.4, (y0 + y1) / 2, label, color=color,
            fontsize=fontsize, va='center', ha='right')

# =============================================================================
# Plot the heatmap with boxes and curly braces.
# =============================================================================
plt.figure(figsize=(12, 12))
ax = sns.heatmap(
    A_ordered,
    cmap="YlOrRd",
    square=True,
    xticklabels=ordered_tickers,
    yticklabels=ordered_tickers,
    cbar=True
)

# Draw a rectangle around each sector block.
for (start, end) in sector_boundaries:
    rect = patches.Rectangle((start, start), end - start, end - start,
                             fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

# Draw curly braces to label the sector groups.
for (start, end), sector in zip(sector_boundaries, ordered_sectors):
    draw_curly_brace(ax, -2.0, start, end, sector, color='black', lw=2, fontsize=12)

plt.title("Normalized Thresholded Adjacency Matrix Grouped by Sector")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
