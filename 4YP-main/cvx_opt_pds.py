import numpy as np
import pandas as pd
import torch
from torch.nn.functional import relu
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import networkx as nx
from collections import defaultdict
from networkx.algorithms.community.quality import modularity

from settings.default import ALL_TICKERS, BBG_SECTORS

# ------------------------------
# 1. Hyperparameters & Data
# ------------------------------
alpha = 100.0   # log barrier weight, for instance
beta  = 0.01    # Frobenius penalty weight
L     = 50      # number of primal-dual iterations (unrolling steps)
eta   = 1e-3    # step size for primal
sigma = 1e-3    # step size for dual
d     = 252     # single lookback period

feature_columns = [
    'norm_daily_return',
    'norm_monthly_return',
    'norm_quarterly_return',
    'norm_biannual_return',
    'norm_annual_return',
    'macd_2_8',
    'macd_4_16',
    'macd_8_32'
]

df = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv")
df['date'] = pd.to_datetime(df['date'])

# ------------------------------
# 2. Build Flattened Feature Matrix V_t
# ------------------------------
ticker_features = []
for ticker in ALL_TICKERS:
    ticker_df = df[df["ticker"] == ticker].sort_values(by="date")
    if ticker_df.shape[0] >= d:
        ticker_data = ticker_df.iloc[-d:][feature_columns].values
    else:
        missing = d - ticker_df.shape[0]
        ticker_data = pd.DataFrame(0, index=range(missing), columns=feature_columns).values
        ticker_data = np.vstack((ticker_data, ticker_df[feature_columns].values))
    ticker_features.append(ticker_data.flatten())

V_t = torch.tensor(ticker_features, dtype=torch.float32)
print("V_t shape:", V_t.shape)

# Optionally, normalize V_t column-wise.
col_mean = V_t.mean(dim=0, keepdim=True)
col_std  = V_t.std(dim=0, keepdim=True) + 1e-8
V_t = (V_t - col_mean) / col_std

N = V_t.shape[0]  # number of tickers
M = N * N         # full NxN matrix

# ------------------------------
# 3. Helper Functions: Pack/Unpack, Prox, etc.
# ------------------------------
def packA(A):
    # Flatten NxN matrix into (N*N,) vector.
    return A.view(-1)

def unpackA(w):
    # Reshape vector back to NxN matrix.
    return w.view(N, N)

def row_sum_operator(A):
    # Sum along rows.
    return A.sum(dim=1)

def prox_nonneg(X):
    # Enforce non-negativity.
    return relu(X)

# ------------------------------
# 4. Primal–Dual Splitting (PDS) Iteration
# ------------------------------
# Initialize the primal variable w (vectorized A) and dual variable v.
w = torch.rand(M, dtype=torch.float32, requires_grad=False)
v = torch.zeros(N, dtype=torch.float32, requires_grad=False)

# (Optional) Store best w if needed; here we run for a fixed number of iterations.
for l in range(L):
    # Dual update: v^{l+1} = prox_{sigma*g^*}( v^l + sigma*K(w^l) )
    # In our simplified example, let K be the row-sum operator.
    A_current = unpackA(w)  # shape: (N, N)
    current_row_sums = row_sum_operator(A_current)  # shape: (N,)
    # Update dual: here, simply add the row sums scaled by sigma.
    v_tilde = v + sigma * current_row_sums
    v = prox_nonneg(v_tilde)  # enforce non-negativity

    # Primal update: w^{l+1} = prox_{eta*f}( w^l - eta*K^*(v^{l+1}) )
    # For demonstration, we compute a naive gradient for the primal variable.
    with torch.no_grad():
        A_current = unpackA(w)
        # Placeholder gradient for the log barrier part:
        rs = row_sum_operator(A_current) + 1e-8  # shape (N,)
        inv_rs = (-alpha / (rs ** 2)).unsqueeze(1)  # shape (N,1)
        grad_log = inv_rs.expand(N, N)  # placeholder gradient term
        grad_frob = 2.0 * beta * A_current  # gradient of Frobenius norm penalty
        # Dual influence (from v): we add v[i] to each element in row i.
        v_expanded = v.unsqueeze(1).expand(N, N)
        grad_dual = v_expanded
        grad_total = grad_log + grad_frob + grad_dual
        grad_w = packA(grad_total)
    
    # Update the primal variable:
    w = prox_nonneg(w - eta * grad_w)

# Reconstruct the final adjacency matrix from w.
A_final = unpackA(w)  # shape: (N, N)

# ------------------------------
# 5. Graph Normalization of the Learned Adjacency
# ------------------------------
A_np = A_final.detach().cpu().numpy()
row_sum = A_np.sum(axis=1)
row_sum[row_sum == 0] = 1e-8
D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
A_norm = D_inv_sqrt @ A_np @ D_inv_sqrt

# ------------------------------
# 6. Compute Statistics on the Final Graph
# ------------------------------
# Edge homophily ratio:
def edge_homophily_ratio(A, tickers, sectors):
    A_np = A.detach().cpu().numpy()
    total_weight = 0.0
    same_sector_weight = 0.0
    n = len(tickers)
    for i in range(n):
        for j in range(i+1, n):
            w = A_np[i, j]
            total_weight += w
            if sectors[tickers[i]] == sectors[tickers[j]]:
                same_sector_weight += w
    return same_sector_weight / total_weight if total_weight > 0 else 0.0

ehr = edge_homophily_ratio(A_final, ALL_TICKERS, BBG_SECTORS)

# Number of edges and average edge weight (count each undirected edge once)
num_edges = 0
total_edge_weight = 0.0
for i in range(N):
    for j in range(i+1, N):
        w_val = A_np[i, j]
        if w_val > 0:
            num_edges += 1
            total_edge_weight += w_val
avg_edge_weight = total_edge_weight / num_edges if num_edges > 0 else 0.0

# Compute Louvain modularity score using the normalized adjacency matrix.
df_norm = pd.DataFrame(A_norm, index=ALL_TICKERS, columns=ALL_TICKERS)
G = nx.from_pandas_adjacency(df_norm)
communities = nx.algorithms.community.louvain_communities(G, resolution=20)
mod_score = modularity(G, communities, weight='weight')

print("Final loss (from PDS unrolling, approximate):", None)  # Not computed in PDS code.
print("Edge Homophily Ratio:", ehr)
print("Number of edges (undirected):", num_edges)
print("Average edge weight:", avg_edge_weight)
print("Louvain modularity score:", mod_score)

# ------------------------------
# 7. Plot the Final Graph Heatmap (Normalized)
# ------------------------------
# For plotting, we reorder the matrix by grouping tickers by sector.
sector_groups = defaultdict(list)
for ticker in ALL_TICKERS:
    sector = BBG_SECTORS.get(ticker, "Unknown")
    sector_groups[sector].append(ticker)
for sec in sector_groups:
    sector_groups[sec].sort()
ordered_sectors = sorted(sector_groups.keys())
ordered_tickers = []
sector_boundaries = []
current_idx = 0
for sec in ordered_sectors:
    tickers_in_sec = sector_groups[sec]
    ordered_tickers.extend(tickers_in_sec)
    sector_boundaries.append((current_idx, current_idx + len(tickers_in_sec)))
    current_idx += len(tickers_in_sec)

ticker_to_index = {t: i for i, t in enumerate(ALL_TICKERS)}
new_indices = [ticker_to_index[t] for t in ordered_tickers]
A_ordered = A_norm[np.ix_(new_indices, new_indices)]

plt.figure(figsize=(12,12))
ax = sns.heatmap(A_ordered,
                 cmap="YlOrRd",
                 square=True,
                 xticklabels=ordered_tickers,
                 yticklabels=ordered_tickers,
                 cbar=True)
for (start, end) in sector_boundaries:
    rect = patches.Rectangle((start, start), end-start, end-start,
                             fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
plt.title("Normalized Ensemble Adjacency Matrix (PDS Unrolling for d=252)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()