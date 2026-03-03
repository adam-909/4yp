import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from collections import defaultdict
from settings.default import ALL_TICKERS, BBG_SECTORS
import networkx as nx
import torch
from torch_geometric.data import Data
from networkx.algorithms.community import louvain_communities, modularity


df = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv")


# =============================================================================
# 1. Parameters and Data Preparation
# =============================================================================
# (We will grid search over these values)
# alpha_list = [0.01, 0.1, 1, 10]  # Regularization weight for log term
# beta_list  = [0.01, 0.1, 1, 10]   # Regularization weight for Frobenius norm
alpha_list = [0.1, 1, 10, 100]  # Example values; can extend to [0.01, 0.1, 1, 10]
beta_list  = [0.01, 0.1, 1, 10]  # Example values; can extend to [0.01, 0.1, 1, 10]
d = 252 * 1        # Lookback period: 1 year of trading days
tau = 0.0         # Threshold for edge weights

min_delta = 1e-2  # Minimal improvement threshold for early stopping
patience = 25

feature_columns = [
    'norm_daily_return',
    'norm_monthly_return',
    'norm_quarterly_return',
    'norm_biannual_return',
    'norm_annual_return',
    'macd_2_8',
    'macd_4_16',
    'macd_8_32'
    'log_moneyness',
    'time_to_expiry'
]

df['date'] = pd.to_datetime(df['date'])

# =============================================================================
# 2. Construct the Flattened Feature Matrix for Each Ticker
# =============================================================================
ticker_features = []
for ticker in ALL_TICKERS:
    ticker_df = df[df["ticker"] == ticker].sort_values(by="date")
    if ticker_df.shape[0] >= d:
        ticker_data = ticker_df.iloc[-d:][feature_columns].values
    else:
        missing_days = d - ticker_df.shape[0]
        ticker_data = pd.DataFrame(0, index=range(missing_days), columns=feature_columns).values
        ticker_data = np.vstack((ticker_data, ticker_df[feature_columns].values))
    ticker_features.append(ticker_data.flatten())

V_t = torch.tensor(ticker_features, dtype=torch.float32)
print("V_t shape:", V_t.shape)

# =============================================================================
# 2.5 Normalize the Feature Matrix (column-wise)
# =============================================================================
V_mean = V_t.mean(dim=0, keepdim=True)
V_std = V_t.std(dim=0, keepdim=True) + 1e-8
V_t = (V_t - V_mean) / V_std

# =============================================================================
# 3. Define Helper Functions: Degree Matrix, Objective Function, etc.
# =============================================================================
def degree_matrix(A):
    return torch.diag(A.sum(dim=1))

def objective_function(A, V, alpha, beta):
    D = degree_matrix(A)
    spectral_term = torch.trace(V.T @ (D - A) @ V)
    log_term = -alpha * torch.sum(torch.log(A.sum(dim=1) + 1e-8))
    frobenius_term = beta * torch.norm(A, p="fro") ** 2
    return spectral_term + log_term + frobenius_term

def edge_homophily_ratio(A, tickers, sectors):
    A_np = A.detach().cpu().numpy()
    total_weight = 0.0
    same_sector_weight = 0.0
    num_nodes = len(tickers)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            weight = A_np[i, j]
            total_weight += weight
            if sectors[tickers[i]] == sectors[tickers[j]]:
                same_sector_weight += weight
    return same_sector_weight / total_weight if total_weight > 0 else 0.0

def adjacency_to_edge_index(A):
    edge_index = torch.nonzero(A > 0, as_tuple=False).T
    edge_weight = A[edge_index[0], edge_index[1]]
    return edge_index, edge_weight

def normalize_adjacency(A):
    """
    Perform symmetric normalization: A_norm = D^{-1/2} * A * D^{-1/2}
    """
    eps = 1e-8
    row_sum = A.sum(axis=1)
    row_sum[row_sum == 0] = eps
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def compute_louvain_modularity(A_thresh):
    """
    Given a thresholded learned adjacency matrix (torch.Tensor), normalize it and compute
    the Louvain modularity score with resolution 20.
    """
    A_thresh_np = A_thresh.detach().cpu().numpy()
    A_norm = normalize_adjacency(A_thresh_np)
    df_norm = pd.DataFrame(A_norm, index=ALL_TICKERS, columns=ALL_TICKERS)
    G_norm = nx.from_pandas_adjacency(df_norm)
    communities = louvain_communities(G_norm, resolution=20)
    mod_score = modularity(G_norm, communities, weight='weight')
    return mod_score

# =============================================================================
# 4. Grid Search over Alpha and Beta with Early Stopping
# =============================================================================
num_epochs = 10000
results = []

for alpha_val in alpha_list:
    for beta_val in beta_list:
        N = V_t.shape[0]
        A_t = torch.nn.Parameter(torch.rand(N, N, requires_grad=True))
        with torch.no_grad():
            A_t.data = (A_t.data + A_t.data.T) / 2  
            A_t.data.clamp_(min=0)
        
        optimizer = torch.optim.Adam([A_t], lr=1e-3)
        best_loss = float('inf')
        epochs_no_improve = 0
        best_A_t = None
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = objective_function(A_t, V_t, alpha_val, beta_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([A_t], max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                A_t.data = (A_t.data + A_t.data.T) / 2
                A_t.data.clamp_(min=0)
                A_t.data.fill_diagonal_(0)
            
            current_loss = loss.item()
            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                epochs_no_improve = 0
                best_A_t = A_t.data.clone()
            else:
                epochs_no_improve += 1
                
            if epoch % 100 == 0:
                print(f"alpha: {alpha_val}, beta: {beta_val}, epoch {epoch}, loss: {current_loss}")
                
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} for alpha {alpha_val}, beta {beta_val} with best loss {best_loss}")
                break
        
        if best_A_t is not None:
            with torch.no_grad():
                A_t.data = best_A_t
        
        ehr = edge_homophily_ratio(A_t, ALL_TICKERS, BBG_SECTORS)
        print(f"Grid Search - alpha: {alpha_val}, beta: {beta_val} -> best_loss: {best_loss}, edge homophily ratio: {ehr}")
        
        A_thresh = A_t.data.clone()
        A_thresh[A_thresh < tau] = 0.0
        ehr_thresh = edge_homophily_ratio(A_thresh, ALL_TICKERS, BBG_SECTORS)
        print(f"Updated Edge Homophily Ratio after thresholding (weights >= {tau}): {ehr_thresh}")
        
        num_edges = 0
        total_edge_weight = 0.0
        A_thresh_np = A_thresh.detach().cpu().numpy()
        for i in range(N):
            for j in range(i+1, N):
                if A_thresh_np[i, j] > 0:
                    num_edges += 1
                    total_edge_weight += A_thresh_np[i, j]
        avg_edge_weight = total_edge_weight / num_edges if num_edges > 0 else 0.0
        print(f"Number of edges (after thresholding): {num_edges}")
        print(f"Average edge weight (after thresholding): {avg_edge_weight}")
        
        # Compute Louvain modularity score on the thresholded matrix (normalized internally)
        mod_score = compute_louvain_modularity(A_thresh)
        print(f"Louvain modularity score: {mod_score}")
        
        results.append({
            "alpha": alpha_val,
            "beta": beta_val,
            "best_loss": best_loss,
            "edge_homophily_ratio": ehr,
            "edge_homophily_ratio_thresholded": ehr_thresh,
            "num_edges_thresholded": num_edges,
            "avg_edge_weight_thresholded": avg_edge_weight,
            "louvain_modularity_score": mod_score
            # "A_t_final": A_t.data.clone()  # Not saved in CSV
        })

# =============================================================================
# Select the run with the highest thresholded edge homophily ratio
# =============================================================================
best_run = max(results, key=lambda r: r["edge_homophily_ratio_thresholded"])
best_alpha = best_run["alpha"]
best_beta = best_run["beta"]
best_ehr = best_run["edge_homophily_ratio_thresholded"]
best_A_t = best_run.get("A_t_final", A_t.data)  # Use final A_t from last grid run if not stored

print(f"\nBest run achieved with alpha = {best_alpha}, beta = {best_beta}, with thresholded edge homophily ratio = {best_ehr}")

# =============================================================================
# 5. Prepare the Graph for PyTorch Geometric using best_A_t
# =============================================================================
edge_index, edge_weight = adjacency_to_edge_index(A_t)
data = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=N)

# =============================================================================
# 6. Final Output: Print Grid Search Results and Save to CSV
# =============================================================================
results_df = pd.DataFrame(results)
print("\nGrid Search Results:")
print(results_df)

# Save only the selected columns to CSV.
selected_columns = ["alpha", "beta", "best_loss", "edge_homophily_ratio",
                    "edge_homophily_ratio_thresholded", "num_edges_thresholded", 
                    "avg_edge_weight_thresholded", "louvain_modularity_score"]
results_df[selected_columns].to_csv("cvx_opt_results.csv", index=False)
print("\nSaved grid search results to 'cvx_opt_results.csv'.")

# =============================================================================
# 7. Graph Normalization and Seaborn Plotting
# =============================================================================
A_thresh_np = A_thresh.detach().cpu().numpy()

A_norm = normalize_adjacency(A_thresh_np)

# =============================================================================
# 7a. Group tickers by sector, reorder, and record boundaries.
# =============================================================================
sector_groups = defaultdict(list)
for ticker in ALL_TICKERS:
    sector = BBG_SECTORS.get(ticker, "Unknown")
    sector_groups[sector].append(ticker)

for sector in sector_groups:
    sector_groups[sector].sort()

ordered_sectors = sorted(sector_groups.keys())
ordered_tickers = []
sector_boundaries = []
current_index = 0
for sector in ordered_sectors:
    tickers_in_sector = sector_groups[sector]
    ordered_tickers.extend(tickers_in_sector)
    sector_boundaries.append((current_index, current_index + len(tickers_in_sector)))
    current_index += len(tickers_in_sector)

ticker_to_index = {ticker: i for i, ticker in enumerate(ALL_TICKERS)}
new_indices = [ticker_to_index[ticker] for ticker in ordered_tickers]
A_ordered = A_norm[np.ix_(new_indices, new_indices)]

# =============================================================================
# 7b. Define function to draw a curly brace with label.
# =============================================================================
def draw_curly_brace(ax, x, y0, y1, label, color='black', lw=2, fontsize=12):
    t = np.linspace(0, np.pi, 100)
    dx = 0.3  
    brace_x = x + dx * np.sin(t)
    brace_y = y0 + (y1 - y0) * (1 - np.cos(t)) / 2.0
    ax.plot(brace_x, brace_y, color=color, lw=lw)
    ax.text(x - 0.4, (y0 + y1) / 2, label, color=color, fontsize=fontsize,
            va='center', ha='right')

# =============================================================================
# 7c. Plot the heatmap with boxes and curly braces.
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

for (start, end) in sector_boundaries:
    rect = patches.Rectangle((start, start), end - start, end - start,
                             fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

for (start, end), sector in zip(sector_boundaries, ordered_sectors):
    draw_curly_brace(ax, -2.0, start, end, sector, color='black', lw=2, fontsize=12)

plt.title("Normalized Thresholded Adjacency Matrix Grouped by Sector")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()