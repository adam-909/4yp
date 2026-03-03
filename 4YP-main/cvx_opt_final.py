import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from collections import defaultdict
import networkx as nx
import torch
from torch_geometric.data import Data
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
from settings.default import ALL_TICKERS, BBG_SECTORS

GEPHI_OUTPUT = True

# =============================================================================
# 1. Parameters and Data Preparation
# =============================================================================
# Define grid search values for hyperparameters.
alpha_grid = [0.1, 1, 10, 100]
beta_grid  = [0.01, 0.1, 1, 10]

d_values  = [252, 504, 756, 1008, 1260]  # Multiple lookback periods
tau       = 0.0         # Threshold for edge weights (applied after training)
num_epochs = 10000
min_delta  = 1e-2
patience   = 25

feature_columns = [
    'norm_daily_return',
    'norm_monthly_return',
    'norm_quarterly_return',
    'norm_biannual_return',
    'norm_annual_return',
    'macd_2_8',
    'macd_4_16',
    'macd_8_32',
    'log_moneyness',
    'time_to_expiry',
]

# Read the feature CSV.
df = pd.read_csv(r"C:\Users\Sean\Documents\gml-master\data\straddle_features\features.csv")
df['date'] = pd.to_datetime(df['date'])


# =============================================================================
# 2. Construct the Flattened Feature Matrix for Each Ticker and Ensemble over d_values
# =============================================================================
def optimize_for_d(d, alpha_val, beta_val):
    """
    Optimize a graph for a given lookback period d using hyperparameters alpha and beta.
    """
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
    # Normalize column-wise
    V_mean = V_t.mean(dim=0, keepdim=True)
    V_std  = V_t.std(dim=0, keepdim=True) + 1e-8
    V_t = (V_t - V_mean) / V_std

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
            print(f"d={d}, epoch {epoch}, loss={current_loss}")
        if epochs_no_improve >= patience:
            print(f"[d={d}] Early stopping at epoch {epoch} with best loss {best_loss}")
            break
    if best_A_t is not None:
        with torch.no_grad():
            A_t.data = best_A_t
    # Threshold the matrix.
    A_thresh = A_t.data.clone()
    A_thresh[A_thresh < tau] = 0.0

    # Compute statistics.
    ehr = edge_homophily_ratio(A_t, ALL_TICKERS, BBG_SECTORS)
    A_np = A_thresh.detach().cpu().numpy()
    num_edges = 0
    total_edge_weight = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if A_np[i, j] > 0:
                num_edges += 1
                total_edge_weight += A_np[i, j]
    avg_edge_wt = total_edge_weight / num_edges if num_edges > 0 else 0.0
    mod_score = compute_louvain_modularity(A_thresh, ALL_TICKERS)
    stats = {
        "d": d,
        "final_loss": best_loss,
        "edge_homophily": ehr,
        "num_edges": num_edges,
        "avg_edge_weight": avg_edge_wt,
        "louvain_modularity": mod_score
    }
    return A_t.data.clone(), stats


# Define objective and helper functions:
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
    n = len(tickers)
    for i in range(n):
        for j in range(i+1, n):
            w = A_np[i, j]
            total_weight += w
            if sectors[tickers[i]] == sectors[tickers[j]]:
                same_sector_weight += w
    return same_sector_weight / total_weight if total_weight > 0 else 0.0

def normalize_adjacency(A):
    eps = 1e-8
    row_sum = A.sum(axis=1)
    row_sum[row_sum == 0] = eps
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

def compute_louvain_modularity(A_thresh, tickers):
    A_np = A_thresh.detach().cpu().numpy()
    A_norm = normalize_adjacency(A_np)
    df_norm = pd.DataFrame(A_norm, index=tickers, columns=tickers)
    G_norm = nx.from_pandas_adjacency(df_norm)
    comms = louvain_communities(G_norm, resolution=20)
    return modularity(G_norm, comms, weight='weight')


# =============================================================================
# 3. Grid Search over alpha and beta
# =============================================================================
for alpha_val in alpha_grid:
    for beta_val in beta_grid:
        print(f"\nRunning grid search for alpha = {alpha_val} and beta = {beta_val}")
        
        # Run optimization for each lookback period (d_value) using current alpha and beta.
        A_list = []
        stats_list = []
        for d_val in d_values:
            A_final, stats_dict = optimize_for_d(d_val, alpha_val=alpha_val, beta_val=beta_val)
            A_list.append(A_final)
            stats_list.append(stats_dict)

        df_stats = pd.DataFrame(stats_list)
        print("\nStats for each d:")
        print(df_stats)

        # Ensemble adjacency: average of all A^(k) for different lookback periods.
        K = len(A_list)
        N = A_list[0].shape[0]
        A_ensemble = torch.zeros(N, N, dtype=torch.float32)
        for A_k in A_list:
            A_ensemble += A_k
        A_ensemble /= K

        # =============================================================================
        # 4. Apply Graph Normalization to the Ensemble
        # =============================================================================
        A_ensem5ble_np = A_ensemble.cpu().numpy()
        row_sum = A_ensemble_np.sum(axis=1)
        row_sum[row_sum == 0] = 1e-8
        D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sum))
        # Here, we are normalizing the ensemble adjacency matrix.
        A_ensemble_norm = D_inv_sqrt @ A_ensemble_np @ D_inv_sqrt

        # =============================================================================
        # 5. Plot the Final Ensemble Heatmap
        # =============================================================================
        # Reorder tickers by sector.
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
            sector_boundaries.append((current_idx, current_idx+len(tickers_in_sec)))
            current_idx += len(tickers_in_sec)
        
        ticker_to_index = {t: i for i, t in enumerate(ALL_TICKERS)}
        new_indices = [ticker_to_index[t] for t in ordered_tickers]
        A_ordered = A_ensemble_norm[np.ix_(new_indices, new_indices)]
        
        # plt.figure(figsize=(12,12))
        # ax = sns.heatmap(A_ordered,
        #                  cmap="YlOrRd",
        #                  square=True,
        #                  xticklabels=ordered_tickers,
        #                  yticklabels=ordered_tickers,
        #                  cbar=True)
        # for (start, end) in sector_boundaries:
        #     rect = patches.Rectangle((start, start), end-start, end-start,
        #                              fill=False, edgecolor='black', linewidth=2)
        #     ax.add_patch(rect)
        # plt.title(f"Ensemble Adjacency (Normalized)\nfor d in {d_values} (alpha={alpha_val}, beta={beta_val})")
        # plt.xticks(rotation=90)
        # plt.yticks(rotation=0)
        # plt.tight_layout()
        # plt.show()

        # =============================================================================
        # 6. Gephi Output: Vary Louvain Resolution Until Number of Communities Matches Bloomberg Sectors
        # =============================================================================
        if GEPHI_OUTPUT:
            df_norm = pd.DataFrame(A_ensemble_norm, index=ALL_TICKERS, columns=ALL_TICKERS)
            G_temp = nx.from_pandas_adjacency(df_norm)
            
            # Number of Bloomberg sectors.
            bbg_comms = set(BBG_SECTORS[ticker] for ticker in ALL_TICKERS)
            num_bbg = len(bbg_comms)
            
            chosen_resolution = None
            chosen_partition = None
            chosen_mod_score = None
            
            # Try a range of resolution values.
            res_range = np.linspace(0.1, 30, 50)
            for res in res_range:
                communities = louvain_communities(G_temp, resolution=res)
                num_comms = len(communities)
                if num_comms == num_bbg:
                    chosen_resolution = res
                    chosen_mod_score = modularity(G_temp, communities, weight='weight')
                    partition = {}
                    for comm_index, comm in enumerate(communities):
                        for node in comm:
                            partition[str(node)] = comm_index
                    chosen_partition = partition
                    print(f"Found resolution {res} yielding {num_comms} communities matching Bloomberg sectors.")
                    break
            if chosen_resolution is None:
                print("Desired number of communities not achieved; using last resolution.")
                chosen_resolution = res
                partition = {}
                for comm_index, comm in enumerate(communities):
                    for node in comm:
                        partition[str(node)] = comm_index
                chosen_partition = partition
                chosen_mod_score = modularity(G_temp, communities, weight='weight')
            
            # Save Gephi-compatible edge list.
            edge_list = []
            for u, v, data in G_temp.edges(data=True):
                weight = data.get('weight', 1)
                edge_list.append({'Source': str(u), 'Target': str(v), 'Weight': weight})
            edges_df = pd.DataFrame(edge_list)
            
            # Save community assignments (node classes).
            nodes_list = [{'Id': str(node), 'Community': comm} for node, comm in chosen_partition.items()]
            nodes_df = pd.DataFrame(nodes_list)
            
            gephi_output_dir = os.path.join("gephi", "cvx_opt", "ensemble", f"{alpha_val}_{beta_val}_edges", "communities")
            os.makedirs(gephi_output_dir, exist_ok=True)
            edges_file = os.path.join(gephi_output_dir, "gephi_edges.csv")
            nodes_file = os.path.join(gephi_output_dir, "gephi_nodes.csv")
            
            edges_df.to_csv(edges_file, index=False)
            nodes_df.to_csv(nodes_file, index=False)
            
            print(f"Saved Gephi edge list to {edges_file}")
            print(f"Saved Gephi community assignments to {nodes_file}")

        # =============================================================================
        # 7. Print Ensemble Statistics and Save Normalized Adjacency Matrix
        # =============================================================================
        print("\nEnsemble Statistics:")
        print("Average final loss (ensemble not computed here, stats per d):")
        print(df_stats)
        example_ehr = edge_homophily_ratio(torch.tensor(A_ensemble_np), ALL_TICKERS, BBG_SECTORS)
        print(f"Ensemble Edge Homophily (computed from one d run as example): {example_ehr}")

        output_dir = os.path.join("data", "graph_structure", "cvx_opt")
        os.makedirs(output_dir, exist_ok=True)
        # Save the normalized adjacency matrix!
        output_file = os.path.join(output_dir, f"{alpha_val}_{beta_val}_cvx.csv")
        adjacency_df = pd.DataFrame(A_ensemble_norm, index=ALL_TICKERS, columns=ALL_TICKERS)
        adjacency_df.to_csv(output_file)
        print(f"Saved normalized adjacency matrix to {output_file}")
