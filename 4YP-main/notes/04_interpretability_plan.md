# Interpretability Analysis Plan

## What We Need Saved for Analysis

| File | Content | Models |
|---|---|---|
| `predictions.npy` | Raw model output | All |
| `attention_weights.npy` | (N, heads, 88, 88) | GAT models |
| `adjacency.npy` | Static or rolling (N, 88, 88) | GCN models |
| `daily_returns.csv` | Portfolio returns | All |
| `test_dates.npy` | Calendar dates | All |
| `graph_stats.csv` | Edge count, entropy, degree over time | Graph models |

## 5a: Graph Structure vs Sector Membership

**Metrics per model:**
- Intra/Inter-sector edge ratio: `mean(within-sector weights) / mean(between-sector weights)`
- Modularity (Q): Newman modularity with GICS sectors as communities
- NMI: Louvain community detection vs GICS sectors
- Sector-level heatmap: aggregate edges by sector -> 10x10 matrix

**Key comparison:** Do Pearson graphs reflect sector structure? Does GAT attention?

**4b-straddle finding:** Hub-dominant attention (GE, BK, AMGN, INTU, EMR, ABT across mixed sectors). Intra/inter ratio only 1.13 -- no sector awareness.

## 5b: Graph Dynamics & Regime Analysis

**Connectivity vs performance:**
- corr(edge_count, VIX): do graphs densify during crises?
- corr(edge_count, rolling_Sharpe): concurrent relationship
- corr(edge_count_t, rolling_Sharpe_{t+20}): predictive relationship
- Median split: high vs low connectivity Sharpe

**Regime events to highlight:**
- COVID: March 2020
- Rate hikes: 2022
- GAT outperformed GCN in both these periods

## 5c: Attention Asymmetry (GAT only)

- GAT attention is asymmetric: A[i,j] != A[j,i]
- GCN fundamentally cannot model this (symmetric adjacency)
- Mean asymmetry: `mean(|A[i,j] - A[j,i]|)` across all pairs
- Top-10 most asymmetric pairs -- do they reflect known lead-lag?
- Asymmetry by sector: inter vs intra-sector

## 5d: Position Analysis Across Models

- Pairwise position correlation matrix across all models
- If GCN and GAT corr > 0.9, graph layer barely affects trades
- Turnover: `mean(|pos_t - pos_{t-1}|)` per model
- Position distribution histograms per model
- Year-by-year: which stocks did GAT and GCN trade differently during 2020/2022?
