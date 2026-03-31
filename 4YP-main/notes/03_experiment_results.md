# Experiment Results Summary

## Main Comparison Table (all 80/20 split)

| Model | Residual | Sharpe | vs LSTM | Attention Entropy Ratio | Attention Structure |
|---|---|---|---|---|---|
| LSTM-only | N/A | 1.00 | baseline | N/A | N/A |
| GCN static | Yes | 1.10 | +0.10 | N/A | Fixed Pearson |
| GCN rolling | Yes | 1.21 | +0.21 | N/A | Rolling Pearson |
| v2 (with residual) | Yes | 1.06 | +0.06 | 0.992 | Uniform (GAT bypassed) |
| 4a per-ts (no res, same HPs) | No | 0.98 | -0.02 | 1.000 | Uniform |
| 4a per-ts (no res, tuned) | No | 0.58-0.73 | -0.42 to -0.27 | 1.000 | Uniform, overfitting |
| 4b equity (no res) | No | 0.94 | -0.06 | 1.000 | Uniform |
| 4b straddle (no res) | No | 0.40 | -0.60 | 0.152 | Hub-dominant (column structure) |
| 4b straddle (no res, scaled) | No | 0.84 | -0.16 | 0.229 | Hub-dominant |
| **4c rolling (no res)** | No | 1.0 | 0.00 | 0.86 (vs nbrs) | Non-uniform within Pearson mask |
| **4c rolling (with res)** | **Yes** | **1.15** | **+0.15** | **0.92 (vs nbrs)** | **Non-uniform within Pearson mask** |

## Key Findings

### 1. Unconstrained GAT cannot learn meaningful attention (4a, 4b)
- 32-dim per-timestep features (4a): uniform attention regardless of residual/hyperparams
- 20-dim equity returns (4b-equity): uniform attention
- 200-dim straddle features (4b-straddle): sharp but hub-dominant (GE, BK, AMGN, INTU, EMR, ABT)
- Hub stocks span multiple sectors -- no sector awareness (intra/inter ratio 1.13)
- **Root cause:** ~57 independent training windows insufficient for 88x88 attention

### 2. Pearson mask enables non-uniform attention (4c)
- First experiment with both meaningful attention AND reasonable Sharpe
- Entropy 2.93 vs uniform-over-neighbors 3.40 (ratio 0.86 at threshold 0.4)
- Attention differentiates within Pearson neighborhood
- With residual: Sharpe 1.148, close to rolling GCN's 1.208

### 3. Residual consistently helps Sharpe
- v2 with res: 1.06 vs 4a without: 0.98 (same HPs, controlled ablation)
- 4c with res: 1.15 vs 4c without: 1.0
- Residual preserves per-stock LSTM signal

### 4. Dynamic graph >> static graph
- Static GCN: 1.10 vs Rolling GCN: 1.21 (+0.11)
- The time-varying structure matters more than the edge weight method

### 5. 4c GAT outperforms GCN in volatile years

| Year | 4c GAT (th=0.4) | GCN Rolling (th=0.5) | Winner |
|---|---|---|---|
| 2017 | **3.17** | 2.89 | GAT |
| 2018 | 0.16 | **0.71** | GCN |
| 2019 | 1.18 | **2.17** | GCN |
| 2020 | **1.05** | 0.23 | **GAT** (COVID) |
| 2021 | 0.68 | **1.03** | GCN |
| 2022 | **1.82** | 1.64 | **GAT** (rate hikes) |

**Hypothesis:** GAT's learned weights adapt better during regime changes. GCN's fixed Pearson weights are more stable in normal periods.

## Threshold Comparison (TODO)

Need to run both GCN and 4c at same thresholds [0.3, 0.4, 0.5, 0.6] for fair comparison.
Currently: 4c at 0.4, GCN at 0.5 -- different sparsity levels.

## Score Scaling Effect

| Model | No Scaling | With Scaling |
|---|---|---|
| 4a (32-dim input) | Entropy 1.0 | Entropy 1.0 (scaling hurts -- makes small scores smaller) |
| 4b straddle (200-dim) | Entropy 0.152, Sharpe 0.4 | Entropy 0.229, Sharpe 0.84 |
| 4b equity (20-dim) | Entropy 1.0 | TBD |
