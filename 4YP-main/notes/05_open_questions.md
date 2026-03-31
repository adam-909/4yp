# Open Questions & TODO

## Unresolved Design Questions

### Train/Valid Split Ratio
- Previous student's LSTM used 60/40, graph models use 80/20
- LSTM performs slightly better with 60/40 (more reliable early stopping)
- Decision: use 80/20 for all for consistency. Note sensitivity in report.
- **Ask supervisor** for guidance (paragraph drafted)

### Threshold Mismatch
- 4c GAT ran at threshold=0.4, GCN rolling at threshold=0.5
- Need to run both at same thresholds for fair comparison
- **TODO:** Threshold sweep [0.3, 0.4, 0.5, 0.6] for both models

### Attention Extraction Bug
- When `scale_scores=True`, the `extract_attention_weights_v2()` function does NOT apply scaling
- Extracted attention doesn't match what model actually uses during training
- **TODO:** Fix extraction function or note discrepancy

## Experiments Still To Run

### High Priority
- [ ] Threshold sweep: both GCN rolling and 4c GAT at [0.3, 0.4, 0.5, 0.6]
- [ ] 4e: Sector-constrained GAT (block-diagonal adjacency from GICS sectors)
- [ ] Multi-seed runs for final comparison table (seeds 40, 41, 42)
- [ ] Save all results via `save_experiment_results()` to FINAL_RESULTS/

### Medium Priority  
- [ ] Ablation: 4c with scale_scores
- [ ] Ablation: 4c with GATv1 instead of GATv2
- [ ] Ablation: hidden size sweep {16, 32, 64} for 4c

### Lower Priority / Future Work
- Entropy regularization in loss
- Top-k sparsification for unconstrained GAT
- Concatenate prev window features with current LSTM hidden states (coupled 4b)
- Trajectory model with sliding windows
- Different lookback periods for 4b

## Notebook-to-Experiment Mapping

| Experiment | Notebook | Status |
|---|---|---|
| 1_lstm_only | `1_lstm_only.ipynb` | Created, needs runs |
| 2_GCN_static | `lstm_gcn_static.ipynb` | Results exist, needs save to FINAL_RESULTS |
| 3_GCN_rolling | `lstm_gcn_rolling_pearson.ipynb` | Results exist, needs save to FINAL_RESULTS |
| 4_diagnostic | `lstm_gat_e2e_v2.ipynb`, `lstm_gat_e2e_v2_prev_window.ipynb` | Done |
| 4a | `4a_GATv2_per_ts_nores.ipynb` | Done |
| 4b | `4b_GATv2_prev_window_nores.ipynb` | Done |
| 4c | `4c_GATv2_rolling_pearson.ipynb` | Running, best result so far |
| 4e | `4e_GATv2_sector.ipynb` | Not created yet |
| 5 | `interpretability_analysis.ipynb` | Not created yet |
| Comparison | `experiment_comparison.ipynb` | Not created yet |

## Key Files Created

| File | Purpose |
|---|---|
| `gml/experiment_utils.py` | Standardized save/load for all experiments |
| `gml/graph_attention_v2.py` | GATv2 layers + all model builders (v2, v3, rolling) |
| `notebooks/4a_GATv2_per_ts_nores.ipynb` | Exp 4a |
| `notebooks/4b_GATv2_prev_window_nores.ipynb` | Exp 4b |
| `notebooks/4c_GATv2_rolling_pearson.ipynb` | Exp 4c |
| `notebooks/1_lstm_only.ipynb` | Standalone LSTM baseline |
