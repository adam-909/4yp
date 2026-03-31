# Architecture Design Decisions

## Per-Timestep vs Trajectory

- **Per-timestep:** GAT runs at each of 20 timesteps independently. LSTM weights shared across time, GAT weights shared across time. Each timestep is an independent observation for training.
- **Trajectory:** Concatenate all 20 LSTM hidden states -> one vector per stock -> GAT runs once.
- **Decision:** Per-timestep with stride=20. Gives ~100k effective training observations (57 windows x 20 timesteps x 88 stocks) while keeping windows independent.
- Trajectory with stride=20 only gets ~57 samples -- sample starvation.

## LSTM Hidden State Resets

- Hidden state resets between windows (Keras default: `stateful=False`)
- h_1 starts from zero in each window
- Within a window, h_t carries information from days 1..t
- h_1 = weak signal (1 day), h_20 = strongest (full context)
- All 20 timesteps contribute to Sharpe loss during training
- Only h_20 used at evaluation

## Previous Window Attention (4b)

- Uses raw features from PREVIOUS non-overlapping window to compute attention
- Current window goes through LSTM for message passing
- No information leakage (attention from strictly past data)
- Analogous to rolling GCN (both use past data for graph), but learned end-to-end
- **Decoupled design:** Attention parameters never see LSTM hidden states, LSTM never sees previous window features. They only interact at message passing.
- Professor noted this is unconventional -- standard GAT uses same features for attention and messages

### Attention Feature Sources (4b)

- `equity_returns` (20-dim): Fair comparison with rolling GCN (same input data)
- `straddle` (200-dim): Richer features (vol, MACD, moneyness) that Pearson never sees

## Rolling Pearson Constrained GAT (4c)

- Pearson determines WHICH edges exist (structure/mask)
- GATv2 learns the WEIGHT of each edge (from LSTM hidden states)
- Uses `DynamicMaskedGATv2Layer` -- masks non-edges to -inf before softmax
- Includes self-loops in mask
- Same data pipeline as rolling GCN (`RollingGraphModelFeatures`)
- **Why it should work:** GCN uses fixed symmetric Pearson weights. GAT can learn context-dependent, asymmetric weights -- e.g., downweight MSFT during its earnings if it's behaving unusually.

## Residual Connection

- Present in GCN and original GAT models (v2)
- Preserves per-stock LSTM signal while graph layer adds neighbor context
- Without residual + uniform attention = averaging all 88 stocks = destroys per-stock signal
- With Pearson mask (4c), no residual is viable because averaging ~15 correlated neighbors is useful
- `use_residual` parameter added to `build_lstm_gat_rolling()` as toggleable option
- 4c with residual: Sharpe 1.148. Without: ~1.0. Residual helps.

## Training Setup

- All graph models: stride=20 (non-overlapping), predict all 20 timesteps, Sharpe loss over everything
- LSTM-only: sliding windows (stride=1), all overlapping windows concatenated across tickers
- Train/valid split: 80/20 temporal split (not random) for all models
- Previous student's LSTM used 60/40 -- produces slightly better LSTM results due to more reliable early stopping
- Decision: Use 80/20 for all models for consistency. Note the sensitivity in report.
