# GAT Diagnosis: Why GAT Underperformed GCN

## Original Issues Found

| # | Issue | Impact |
|---|-------|--------|
| 1 | LSTM hidden=10 -> GAT gets 10-dim features -> attention degenerates to near-uniform | Critical |
| 2 | GATv1 attention not truly pairwise (`f(i) + g(j)`, ranking of j independent of i) | High |
| 3 | Per-timestep GAT: attention fluctuates across timesteps (temporal incoherence) | High |
| 4 | Single `DROPOUT_RATE=0.5` for both LSTM and attention -> attention too noisy | High |
| 5 | `GAT_UNITS=8` with head averaging -> final dim=8, half of GCN's 16 | Medium |
| 6 | `LayerNormalization` after GAT (GCN has none) -> suppresses cross-stock variance | Medium |
| 7 | `clipnorm=0.01` starves attention parameter gradients | Medium |
| 8 | GCN gets free domain knowledge from precomputed correlation graph; GAT must learn from scratch | Structural |

## GATv1 vs GATv2

- **GATv1:** `score(i,j) = LeakyReLU(a_src^T * W*h_i + a_dst^T * W*h_j)` -- the ranking of j is the SAME regardless of who i is (global popularity contest)
- **GATv2:** `score(i,j) = a^T * LeakyReLU(W_src*h_i + W_dst*h_j)` -- the LeakyReLU creates nonlinear interactions between i and j BEFORE collapsing to scalar. Ranking of j DEPENDS on who i is.
- The key swap: nonlinearity and projection to scalar are swapped in order

## Attention Entropy

- Measures how spread out each node's attention is across neighbors
- High entropy (~log(88) = 4.48): uniform attention, no meaningful graph learned
- Low entropy: focused attention on specific neighbors
- Entropy alone doesn't tell if discrimination is meaningful -- need sector alignment analysis too

## Score Scaling (sqrt normalization)

- Without normalization, softmax input magnitude depends on dimension of vectors
- `units=8` -> scores ~sqrt(8) = 2.8 -> softmax near-uniform
- `units=32` -> scores ~sqrt(32) = 5.7 -> some differentiation
- 200-dim input -> large scores -> softmax saturates -> hub-dominant peaks
- 20-dim input -> small scores -> softmax near-uniform
- Fix: `scores = scores / sqrt(units)` before softmax (standard in Transformers, not in original GAT)
- Implemented as toggleable `scale_scores` parameter in all layers

## Residual Connection Problem

The v2 models have a residual path: `LSTM -> Dense(16) -> Add with GAT output -> Dense(1) -> position`

- The model achieves good Sharpe (1.06) through the residual path alone (LSTM -> Dense -> Dense -> tanh)
- The GAT layer becomes a no-op -- uniform attention means it just adds a constant offset
- Removing residual: Sharpe drops slightly (~0.98) but attention still doesn't sharpen
- **Root cause:** Not the residual itself, but insufficient features/samples for attention to learn

## Key Insight: Bias-Variance Tradeoff

- **GCN + precomputed graph** = high bias (fixed graph), low variance (nothing to learn)
- **GAT** = low bias (any graph), high variance (must learn 88x88 from noise)
- With ~57 independent training windows, GAT can't learn meaningful structure
