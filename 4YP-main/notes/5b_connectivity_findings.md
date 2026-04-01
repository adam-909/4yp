# 5b: Graph Connectivity vs Trading Performance — Findings & Discussions

## Setup

- **Model:** LSTM-GCN Rolling Pearson (lookback=20, threshold=0.4, equity returns)
- **Test period:** 2017-2023 (~1400 sliding windows)
- **Connectivity:** Edge count per window from rolling Pearson adjacency matrices
- **Performance:** Rolling Sharpe computed from per-window returns

## Rolling Sharpe Computation

Two approaches verified to be equivalent:
1. `daily_returns.rolling(N).mean() / daily_returns.rolling(N).std() * sqrt(252)` (from daily returns)
2. `pd.Series(window_returns).rolling(N).apply(sharpe)` (from per-window returns, as in `connectivity_analysis.py`)

Both produce the same values because test uses stride=1 (one window per day). Small numerical differences in correlation (r=0.0905 vs r=0.0776) are due to date alignment / dropna differences, not computation.

**Decision:** Use the `connectivity_analysis.py` approach (window returns + edge_counts as raw arrays) — simpler, no date alignment step that can silently drop rows.

## Concurrent Correlation Results

| Rolling Window | Pearson r | p-value | Significant? |
|---|---|---|---|
| 20-window | 0.0776 | 3.92e-03 | Yes |
| 60-window | -0.1623 | 2.25e-09 | Yes |

**Key finding: sign flip between short and medium term.**
- Short-term (20 windows): weakly positive — dense graph slightly helps immediate performance
- Medium-term (60 windows): moderately negative — dense graph associated with worse longer-term performance

**Interpretation:** High connectivity (correlation spike) may help in the immediate term (model correctly captures that stocks are moving together), but over a longer horizon the dense/stale graph hurts (the model keeps using high-correlation structure into the recovery when correlations have broken).

## Lagged Cross-Correlation

- Peak at lag = -37, r = -0.34
- Meaning: `corr(edge_count_t, sharpe_{t-37}) = -0.34`
- **Today's edge count is most correlated with Sharpe from 37 days ago**

### Discussion: Is this meaningful?

**Initial interpretation (WRONG):** "Sharpe drops lead to connectivity increases 37 days later — the Pearson graph is a lagging indicator."

**Problem with this interpretation:** The Sharpe is the MODEL's predicted trading performance, not a market event. There's no causal mechanism where the model's bad trades cause Pearson correlation to change. The model's Sharpe is a consequence of its positions, not an exogenous shock.

**More honest interpretation:** Both connectivity and model Sharpe respond to the same underlying market dynamics (volatility, regime shifts). The lag reflects different response speeds. But this is speculative and unfalsifiable.

**Conclusion:** The lagged cross-correlation doesn't add interpretive value. The peak at -37 is a statistical artifact of two time series responding to a common driver with different lags. **Skip in report or mention briefly as "no evidence of a predictive relationship."**

## What IS useful from 5b

1. **Concurrent correlation with sign flip** (short-term positive, medium-term negative)
2. **Regime analysis** (high vs low connectivity Sharpe) — TODO
3. **Visual overlay** showing alignment at specific events (COVID, rate hikes, VIX spikes)
4. **VIX overlay** — validates whether connectivity is a proxy for market stress

## Observations from Visual Analysis

From the `lstm_gcn_rolling_pearson.ipynb` connectivity plot:
- Connectivity varies between ~500-1000 edges (low) and 2500+ edges (high)
- Rolling 20-window Sharpe movement SOMETIMES aligns with connectivity
- Alignment observed at:
  - Mid 2022 (rate hikes)
  - Early/mid 2020 (COVID)
  - Early/mid 2018 (VIX spike)
- But NOT always — alignment is intermittent, not consistent

## Outstanding Questions

- [ ] What does the regime analysis (high vs low connectivity Sharpe) show?
- [ ] Does VIX correlate with connectivity? (validates graph density as stress proxy)
- [ ] Does derivative correlation (turning points) show significance?
- [ ] Repeat analysis for 4c GAT rolling — does GAT show different connectivity-performance pattern than GCN?
- [ ] Does the sign flip (positive short-term, negative medium-term) replicate for GAT?

## Relevance to Thesis

- The concurrent correlation and regime analysis are reportable findings (~1-2 pages)
- Motivates the question: could a graph that reacts faster (learned, not backward-looking Pearson) improve performance?
- But 4c GAT didn't outperform GCN, so this motivation is partially undermined
- The analysis is more descriptive (characterising graph properties) than prescriptive (actionable insight)
- Best framed as: "understanding the dynamics of the rolling correlation graph and its relationship to strategy performance"
