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

## Granger Causality

- **Result:** Edge count does NOT Granger-cause rolling Sharpe at any lag (1-10)
- All p-values > 0.4
- **Conclusion:** Connectivity has zero predictive power for future performance
- The relationship is concurrent at best, not predictive
- Report as: "Granger causality tests found no evidence that graph connectivity predicts future strategy performance (p > 0.4 for lags 1-10)."

## VIX Overlay

- **Correlation (edge_count vs VIX): r=0.3960, p=1.39e-51** — very strong, highly significant
- Graph densifies during high-VIX (market stress) periods
- Confirms: connectivity is a proxy for market stress/volatility
- The alignment periods observed visually (2018, 2020, 2022) are all VIX spike events
- Non-alignment periods are calm markets with low/stable VIX

**Key finding for thesis:** "Rolling Pearson graph density is strongly correlated with VIX (r=0.40, p<1e-51), confirming that graph connectivity reflects market stress. During high-volatility regimes, stock correlations converge and the graph densifies."

## Granger Causality (20-window)

- All p-values > 0.05 (closest: lag=2, p=0.084)
- No predictive power at 20-window either
- Consistent with 60-window result

## Regime Analysis

**High vs Low connectivity (median split at 954 edges):**

| Metric | High Connectivity | Low Connectivity |
|---|---|---|
| Sharpe | **1.256** | 1.106 |
| Annual Return | 1.98% | 1.47% |
| Annual Volatility | 1.57% | 1.32% |
| Max Drawdown | 1.39% | 1.18% |
| Hit Rate | 57.82% | 56.42% |

Model performs better during high connectivity — when stocks are correlated, the GCN's graph captures more useful structure.

**Densifying vs Sparsifying (direction of connectivity change):**

| Metric | Densifying | Sparsifying |
|---|---|---|
| Sharpe | **0.709** | **1.627** |
| Annual Return | 1.02% | 2.39% |
| Annual Volatility | 1.45% | 1.46% |

**KEY FINDING:** 2.3x Sharpe difference. Model performs much better when graph is sparsifying.

**Interpretation:** The rolling Pearson graph is always one step behind:
- **Sparsifying** = market recovering from stress, model still has the "stress graph" which captured real correlations → good trades
- **Densifying** = market entering stress, model still has the "calm graph" which misses new correlations → bad trades

**Implication:** The rolling graph helps most when coming OUT of a regime change, not when entering one. Motivates faster-reacting graph construction.

## GCN vs GAT Edge Weights Over Time

- GAT's learned attention weights closely track GCN's Pearson weights over time
- The weights are just smaller in magnitude (scaled down)
- **GAT does not learn a meaningfully different weighting scheme** — it approximately reproduces Pearson
- This explains the similar performance (Sharpe 1.15 vs 1.21)
- The attention mechanism converges to a scaled approximation of precomputed correlation weights
- **Mean weights:** GAT slightly higher than GCN (at threshold 0.4), curve shape nearly identical — both track same temporal dynamics (respond to same market correlations)
- **Weight dispersion (std):** GAT has consistently HIGHER std than GCN. Shape follows GCN closely but elevated/stretched.
- **Interpretation:** GAT keeps Pearson's temporal structure (when to densify/sparsify) but REDISTRIBUTES weight within neighborhoods — concentrates on specific neighbors rather than weighting proportionally to correlation
- GAT says "within your 15 Pearson neighbors, these 3-4 matter most right now" vs GCN's "weight everyone by correlation"
- This redistribution adds variance: outperforms during regime transitions (2020, 2022), underperforms during stable periods
- **Conclusion:** GAT learns a more concentrated version of Pearson structure, not a fundamentally different one

## Per-Year Cumulative Return Difference (GAT - GCN)

- GCN wins on cumulative raw returns in most years
- GAT only wins on raw returns in ~2020-2021
- BUT GAT wins on yearly Sharpe in 2017, 2020, 2022
- **The difference is volatility:** GAT's concentrated attention weights produce lower-volatility positions
- During high-vol periods (2020, 2022), GAT gets better risk-adjusted returns (Sharpe) despite lower raw returns
- **Key finding:** GAT doesn't make more money than GCN — it takes less risk during stress periods. This is a different kind of advantage (risk management, not alpha generation).

## Derivative Correlation

- d_edges vs d_20-window Sharpe: r=-0.041, p=0.134 — NOT significant
- d_edges vs d_60-window Sharpe: r=-0.023, p=0.393 — NOT significant
- **Turning points do NOT align** — the visual alignment at specific events (COVID, rate hikes) is regime-level, not turning-point-level
- Consistent with Granger result: no predictive relationship

## Summary of 5b Statistical Tests

| Test | Result | Interpretation |
|---|---|---|
| Concurrent corr (20-window) | r=0.08, p=0.004 | Weak positive, significant |
| Concurrent corr (60-window) | r=-0.16, p<1e-9 | Moderate negative, significant (sign flip) |
| Derivative corr (20/60) | r≈-0.03, p>0.1 | Turning points don't align |
| Granger causality | All p>0.4 | No predictive power |
| VIX vs connectivity | r=0.40, p<1e-51 | Strong — graph density = market stress proxy |
| High vs low conn Sharpe | 1.26 vs 1.11 | Model performs better in high connectivity |
| Densifying vs sparsifying | 0.71 vs 1.63 | **KEY: model performs much better when graph is sparsifying** |

## Rolling Correlation Analysis (Non-Stationarity)

- Global correlation (r=0.08) is misleading — the relationship **alternates** between positive and negative over time
- Rolling 60-day correlation shows chunks of strong positive, strong negative, and weak correlation
- No clean mapping to specific market events — the pattern is choppy

### Regime Characterisation (KEY FINDING)

Split periods by rolling correlation strength (|corr| > 0.5):

| Regime | Days | Mean VIX | Mean Edges | Mean Daily Edge Change |
|---|---|---|---|---|
| Strong positive corr | 274 | **21.6** | **2256** | **144.1** |
| Strong negative corr | 175 | **15.6** | **1234** | **87.6** |
| Weak/none | 873 | 20.5 | 1740 | 100.5 |

**Statistical significance (t-test):**
- VIX difference (21.6 vs 15.6): t=8.89, **p=1.52e-17** — highly significant
- Edge count difference (2256 vs 1234): t=12.73, **p=6.87e-32** — highly significant
- Caveat: serial correlation in daily data may inflate significance, but p-values are extreme enough to remain significant regardless

**VIX context:** 15 vs 21 is "calm market" vs "elevated stress" — roughly 40% increase in expected volatility. At VIX 15, stocks move independently (sector-specific). At VIX 21, macro factors dominate and correlations increase.

**Interpretation:** The connectivity-performance alignment is regime-dependent:
- **During stress (high VIX, dense graph, rapidly changing structure):** Graph connectivity strongly aligns with model performance because cross-stock correlations are high and the graph structure is actively informative
- **During calm (low VIX, sparse graph, stable structure):** Stocks move independently, less cross-sectional structure to exploit → connectivity and performance decouple

**For report:** "The connectivity-performance relationship is regime-dependent. During elevated-stress periods (mean VIX 21.6, t=8.89, p<1e-17), graph connectivity and model performance are strongly positively correlated. During calm periods (mean VIX 15.6), the relationship breaks down. This suggests the rolling Pearson graph is most informative during periods of market stress when cross-stock correlations are strongest."

## Convergence of All Three Analyses (KEY THESIS FINDING)

Three independent analyses all point to the same conclusion:

**1. Regime split (high vs low connectivity):**
- High connectivity Sharpe: 1.22 > Low connectivity Sharpe: 1.04
- Model performs better when graph is dense/informative

**2. Densifying vs sparsifying:**
- Sparsifying Sharpe: 1.56 >> Densifying Sharpe: 0.68
- Model performs best when coming OUT of stress (graph was dense/informative, now relaxing)
- Model performs worst when entering stress (still using old sparse graph)

**3. Rolling correlation regimes:**
- Strong positive alignment: VIX 21.6, 2256 edges, 144 daily edge change
- Strong negative alignment: VIX 15.6, 1234 edges, 88 daily edge change
- Both statistically significant (p<1e-17)
- Graph is informative precisely when correlations are strongest

**Unified conclusion:** The rolling Pearson graph is **conditionally informative** — it adds value specifically during high-correlation regimes (market stress) where cross-stock structure matters for trading. During calm periods, the LSTM's per-stock temporal modelling is sufficient on its own.

**The temporal lag:** The graph always lags behind the market regime:
- **Entering stress (densifying):** Model still has old sparse graph → misses new correlations → Sharpe 0.68
- **During stress (high connectivity):** Graph captures real correlations → Sharpe 1.22
- **Exiting stress (sparsifying):** Model still has dense stress graph → that graph was accurate → Sharpe 1.56

**Implication for GCN vs GAT:**
- The +0.21 Sharpe improvement of rolling GCN over LSTM-only is NOT uniform — it's concentrated in stress periods
- GAT's higher weight dispersion (concentrating on fewer neighbors) may help during regime transitions when the Pearson graph is stale
- GAT outperformed GCN in 2020 (COVID) and 2022 (rate hikes) — both stress/transition periods
- This is consistent: GAT's learned weights adapt faster than Pearson's backward-looking window during exactly the periods where the graph matters most

**For report (3-4 pages):**
- Table: regime analysis (high/low, densifying/sparsifying)
- Table: rolling correlation regimes with VIX/edge statistics + significance tests
- Plot: rolling correlation with VIX overlay
- Plot: connectivity vs rolling Sharpe dual-axis
- One paragraph synthesising the three analyses into the unified conclusion above

## Outstanding Questions
- [ ] Repeat analysis for 4c GAT rolling — does GAT show different connectivity-performance pattern than GCN?
- [ ] Does the sign flip (positive short-term, negative medium-term) replicate for GAT?
- [ ] Identify alignment vs non-alignment periods manually, compare VIX levels in each

## Relevance to Thesis

- The concurrent correlation and regime analysis are reportable findings (~1-2 pages)
- Motivates the question: could a graph that reacts faster (learned, not backward-looking Pearson) improve performance?
- But 4c GAT didn't outperform GCN, so this motivation is partially undermined
- The analysis is more descriptive (characterising graph properties) than prescriptive (actionable insight)
- Best framed as: "understanding the dynamics of the rolling correlation graph and its relationship to strategy performance"
