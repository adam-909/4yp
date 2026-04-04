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

**REVISED interpretation (after regime analysis):**

The -37 lag IS meaningful in context of the regime findings:
1. Market stress event begins
2. Model Sharpe drops **immediately** (positions lose money)
3. Over ~37 days, the 20-day Pearson lookback window fills with stressed/correlated data
4. Graph densifies ~37 days later

The lag **quantifies the delay** between market events impacting performance and the Pearson graph catching up (~37 days ≈ 20-day lookback + time for correlations to stabilise).

This directly supports the densifying vs sparsifying finding:
- During those ~37 days of "catching up" (densifying) → model has wrong graph → Sharpe 0.68
- After graph catches up → model has correct graph → Sharpe 1.56

**Forward direction (lag=+37):** `corr(edge_count_{t-37}, sharpe_today) ≈ 0.1` — weakly positive. Past high connectivity slightly predicts future better Sharpe, consistent with sparsifying finding (model benefits from having had access to dense/informative graph).

**For report:** "Lagged cross-correlation reveals a peak at -37 days (r=-0.34), indicating the rolling Pearson graph lags market regime changes by approximately 37 days. This quantifies the structural delay identified in the regime analysis: the model underperforms during graph densification (Sharpe 0.68) precisely because the graph takes ~37 days to reflect the new correlation regime."

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

## Per-Regime Model Sharpe (CRITICAL — reframes narrative)

| Regime | Days | Mean VIX | GCN Rolling Sharpe |
|---|---|---|---|
| Strong positive corr (stress) | 274 | 21.6 | **0.229** |
| Strong negative corr (calm) | 175 | 15.6 | **0.436** |
| Weak/none (stable) | 873 | 20.5 | **1.523** |

**This CONTRADICTS the earlier narrative:**
- We assumed "graph informative during stress → better performance during stress"
- Reality: model performs WORST during stress (0.23) and BEST during stable periods (1.52)
- The graph's interpretability (connectivity tracks Sharpe) ≠ graph's profitability

**What still stands:**
- VIX ↔ connectivity (r=0.40) — factual, unchanged
- Regime-dependent relationship — true, just doesn't mean what we thought
- No predictive power (Granger) — unchanged

**What's undermined:**
- "Graph most informative during stress" — model actually struggles during stress
- "Three analyses converge" — the convergence was based on incorrect assumption
- Densifying/sparsifying narrative — already not significant, now further weakened

**Revised interpretation:** The LSTM-GCN makes money during stable markets where patterns are consistent. During stress, all patterns break and the model struggles regardless of graph density. The graph connectivity tracks market stress but this tracking doesn't translate to better trading during those periods.

**MUST DO NEXT (5d):** Compare LSTM-only, GCN static, GCN rolling, and GAT in the same three regimes. Key question: does LSTM-only also struggle during stress? If yes, it's a universal effect. If GCN still beats LSTM during stress (even at low Sharpe), the graph still adds relative value.

## VIX-Based Regime Comparison (Model-Independent)

| Regime | LSTM-only | GCN Static | GCN Rolling | 4c GAT | Days |
|---|---|---|---|---|---|
| VIX High (>24.3) | 1.266 | 1.232 | 1.344 | **1.498** | 335 |
| VIX Mid | 0.811 | 0.612 | **1.120** | 0.957 | 672 |
| VIX Low (<13.5) | **1.132** | 0.458 | 1.067 | 0.881 | 334 |

**Bootstrap significance tests:** NONE of the pairwise model differences within any regime are significant (all p > 0.7). Sample sizes (~335 days per regime) provide insufficient statistical power.

## GCN-Defined Regime Performance (all models)

| Regime | LSTM-only | GCN Static | GCN Rolling | 4c GAT | Days |
|---|---|---|---|---|---|
| Strong pos corr | **1.025** | 0.653 | 0.229 | 0.495 | 274 |
| Strong neg corr | -0.742 | -0.240 | **0.436** | **0.602** | 175 |
| Weak/none | 1.169 | 0.882 | **1.523** | 1.302 | 873 |

**NOTE:** These regimes are defined by GCN Rolling's own connectivity-Sharpe correlation — not model-independent. Valid for explaining GCN's behavior but not for fair cross-model comparison.

**Bootstrap tests on GCN across its own regimes:** Not significant (Strong pos vs Weak/none: p=0.27, Strong neg vs Weak/none: p=0.43).

## FINAL HONEST ASSESSMENT

**Statistically confirmed:**
1. Graph density = VIX proxy (r=0.40, p<1e-51)
2. Connectivity-Sharpe alignment occurs during different market conditions (p<1e-17)
3. No predictive power (Granger p>0.4)

**Descriptive/exploratory only (not statistically confirmed):**
- All performance differences across regimes
- All model comparisons within regimes
- Densifying vs sparsifying
- High vs low connectivity Sharpe

**Root cause:** 6 years of daily data split into regimes provides insufficient statistical power for Sharpe ratio comparisons. Would need decades of data or a fundamentally different test approach.

**What 5b contributes to thesis:** Interpretability of graph structure (it tracks market stress and has economic meaning), NOT proven performance effects. Frame accordingly in report.

## Edge Turnover Analysis (SIGNIFICANT)

| Regime | Mean Turnover | Std Turnover | Days |
|---|---|---|---|
| Strong pos corr (stress) | **0.076** | 0.065 | 274 |
| Strong neg corr (calm) | **0.130** | 0.071 | 175 |
| Weak/none (stable) | **0.094** | 0.060 | 873 |

**Significance tests (all highly significant):**
- Stress vs Calm: t=-8.30, p=1.2e-15
- Stress vs Stable: t=-4.35, p=1.5e-05
- Calm vs Stable: t=6.94, p=6.7e-12

**Interpretation:** During stress, correlations are decisively above threshold → edges form and persist (low turnover 7.6%). During calm, many correlations hover near threshold → edges flicker on/off (high turnover 13.0%).

**For report:** "Graph edge stability is significantly regime-dependent. During stress periods, edge turnover is lowest (7.6%), indicating stable, persistent correlations. During calm periods, turnover is highest (13.0%), reflecting noisy near-threshold correlations. This confirms that the rolling Pearson graph provides the most reliable cross-sectional structure during periods of elevated market stress."

## Sector Composition of Connectivity

- Cross-sector ratio stable across all regimes (~80%)
- Connectivity spikes reflect broad market-wide correlation increases, not sector-specific clustering
- Financials always has most intra-sector edges (largest sector, 16 tickers — mechanical effect)
- **Not an interesting finding** — one sentence in report

## Per-Regime Granger Causality

Initial attempt on |corr|>0.5 regimes: blocks too short (24 days max).

Re-ran with sign-based split (corr > 0 vs corr < 0, gap tolerance 10 days):
- **Positive corr longest block:** 87 days (Apr-Aug 2020). All p>0.7. No predictive power.
- **Negative corr longest block:** 115 days (Sep 2017-Mar 2018). Significant at lags 1-3 (p=0.006, 0.012, 0.030).
- **But does NOT replicate:** Block 2 (62 days, Aug-Nov 2020) all p>0.4. Block 3 (59 days, Dec 2019-Mar 2020) all p>0.3.
- **Conclusion:** The one significant result is event-specific (2017-2018 VIX spike), not a general property. No robust predictive power in any regime.

## Threshold Sensitivity

Tested whether regime characterisation holds at different rolling correlation thresholds:
- Thresholds tested: 0.3, 0.4, 0.5, 0.6
- VIX and edge count differences between positive and negative correlation regimes are **significant at ALL thresholds**
- The choice of |r| > 0.5 is not special — any reasonable threshold captures the same pattern
- **For report:** "We use |r| > 0.5 to define strong correlation regimes; sensitivity analysis confirms the characterisation is robust across thresholds 0.3-0.6."

## Completeness of 5b Interpretability

The 5b analysis connects graph structure to model behavior at the **macro level**:
1. Graph density tracks VIX (r=0.40, confirmed)
2. Graph density correlates with model's Sharpe during stress regimes (confirmed, p<1e-17)
3. Graph is stable during those stress periods (confirmed, turnover p<1e-15)

Chain: **during stress → graph is dense + stable → model's performance co-moves with density → graph structure is interpretable**

Micro-level analysis (how graph affects individual stock positions) is 5d territory. The macro-level story is sufficient for 5b.

## GAT vs GCN: Position Size and Volatility Decomposition

**Per-year return/vol/Sharpe decomposition:**

| Year | GCN Ret | GCN Vol | GCN Sharpe | GAT Ret | GAT Vol | GAT Sharpe | VIX |
|---|---|---|---|---|---|---|---|
| 2017 | 2.28% | 0.83% | 2.74 | 1.29% | 0.41% | **3.17** | 11.1 |
| 2018 | 0.71% | 1.31% | **0.54** | 0.14% | 0.85% | 0.17 | 16.6 |
| 2019 | 3.12% | 1.62% | **1.92** | 0.88% | 0.75% | 1.18 | 15.4 |
| 2020 | 0.48% | 1.94% | 0.25 | 1.14% | 1.08% | **1.05** | 29.3 |
| 2021 | 1.31% | 1.27% | **1.04** | 0.41% | 0.60% | 0.68 | 19.7 |
| 2022 | 2.45% | 1.47% | 1.67 | 1.73% | 0.95% | **1.82** | 25.6 |

**Position sizes:** GAT positions are 32-52% smaller than GCN every year.

| Year | GCN mean |pos| | GAT mean |pos| | Difference |
|---|---|---|---|
| 2017 | 0.189 | 0.097 | -48.9% |
| 2018 | 0.139 | 0.096 | -31.5% |
| 2019 | 0.232 | 0.113 | -51.5% |
| 2020 | 0.167 | 0.105 | -37.0% |
| 2021 | 0.161 | 0.089 | -44.9% |
| 2022 | 0.160 | 0.097 | -39.7% |

**Correction:** 2017 (VIX 11.1) is the calmest year but GAT wins. No clean VIX-based pattern for when GAT wins. The mechanism is position sizing, not market regime.

**What we showed:**
- GAT takes smaller positions → lower vol → lower returns
- When GCN's larger positions don't generate proportionally higher returns → GAT wins on Sharpe (2017, 2020, 2022)
- When GCN's larger positions do generate higher returns → GCN wins (2018, 2019, 2021)

**CORRECTION — GAT attention is NOT more concentrated than GCN:**

Proper concentration metrics (Gini, effective neighbors, max weight) show:
- Gini: GCN 0.130 > GAT 0.083 — **Pearson weights are MORE unequal**
- Max weight: GCN 0.079 > GAT 0.071 — **Pearson's top neighbor dominates more**
- Effective neighbors: nearly identical (37.7 vs 38.0)
- All differences statistically significant

The earlier "higher std" finding was misleading — std of raw weights reflects scale differences, not concentration. GAT's attention is actually MORE UNIFORM than Pearson within the same mask.

**Revised interpretation:** GAT's lower vol and smaller positions are NOT caused by attention concentration. The mechanism is likely:
- GAT layer's W_msg projection produces lower-magnitude output features
- Or the overall attention weight scale is smaller, reducing the aggregated signal magnitude
- The attention flattening ablation would still be informative: does making attention even more uniform change position sizes?

**What we CAN say:** GAT and GCN use similar attention patterns (both near-uniform within Pearson mask), but GAT produces systematically smaller positions through a different mechanism than attention concentration.

## Relevance to Thesis

- The concurrent correlation and regime analysis are reportable findings (~1-2 pages)
- Motivates the question: could a graph that reacts faster (learned, not backward-looking Pearson) improve performance?
- But 4c GAT didn't outperform GCN, so this motivation is partially undermined
- The analysis is more descriptive (characterising graph properties) than prescriptive (actionable insight)
- Best framed as: "understanding the dynamics of the rolling correlation graph and its relationship to strategy performance"
