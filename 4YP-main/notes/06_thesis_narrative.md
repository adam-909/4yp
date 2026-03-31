# Thesis Narrative & Story Arc

## Progressive Story

```
Baseline (no graph)
  -> "Does graph structure help?" -> Static GCN (+0.10 Sharpe)
    -> "Does adapting the graph help?" -> Rolling GCN (+0.21 Sharpe)
      -> "Can we learn the graph instead of precomputing it?"
        -> 4a: Unconstrained per-timestep (FAILS - uniform attention)
        -> 4b: Unconstrained prev-window (FAILS - hub-dominant attention)
        -> 4c: Constrained by Pearson mask (WORKS - Sharpe 1.15, non-uniform attention)
      -> "What did each model actually learn?" -> Interpretability
```

Each step motivated by a limitation of the previous.

## Key Arguments

### 1. Graph structure matters, but DYNAMIC structure matters more
- Static GCN: +0.10 over LSTM
- Rolling GCN: +0.21 over LSTM
- The time-varying nature of stock relationships is more important than having a graph at all

### 2. Learning the graph end-to-end is hard in low-data regimes
- ~57 independent training windows for 88 stocks
- Unconstrained GAT cannot discover meaningful structure (uniform or hub-dominant attention)
- Feature dimensionality affects attention: 20-dim -> uniform, 200-dim -> peaked but wrong structure
- Score magnitude (no sqrt scaling) causes dimensionality-dependent attention sharpness

### 3. Domain priors (Pearson structure) are essential
- 4c (Pearson mask + learned weights) achieves Sharpe 1.15 -- best GAT result
- The mask reduces the learning problem from 88x88 = 7744 potential edges to ~15 per node
- GAT can learn meaningful weight differentiation within this constrained neighborhood

### 4. Learned weights vs fixed weights: a nuanced story
- Overall: GCN rolling (1.21) > 4c GAT (1.15) -- fixed Pearson weights slightly better
- BUT year-by-year: GAT outperforms during regime changes (2020 COVID, 2022 rate hikes)
- GCN outperforms during calm periods (2018, 2019, 2021)
- Interpretation: learned weights adapt to regime shifts, fixed weights are more stable in normal times

### 5. Even when GAT underperforms, it offers interpretability advantages
- Asymmetric attention: A[i,j] != A[j,i] -- models directional influence
- GCN fundamentally cannot do this (symmetric adjacency)
- Attention entropy, sector alignment, regime analysis all possible with GAT

## Report Framing

### What the previous student did (brief)
- Straddle portfolio construction
- Feature engineering (momentum, MACD, vol)
- LSTM baseline + static GCN

### What this thesis contributes
1. Rolling GCN with dynamic Pearson correlation graphs
2. Systematic investigation of GAT architectures (4a, 4b, 4c, 4e)
3. Diagnosis of GAT failure modes (uniform attention, hub dominance, residual bypass)
4. Pearson-constrained GAT as principled middle ground
5. Interpretability framework: sector alignment, regime analysis, attention asymmetry
6. Characterization of when learned graphs outperform prescribed graphs (regime shifts)

## Chapter 4 (Methodology) Structure

1. LSTM backbone (shared across all models)
2. Static GCN: fixed Pearson graph
3. Rolling GCN: dynamic Pearson graph (motivation: relationships change)
4. GAT variants:
   a. Why GAT? (learn graph end-to-end, asymmetric weights, data-dependent)
   b. GATv1 vs GATv2 (truly pairwise attention)
   c. Unconstrained attempts and why they fail (4a, 4b)
   d. Pearson-constrained GAT (4c) -- the hybrid approach
5. Interpretability framework
