# EMA + ML Compound Filter — Executive Summary

**Date:** 2026-06-07 **Author:** Elliott Middleton, assisted by Claude **Period:** 2010-01-01 → 2026-05-13 (walk-forward OOS) **Symbols:** SPY, QQQ (independently)


## Executive Summary

We tested whether layering a monthly ML regressor (SP500/NDX MODERATE walk-forward XGBoost) onto a rule-based EMA crossover strategy can recover CAGR without sacrificing the strategy's core drawdown protection. Four compound variants were evaluated:

1. **Exit filter** — suppress EMA/trail exits when ML forecast \> threshold (position stays long)

2. **Entry gate** — block EMA entries when ML forecast ≤ threshold (stay flat instead of entering)

3. **Combined** — both filters at the same threshold

4. **L/S filter** — exit filter + go short (via inverse ETF) when an un-suppressed exit fires

All variants were tested via nested walk-forward: threshold selected by IS Sharpe maximisation on the prior 9 calendar years, applied blind to the next OOS year (2010–2026, 17 folds). 5 bps round-trip transaction costs throughout.

**Primary finding:** The exit filter is the only variant that consistently adds value. It recovers ~83–92% of buy-and-hold CAGR while roughly halving drawdown and lifting Sharpe from ~0.9 to ~1.3. All other variants either give up too much CAGR (entry gate, combined) or add risk without proportional reward (L/S filter).


## EMA Baseline Parameters

| Symbol | Entry | Exit | Trail Stop | IS Window |
| - | - | - | - | - |
| SPY | EMA(12) × EMA(16) | EMA(8) × EMA(10) | 6% from peak | 2000–2019 |
| QQQ | EMA(12) × EMA(24) | EMA(10) × EMA(23) | 5% from peak | 2000–2019 |


Parameters are fixed (not re-optimised per fold). Only the ML filter threshold is walk-forward selected.

**ML prediction source:** SP500 MODERATE / NDX MODERATE monthly regressor. Each month-end row forecasts the next 1–3 months. Applied with a +1 month shift (month M forecast applied to trading days in M+1) to eliminate look-ahead bias.


## Walk-Forward OOS Results (2010–2026, 17 Folds)

### SPY

| Metric | Buy-Hold | EMA Baseline | Exit Filter | Entry Gate | Combined | L/S Filter |
| - | - | - | - | - | - | - |
| Chain CAGR | +14.2% | +6.9% | **+13.1%** | +5.0% | +10.3% | +13.0% |
| Avg OOS Sharpe | +0.86 | +0.80 | **+1.26** | +1.12 | **+1.43** | +1.17 |
| Chain MaxDD (daily) | -33.7% | -14.9% | -19.4% | **-11.8%** | -21.5% | -25.7% |
| Hit rate (yr \> 0) | — | — | **88%** | 59% | 76% | 76% |
| Worst suppressed DD | — | — | -13.0% | — | -18.8% | -18.8% |
| Avg IS threshold | — | — | +0.2% | +2.1% | +1.2% | -0.2% |


### QQQ

| Metric | Buy-Hold | EMA Baseline | Exit Filter | Entry Gate | Combined | L/S Filter |
| - | - | - | - | - | - | - |
| Chain CAGR | +19.3% | +12.9% | **+15.8%** | +10.7% | +14.1% | **+17.6%** |
| Avg OOS Sharpe | +0.96 | +1.05 | **+1.37** | +1.12 | +1.25 | +1.36 |
| Chain MaxDD (daily) | -35.1% | -17.5% | -22.8% | **-10.8%** | -22.8% | -22.8% |
| Hit rate (yr \> 0) | — | — | 76% | 71% | 71% | **82%** |
| Worst suppressed DD | — | — | -16.9% | — | -16.9% | -16.9% |
| Avg IS threshold | — | — | +0.6% | +4.0% | +3.0% | -1.1% |



## Variant-by-Variant Assessment

### Exit Filter — Recommended Production Rule

The only variant that consistently delivers across both symbols. **The mechanism is intuitive: EMA exits are suppressed when the ML model expects positive forward returns.** The strategy stays long through noisy EMA crosses in trending bull markets, re-exposing CAGR that the baseline surrenders to whipsaw.

- **SPY:** +13.1% CAGR (vs +14.2% BH, +6.9% baseline). Sharpe +1.26 vs +0.86 BH. MaxDD -19.4% vs -33.7% BH. Positive in 15 of 17 OOS years (88% hit rate).

- **QQQ:** +15.8% CAGR (vs +19.3% BH, +12.9% baseline). Sharpe +1.37 vs +0.96 BH. MaxDD -22.8% vs -35.1% BH.

- **Tail risk (HIGH-3):** The strategy's entire drawdown risk is concentrated in months where an exit is suppressed and the ML model is subsequently wrong. Worst suppressed-hold drawdown: -13.0% (SPY), -16.9% (QQQ). These are not low-probability tails — they happen. The Sharpe improvement is real but it is purchased by accepting this concentrated tail.

### Entry Gate — Not Promoted

Blocking entries when ML is bearish mostly just keeps the strategy flat. It modestly improves Sharpe but at the cost of CAGR (SPY +5.0%, QQQ +10.7%) and hit rate (59% SPY). The ML model is a poor entry-timing signal: the EMA crossover already encodes trend inflections; adding a macro-prediction gate adds noise, not signal. Avg IS threshold of +2–4% (very restrictive) confirms the optimizer is trying to stay flat most of the time.

### Combined — Mixed

Better Sharpe than the exit filter alone (+1.43 SPY) but at meaningful CAGR cost (-2.8pp SPY vs exit filter). The entry gate component is again the drag. Not recommended: the Sharpe improvement over exit filter alone doesn't justify the CAGR reduction.

### L/S Filter — Not Promoted

Goes short (inverse ETF, 0.5 bps/day drag ≈ 1.25%/yr) when the exit filter allows an exit to fire. Results are asset-specific:

- **SPY:** Strictly worse than exit filter on every dimension. CAGR nearly identical (+13.0% vs +13.1%) but Sharpe drops to +1.17 and MaxDD deepens from -19.4% to -25.7%. Bull market snap-backs during short periods are punishing for a low-beta, slow-trending index. The earlier pre-filter SH-overlay rejection (2026-05-13) is confirmed by this more rigorous test.

- **QQQ:** More interesting. +17.6% CAGR (+1.8pp vs exit filter), Sharpe +1.36 (essentially identical), MaxDD -22.8% (same). The short leg captures QQQ's sharper bear-market declines and the momentum-index characteristic of prolonged directional moves. Hit rate improves to 82%. However, the edge is thin and IS-threshold selection is noisy at avg -1.1% (very permissive — most exits allowed to short). One bad short-into-a-rip year (-33% 2009-style) would erase multiple years of edge. Not promoted to production.


## In-Sample vs Walk-Forward Gap

IS numbers (scanning 2010–present for best threshold, no folding) are materially inflated:

| Symbol | Variant | IS CAGR | WF CAGR | Gap |
| - | - | - | - | - |
| SPY | Exit filter | +17.1% | +13.1% | -4.0pp |
| SPY | L/S filter | +19.2% | +13.0% | -6.2pp |
| QQQ | Exit filter | +22.2% | +15.8% | -6.4pp |
| QQQ | L/S filter | +24.1% | +17.6% | -6.5pp |


The gap reflects both IS overfitting (threshold tuned to the same period being evaluated) and the expected IS→OOS decay. Walk-forward numbers are the operative figures.


## Threshold Stability

The IS-optimal threshold is near zero for the exit filter on both symbols (avg +0.2% SPY, +0.6% QQQ), which validates the practical 0.0% threshold used in the daily scanner. The threshold does not need to be negative to generate value: suppressing exits only when the model explicitly expects positive returns is the right operating point.


## Production Recommendation

**Deploy the exit filter at 0.0% threshold for both SPY and QQQ.** This is already implemented in `ema\_spy\_qqq\_scan.py`. No changes needed from this analysis.

Do not deploy:

- Entry gate: kills CAGR without adequate Sharpe compensation.

- Combined: marginal Sharpe gain vs exit filter, meaningful CAGR loss.

- L/S filter: adds risk (MaxDD, short squeeze exposure) for returns that do not survive walk-forward discipline on SPY; QQQ edge is real but thin and tail-exposed.


## Key Caveats

1. **EMA parameters are fixed** (IS-optimized 2000–2019). Only the ML threshold is walk-forward validated here. A fully nested optimization (EMA params + threshold co-selected per fold) would give cleaner estimates but was not run.

2. **2026 is a partial year** (through May 2026) and contributes proportionally less to chain CAGR.

3. **Prediction coverage ends May 2026.** Days beyond that date have no ML prediction; the filter behaves as baseline (exits are not suppressed). Re-run MODERATE models monthly to keep the filter active.

4. **Suppressed-exit tail is concentrated, not diversified.** The worst suppressed-hold drawdowns (-13% SPY, -17% QQQ) are real, observable events. Position sizing should account for this.

5. **L/S uses idealized inverse-ETF returns** (−daily return − 0.5 bps/day). Actual SH/PSQ performance can diverge from this model in prolonged high-volatility environments due to compounding drag. Real-world L/S results would likely be modestly worse than reported here.

