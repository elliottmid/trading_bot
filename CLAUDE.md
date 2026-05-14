# trading-bot

Research environment for ETF return prediction and backtesting. **Not production. Not alpha-generating yet.** See "State as of 2026-04-17" below before continuing.

## Data (pulled, stored, validated)

Paths are relative to repo root.

- `data/raw/equities_daily.parquet` — SPY, QQQ, PSLV, PHYS · daily OHLCV · 2020-12-14 → present (~1,340 bars/symbol) · Alpaca IEX feed
- `data/raw/equities_hourly.parquet` — same 4 symbols · hourly · ~8k–11k bars/symbol · Alpaca IEX. **Not yet used in any model.**
- `data/raw/breadth_daily.parquet` — IWM, XLK, XLY, XLF, XLI, XLP, XLU, XLV · daily · Alpaca
- `data/raw/macro_daily.parquet` — FRED: DGS3MO, DGS1, DGS10 (rates), VIXCLS, VXVCLS (VIX + 3m VIX), BAMLH0A0HYM2 (HY spread), DTWEXBGS (USD index)
- `data/raw/spy_qqq_2000_daily.parquet` — SPY + QQQ daily OHLCV · 2000-01-03 → present (6,630 bars/symbol) · fetched via yfinance · used for long-history EMA optimization
- `data/processed/features_daily_H{N}.parquet` — per-symbol feature matrix with N-day forward log-return target
- `data/models/wf_preds_H20.parquet` — pooled OOS walk-forward predictions (all 4 symbols)
- `data/models/wf_preds_H20_SPY.parquet` — SPY-only OOS predictions
- `data/models/wf_folds_H20.csv` — per-fold IC / MAE / hit-rate

**Hourly data is capped at ~24 months on yfinance; Alpaca gives longer.** We have ~60 months hourly from Alpaca but haven't modeled it.

## Scripts (all in `scripts/`)

| File | Purpose |
|---|---|
| `fetch_alpaca.py` | 60mo daily + hourly for SPY/QQQ/PSLV/PHYS from Alpaca |
| `fetch_breadth.py` | Daily bars for IWM + 7 sector SPDRs |
| `fetch_fred.py` | Rates + VIX + HY spread + USD index from FRED CSV endpoint |
| `build_features.py` | Feature engineering, configurable `--horizon` (N-day forward log return target) |
| `walk_forward.py` | Rolling 24mo-train / 6mo-test XGBoost, embargo = horizon + 5d. `--symbol` filters to one asset |
| `backtest_rule.py` | Conviction-weighted long-only SPY rule (**underperformed buy-and-hold** — see below) |
| `backtest_rotation.py` | Cross-sectional rotation backtest with configurable `--universe` and `--tc-bps` |
| `backtest_coppock.py` | Historical Coppock-trough trigger scan. `--symbol SPY` (default). Writes per-trigger log + 20-day fwd returns to `data/models/coppock_triggers_{SYMBOL}.csv` |
| `coppock_daily_scan.py` | Daily single-bar Coppock check. Fetches SPY daily from Alpaca, emits one-line verdict to stdout only when a new bar is seen. Silent otherwise; non-zero exit on error. State file: `data/models/coppock_last_scan.txt` |
| `coppock_notify.sh` | launchd wrapper for `coppock_daily_scan.py`. Logs to `~/trading-bot-daily.log`; fires macOS notification on new-bar verdicts and scan errors |
| `backtest_ema_grid_search.py` | Grid search over EMA entry/exit parameter pairs on the 13-symbol ETF universe. Ranks by mean Sharpe. Output: `data/models/ema_grid_search_results.csv` |
| `backtest_ema_walkforward.py` | Walk-forward backtest of the EMA crossover rule with trailing stop, fold-by-fold metrics |
| `ema_4assets.py` | Daily EMA signal scanner for 4-asset portfolio (SPY, QQQ, XLF, XLY). Prints BUY/SELL/HOLD/FLAT signals to stdout |
| `backtest_ema_sh_overlay.py` | Compares baseline 4-asset EMA strategy vs. SH (inverse SPY) overlay variant. See EMA strategy section below |
| `backtest_ema_qqq_grid.py` | Grid search over EMA entry/exit pairs × trailing stop levels (1–6%) on QQQ 2000–present. Output: `data/models/ema_qqq_grid_results.csv` |
| `backtest_ema_qqq_full.py` | Full 26-year backtest of winning QQQ parameters. Output: `data/models/ema_qqq_trades.csv`, `data/models/ema_qqq_equity_curve.csv` |
| `ema_spy_qqq_scan.py` | **Daily close scanner** — SPY + QQQ with individually optimised parameters. Reports signal, open trade entry/peak/stop, or BUY instructions with initial stop price. Run after market close. |

## Credentials

`.env` at repo root (gitignored) contains `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `ALPACA_BASE_URL` (paper account). FRED endpoints are public, no key needed. Paper keys should be rotated — they were pasted into chat during setup.

## State as of 2026-04-17 — what we learned, read this before continuing

Ran a full pipeline: fetch → features → walk-forward XGBoost → backtest, with SPY/QQQ/PSLV/PHYS + macro + breadth features. N-day forward log return target, H=20.

### Model metrics (pooled H=20 walk-forward, 6 folds, OOS)

- Mean rank-IC: **+0.143**
- Mean hit-rate: **49.5%** (no better than coin-flip on direction)
- 5 of 6 folds positive IC, one fold ~0

### Backtest (full universe, Dec 2022 → Apr 2026, 38 rebalances)

| Strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| top_half_eq (rotation) | 42.5% | 2.00 | -14.7% |
| equal_weight (no model) | 32.0% | 1.67 | -19.0% |
| buy_hold_SPY | 18.5% | 1.20 | -19.0% |

Looked great. It's not.

### Three sanity checks that killed the thesis

1. **SPY/QQQ only universe**: rotation (23.1% CAGR, Sharpe 1.25) ties equal-weight (23.0%, 1.29) to the third decimal. Buy-hold QQQ alone beats both (27.5%, 1.32). **The model cannot distinguish SPY from QQQ.**
2. **Per-fold decomposition**: Fold 1 had model IC ≈ 0 (−0.02) but rotation still beat benchmark by +9.6%. Only 3 of 6 folds show clean model→alpha linkage. Alpha is not coming from the model in multiple folds.
3. **Transaction costs (1bp → 10bp)**: Sharpe 2.00 → 1.96. Costs are not the story.

### Honest conclusion

The full-universe rotation's 42% CAGR was almost entirely **"model overweighted PSLV/PHYS during a historic 2023-2026 metals bull run."** Any buy-and-hold basket containing metals during that window did similarly. The model provides no durable cross-sectional timing edge on the equity pair (SPY vs QQQ), which is where real alpha would live.

**Not deployable. Not alpha. Treat this as a null result plus plumbing.**

### What the single-asset backtest (`backtest_rule.py`) showed separately

Conviction-weighted long-only SPY: 7.2% CAGR vs 18.5% buy-hold. The model's *ranking* signal (IC) does not translate to *direction* signal (hit-rate 51%), so a timing rule that goes to cash on low predictions is wrong for this signal.

## Design lessons for the next attempt

- **Universe of 4 is too thin for cross-sectional ranking.** Need 15+ assets to generate real alpha from IC.
- **Include a "naive" benchmark that matches the universe.** Buy-hold SPY is a weak reference when the strategy trades 4 things. Equal-weight of the universe is the honest baseline.
- **Per-fold decomposition of backtest excess return vs. per-fold IC** should be a standard diagnostic. If IC and excess don't correlate fold-by-fold, the backtest is riding regime, not model.
- **Direction vs ranking**: a regression target with strong IC but weak hit-rate cannot drive single-asset timing rules. Either add breadth to exploit ranking, or switch to a classification/direction target.
- **SPY/QQQ are near-efficient on daily frequency.** Monthly (H=20) helps; even-longer horizons may help more. Don't model daily.
- **Metals (PSLV/PHYS) inflate cross-asset IC artificially in this regime.** Bull markets in any included asset are confounders.

## Coppock trough-trigger (added 2026-04-19)

Simple event-driven signal, parameter-fixed, no training. Separate from the ML pipeline.

**Rule (daily, "local trough below zero"):** `Coppock[t-1] < 0` AND `Coppock[t-1] <= Coppock[t-2]` AND `Coppock[t] > Coppock[t-1]`, where `Coppock = WMA(10) of [ROC(14) + ROC(11)]` on close.

**Historical scan results (2020-12-14 → 2026-04-16, 20-day forward return):**

| | SPY | QQQ |
|---|---|---|
| triggers | 44 | 49 |
| hit rate | 65.9% | 61.2% |
| mean ret | +1.29% | +1.80% |
| stdev | 4.67% | 5.37% |
| approx annualized Sharpe | 0.98 | 1.19 |

Caveats: overlapping 20d windows (correlated samples), ~5yr sample skewed toward a bull regime, no costs, no position sizing — this is a return-log, not a strategy backtest.

## Daily automation (macOS launchd)

`~/Library/LaunchAgents/com.elliott.coppock-daily.plist` — weekdays 16:00 America/Chicago. Runs `scripts/coppock_notify.sh` → `scripts/coppock_daily_scan.py`. Logs every run to `~/trading-bot-daily.log`. Notifications fire only on a new bar or an error; holidays/no-new-bar days are silent.

- Enable:  `launchctl load -w ~/Library/LaunchAgents/com.elliott.coppock-daily.plist`
- Disable: `launchctl unload ~/Library/LaunchAgents/com.elliott.coppock-daily.plist`
- Verify:  `launchctl list | grep coppock`
- Manual run: `scripts/coppock_notify.sh`

## EMA crossover strategy (added 2026-05-11)

Rule-based, parameter-fixed, no ML. Separate from the Coppock and ML pipelines.

**Parameters (grid-search winner on 13-symbol ETF universe):**
- Entry: EMA(9) crosses above EMA(11)
- Exit: EMA(10) crosses below EMA(20)
- Trailing stop: 4% from peak (optimal from trailing-stop sweep)
- Universe: SPY, QQQ, XLF, XLY — equal 25% allocation each

**Backtest results (2021-01-04 → 2026-04-23, 5bps TC):**

| Strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 4-asset EMA + 4% trailing stop | 10.3% | 0.97 | -16.4% |
| SPY buy-and-hold | 14.7% | 0.89 | -24.5% |

Strategy trades ~4% CAGR for drawdown protection. True value is capital preservation in sustained bear cycles: a 55% SPY drawdown (2008-style) requires +100% recovery; this strategy estimated at -15–20% requires only +18–25%. The Sharpe (0.97 vs 0.89) is the measurable edge.

**SH overlay — tested and rejected (2026-05-13):**

Hypothesis: when SPY fires an EXIT signal, allocate SPY's 25% slice to SH (ProShares Short S&P500) instead of cash.

Result: **SH overlay hurts.** −0.68% CAGR, Sharpe drops 0.97 → 0.89, MaxDD barely moves. Over 25 SH trades in 5 years, the overlay helped in 1 of 6 calendar years (2022: +1.3%) and hurt in the other 4 (2021: −2.2%, 2023: −0.9%, 2024: −2.3%). Volatility decay in sideways/bull markets consistently exceeds the bear-market protection gained.

**Lesson:** The strategy's drawdown protection comes from going to *cash*, not from active shorting. Inverse ETF decay makes SH a net negative in any environment except a sustained directional bear.

## SPY + QQQ individually optimised EMA strategy (added 2026-05-13)

Separate grid search on 26-year history (2000–present) for each ticker. Trailing stop is **from the highest close since entry** (true trailing stop — not anchored to entry price).

### Optimised parameters

| Ticker | Entry | Exit | Trailing stop | Selection method |
|---|---|---|---|---|
| SPY | EMA(12) crosses above EMA(16) | EMA(8) crosses below EMA(10) | 6% from peak | IS grid search 2000–2019: best Sharpe with trail exits <30% |
| QQQ | EMA(12) crosses above EMA(24) | EMA(10) crosses below EMA(23) | 5% from peak | IS grid search 2000–2019: best Sharpe |

**SPY parameter note:** Earlier versions documented EMA(7/11) entry + EMA(17/20) exit + 4% trail. Those were overfit to the 2020–2026 OOS period (IS Sharpe 0.000, IS CAGR −0.65% on 2000–2019). The corrected parameters above were selected purely on IS evidence, then validated OOS (2020–2026 Sharpe 0.910 vs buy-hold 0.707, MaxDD −13.2% vs −33.7%). Stop criterion: highest IS Sharpe where trailing-stop exits <30% of trades (prevents selecting stops that whipsaw the signal).

### QQQ backtest results (2000-01-03 → 2026-05-13, 5bps TC)

| Metric | Strategy | Buy-Hold QQQ |
|---|---|---|
| CAGR | 8.25% | 8.67% |
| Ann. volatility | 13.5% | 26.8% |
| Sharpe | 0.589 | 0.311 |
| Max drawdown | -31.1% | -83.0% |
| Calmar | 0.27 | 0.10 |

107 trades over 26 years. Win rate 49.5%. Avg win +7.40%, avg loss −3.04%. Exit split: 49% trailing stop / 51% EMA crossover.

### SPY full backtest results (2000-01-03 → 2026-05-13, 5bps TC) — see `backtest_ema_spy_full.py`

| Metric | Strategy | Buy-Hold SPY |
|---|---|---|
| CAGR | 5.49% | 8.27% |
| Ann. volatility | 9.76% | 19.33% |
| Sharpe | 0.549 | 0.412 |
| Max drawdown | -26.1% | -55.2% |
| Calmar | 0.21 | 0.15 |

128 trades over 26 years. Win rate 49.2%. Avg win +4.59%, avg loss −2.03%. Exit split: 6% trailing stop / 94% EMA crossover.

**Key findings:**
- QQQ strategy barely matches buy-hold CAGR (8.25% vs 8.67%). Value is entirely in risk reduction: volatility halved, Sharpe nearly doubled, max DD cut from −83% to −31%.
- SPY strategy: value is drawdown compression (MaxDD ~−13% vs buy-hold ~−34% OOS). CAGR trails buy-hold in bull markets; strategy earns its keep in bear cycles.
- Optimal trailing stop is 5% for QQQ vs 6% for SPY — 6% keeps trail exits to ~12% OOS, meaning the EMA crossover signal drives exits rather than the mechanical stop.
- Exit EMA does real work on QQQ (51% of exits) and SPY (88% OOS). Exit parameter choice is not decorative.
- Best bear protection years (OOS): 2022 (strategy −3.4% vs buy-hold −18.7%).
- Worst lag years: sharp recoveries where re-entry signal is slow (2021, 2023, 2024).

### Daily scanner (`ema_spy_qqq_scan.py`)

Run at market close: `python3 scripts/ema_spy_qqq_scan.py`

Fetches 2 years of history via yfinance, reconstructs current open trade for each ticker, and prints:
- **BUY** — enter tomorrow's open, set trailing stop at N% below entry (stop rises with price)
- **HOLD** — entry date/price, peak price, live stop level, cushion % to stop, unrealised P&L
- **SELL** — exit triggered (EMA cross or trail breached); exit tomorrow's open
- **FLAT** — no position, no signal; shows current EMA gap

No launchd wrapper needed — user runs manually after close.

## What the next approach should address

- State the problem before picking a model. "Predict returns and rotate" is the default retail framing and why retail systematic trading mostly loses. Alternatives: predict volatility/regime, build risk-parity with regime overlay, options-based tail hedging, factor tilt.
- Pick a universe and a baseline *first*, then only build a model if the baseline leaves room.
- If continuing with ML: broader universe, classification target if direction matters, or rank-based cross-sectional portfolio with 15+ names.
