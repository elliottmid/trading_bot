# trading-bot

Research environment for ETF return prediction and backtesting. **Not production. Not alpha-generating yet.** See "State as of 2026-04-17" below before continuing.

## Data (pulled, stored, validated)

Paths are relative to repo root.

- `data/raw/equities_daily.parquet` — SPY, QQQ, PSLV, PHYS · daily OHLCV · 2020-12-14 → present (~1,340 bars/symbol) · Alpaca IEX feed
- `data/raw/equities_hourly.parquet` — same 4 symbols · hourly · ~8k–11k bars/symbol · Alpaca IEX. **Not yet used in any model.**
- `data/raw/breadth_daily.parquet` — IWM, XLK, XLY, XLF, XLI, XLP, XLU, XLV · daily · Alpaca
- `data/raw/macro_daily.parquet` — FRED: DGS3MO, DGS1, DGS10 (rates), VIXCLS, VXVCLS (VIX + 3m VIX), BAMLH0A0HYM2 (HY spread), DTWEXBGS (USD index)
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

## What the next approach should address

- State the problem before picking a model. "Predict returns and rotate" is the default retail framing and why retail systematic trading mostly loses. Alternatives: predict volatility/regime, build risk-parity with regime overlay, options-based tail hedging, factor tilt.
- Pick a universe and a baseline *first*, then only build a model if the baseline leaves room.
- If continuing with ML: broader universe, classification target if direction matters, or rank-based cross-sectional portfolio with 15+ names.
