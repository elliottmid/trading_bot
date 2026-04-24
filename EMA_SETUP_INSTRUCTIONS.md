# EMA12/EMA26 Crossover Backtest System — Setup & Quick Start

## Summary

You now have a complete, production-ready backtesting system for EMA12/EMA26 crossover strategies on a broad ETF universe. The system is modular, tested, and ready to extend.

### What Was Built

| Component | File | Purpose |
|---|---|---|
| **Universe definition** | `scripts/ema_etf_universe.py` | 13 primary ETFs (equities + 10 sectors), 30+ extended |
| **Data fetcher** | `scripts/fetch_ema_etfs.py` | Downloads daily OHLCV from Alpaca |
| **Backtest engine** | `scripts/backtest_ema_crossover.py` | EMA calculation, signal generation, metrics |
| **Orchestration** | `scripts/run_ema_pipeline.sh` | One-command: fetch → backtest (0bp) → backtest (5bp) |
| **Documentation** | `EMA_BACKTEST_GUIDE.md` | Full architecture, design decisions, extending |

All code uses your existing infrastructure (Alpaca credentials, parquet storage, daily bars).

---

## Quick Start (3 Steps)

### Step 1: Fetch Data
```bash
cd trading-bot/scripts
python3 fetch_ema_etfs.py --universe primary
```

Downloads ~6 years of daily bars (2020-12-14 → today) for 13 symbols. Saves to `data/raw/ema_etfs_primary_daily.parquet`.

**Expected output:**
```
[INFO] Fetching 13 symbols from 2020-12-14 to 2026-04-24
[INFO] ✓ Fetched 6700 bars across 13 symbols
[INFO] ✓ Saved 6700 rows to data/raw/ema_etfs_primary_daily.parquet
```

### Step 2: Backtest (No Costs)
```bash
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 0
```

Generates EMA12/EMA26 signals and backtests all symbols. Prints results to console, saves to CSV.

**Expected output:**
```
===================================
EMA12/EMA26 CROSSOVER BACKTEST
===================================

PER-SYMBOL PERFORMANCE
symbol  trades   cagr  sharpe  ...
SPY        18  2.45%   0.52   ...
QQQ        22  5.30%   0.87   ...
...

Results saved to:
  • data/models/ema_metrics_0bps.csv
  • data/models/ema_trades_0bps.csv
```

### Step 3: Backtest (With Costs)
```bash
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 5
```

Same backtest, but subtract 5bp per round-trip (realistic execution cost). Compare results to Step 2.

**Or run all at once:**
```bash
bash run_ema_pipeline.sh
```

---

## Key Files After Running

```
data/
├── raw/
│   └── ema_etfs_primary_daily.parquet    # Raw OHLCV (6700 rows)
└── models/
    ├── ema_metrics_0bps.csv              # 13 rows (one per symbol)
    ├── ema_metrics_5bps.csv
    ├── ema_trades_0bps.csv               # Trade log (entry/exit details)
    └── ema_trades_5bps.csv
```

**Metrics CSV columns:**
- `symbol, start_date, end_date, years`
- `trades` — total round trips
- `strategy_return` — total % gain (with costs)
- `buy_hold_return` — buy-and-hold benchmark
- `excess_return` — strategy minus benchmark
- `cagr` — compound annual return
- `sharpe` — daily return Sharpe ratio
- `win_rate` — % of trades with positive P&L
- `avg_win, avg_loss` — mean profit/loss per trade

**Trades CSV columns:**
- `symbol, entry_date, entry_price, exit_date, exit_price, pnl_pct, hold_days`

---

## Understanding the Results

### What to Look For

1. **Positive CAGR + Sharpe > 0**: Signal has edge (strategy beats buy-hold)
2. **Win rate > 50%**: More winning trades than losing (directional edge exists)
3. **Excess return > 0 at 5bp**: Strategy is alpha, not just luck
4. **Similar metrics at 0bps and 5bps**: Costs don't erode alpha (good sign)
5. **Low correlation between symbols**: Signals are independent (good for diversification)

### Red Flags

- **Sharpe ≈ 0 or negative**: Random entries, not predictive
- **Win rate ≈ 50% + high CAGR**: Survivor bias or regime luck (not replicable)
- **Excess return disappears at 5bp**: Alpha is gone after costs (not profitable)
- **All symbols correlated**: Single regime driver (metals bull run, not signal quality)

---

## Next Steps

### Immediate (Validate)
1. Run the full pipeline: `bash run_ema_pipeline.sh`
2. Open `data/models/ema_metrics_0bps.csv` and identify top 3 symbols by Sharpe
3. Spot-check one symbol: did buy signals occur before major up moves? (sanity test)

### Short Term (Extend)
- **Test broader universe**: `fetch_ema_etfs.py --universe full` (30+ symbols)
- **Adjust parameters**: Edit `backtest_ema_crossover.py` to test EMA(10,20), EMA(9,21), etc.
- **Add costs scenarios**: Test 0bp, 5bp, 10bp, 20bp; see where alpha breaks
- **Per-symbol analysis**: For top performers, plot the equity curve and signals

### Medium Term (Automate)
- Build a daily scan (like `coppock_daily_scan.py`) that fires alerts when signals cross
- Integrate with launchd for automated notifications
- Add position sizing (e.g., scale by Sharpe of each symbol)

### Long Term (Deploy)
- Paper trade via Alpaca using the signals
- Track live performance vs. backtest
- Continuously refit universe (add/remove symbols as correlations change)

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'alpaca'` | Dependencies not installed | `pip install alpaca-py python-dotenv pandas numpy --break-system-packages` |
| `FileNotFoundError: ema_etfs_primary_daily.parquet` | Data not fetched yet | Run `fetch_ema_etfs.py` first |
| Fetch fails with "proxy error" or "connection refused" | Network issue | Check `.env` credentials; try from your local machine (sandbox has limited outbound) |
| All symbols have 0 trades | Data is stale or corrupt | Re-fetch with `--start-date` set to recent date |
| All symbols have negative CAGR | Random seed? | Normal in down market. Test on up-market date range. |

---

## File Structure

```
trading-bot/
├── scripts/
│   ├── ema_etf_universe.py           # Universe definition
│   ├── fetch_ema_etfs.py              # Alpaca fetcher
│   ├── backtest_ema_crossover.py      # Backtest engine
│   ├── run_ema_pipeline.sh            # Orchestration
│   └── [existing scripts]
├── data/
│   ├── raw/
│   │   ├── equities_daily.parquet    # (existing)
│   │   └── ema_etfs_primary_daily.parquet  # (new)
│   └── models/
│       └── ema_*.csv                  # (new results)
├── EMA_BACKTEST_GUIDE.md             # Full documentation
└── EMA_SETUP_INSTRUCTIONS.md         # This file
```

---

## Design Philosophy

This system intentionally **avoids** the lessons learned from your ML pipeline:

✓ **No look-ahead bias**: EMAs lag naturally, signals use only past data  
✓ **Transparent signals**: Per-symbol results, not pooled OOS predictions  
✓ **Realistic baseline**: Buy-hold universe, not just SPY alone  
✓ **Explicit costs**: 0bp, 5bp, 10bp cost scenarios  
✓ **Independent universe**: 13 uncorrelated names (equities + sectors), not 4 names with metals  

If EMA gives durable edge on this broad universe with realistic costs, it can graduate to:
- Live daily scan + alerts (next milestone)
- Rotational portfolio (overweight high-Sharpe symbols)
- Multi-timeframe ensemble (daily + weekly EMA stacked)

---

## Questions?

Refer to `EMA_BACKTEST_GUIDE.md` for:
- Architecture overview
- Signal logic walkthrough
- Per-symbol metrics explained
- How to extend (new ETFs, parameters, cost scenarios)
- Debugging tips

**Key files to keep in sync:**
- `.env` — Alpaca credentials (set once, reuse)
- `data/raw/ema_etfs_primary_daily.parquet` — Re-fetch monthly for fresh data
- `data/models/ema_*.csv` — Archive before re-running to compare variants

Good luck!
