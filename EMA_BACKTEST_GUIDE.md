# EMA12/EMA26 Crossover Backtest System

Complete backtesting apparatus for EMA crossover strategies on a broad ETF universe (13 primary symbols, 30+ extended).

## Quick Start

```bash
cd scripts
bash run_ema_pipeline.sh
```

This fetches daily data from Alpaca and runs backtests with 0bp and 5bp transaction costs. Results go to `data/models/`.

## Architecture

### 1. `ema_etf_universe.py`
Canonical ETF lists grouped by asset class:

| Group | ETFs | Purpose |
|---|---|---|
| Equity broad | SPY, QQQ, IWM | Core equity indices |
| Sector SPDR | XLK, XLV, XLF, XLY, XLP, XLE, XLI, XLU, XLRE, XLC | 10 major sectors |
| Sector iShares | IYW, IYH, IYF, IYC, IYE, IYJ | Sector alternatives |
| Commodities | GLD, SLV, USO, DBC, DBB, UNG | Commodity exposure |
| Bonds | BND, TLT, SHV, HYG, LQD | Fixed income |
| Currencies | FXE, FXY, FXB, FXA | FX pairs |
| Growth | TQQQ, UPRO, SQQQ, EEM | Leveraged / EM |

**Primary universe** (first backtest): 13 equities + sectors (SPY, QQQ, IWM + 10 sector SPDR).  
**Full universe**: 30+ total symbols for later exploration.

### 2. `fetch_ema_etfs.py`
Downloads daily OHLCV bars from Alpaca.

**Usage:**
```bash
# Fetch primary universe
python3 fetch_ema_etfs.py --universe primary --start-date 2020-12-14

# Fetch full universe (may take longer)
python3 fetch_ema_etfs.py --universe full --start-date 2020-12-14

# Custom date range
python3 fetch_ema_etfs.py --universe primary --start-date 2023-01-01 --end-date 2024-12-31
```

**Output:** `data/raw/ema_etfs_{primary|full}_daily.parquet`

Requires `.env` with `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL` (already configured).

### 3. `backtest_ema_crossover.py`
Core backtesting engine. Implements EMA12/EMA26 crossover logic and performance calculations.

**Signal logic:**
- **BUY**: EMA12 crosses above EMA26
- **SELL**: EMA12 crosses below EMA26
- **Position**: Long from buy signal to sell signal (hold until opposite signal)

**Metrics per symbol:**
- Trades: Number of completed round trips
- Strategy return: Total P&L as pct (with transaction costs)
- Buy-hold return: Benchmark (long from start to end)
- CAGR: Compound annual growth rate
- Sharpe: Daily return Sharpe ratio (annualized)
- Win rate: % of trades with positive P&L
- Avg win / loss: Mean P&L on winning and losing trades

**Usage:**
```bash
# Backtest with 0bp costs (no slippage)
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 0

# Backtest with 5bp costs (realistic)
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 5

# Backtest with higher costs
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 10
```

**Output:**
- Console: Summary table of all symbols + rankings
- CSV: `data/models/ema_metrics_{Xbps}.csv` (metrics per symbol)
- CSV: `data/models/ema_trades_{Xbps}.csv` (trade log)

### 4. `run_ema_pipeline.sh`
Orchestration script. Runs fetch → backtest (0bp) → backtest (5bp) in sequence.

## Key Design Choices

### Why primary universe first?
- **13 symbols** is manageable (clear readability of results, faster iteration)
- **Equities + sectors** are highly tradeable, deep liquidity
- Lower correlation within sectors helps validate the signal
- Commodities/bonds/currencies can be added after primary validates

### Why daily bars?
- Cleaner EMA calculation than intraday (fewer re-entries per day)
- Aligns with your existing daily data pipeline
- ETF liquidity is tight on daily close
- Hourly data available for later refinement if needed

### Why hold-until-signal?
- Simplest logic (no time stops, no take-profit levels)
- Aligns with your preference ("any holding period")
- Tests the fundamental EMA signal quality
- Later: add max holding period cap (e.g., 20 days) if churn is high

### Transaction costs?
- **0bp**: theoretical best case (no slippage, free execution)
- **5bp**: realistic for 13 ETFs (IB/Alpaca typical bid-ask is 1-2bp, buffer for slippage)
- **10bp**: conservative (if using market orders during heavy volume)

Cost sensitivity tells you if alpha survives execution friction.

## Expected Results Structure

After running `run_ema_pipeline.sh`:

```
data/raw/
  └─ ema_etfs_primary_daily.parquet       (1,200-1,400 rows per symbol, ~20k total)

data/models/
  ├─ ema_metrics_0bps.csv                 (13 rows, per-symbol metrics)
  ├─ ema_metrics_5bps.csv
  ├─ ema_trades_0bps.csv                  (trade log, entry/exit details)
  └─ ema_trades_5bps.csv
```

**Typical metrics CSV row:**
```
symbol,start_date,end_date,years,trades,strategy_return,buy_hold_return,excess_return,cagr,sharpe,win_rate,avg_win,avg_loss
SPY,2020-12-14,2026-04-24,5.34,18,0.245,-0.045,0.290,0.044,0.52,0.556,0.015,-0.008
```

**Typical trades CSV row:**
```
symbol,entry_date,entry_price,exit_date,exit_price,pnl_pct,hold_days
SPY,2021-03-15,382.45,2021-05-22,417.93,0.0928,68
```

## Next Steps After Backtest

1. **Evaluate:** Which symbols have positive CAGR? Do Sharpe + excess_return align?
2. **Robustness check:** Run on full universe (commodities, bonds) — does signal generalize?
3. **Cost sensitivity:** Compare 0bps, 5bps, 10bps metrics. Does alpha persist?
4. **Parameter sweep (optional):** Test EMA(10,20), EMA(9,21), etc. to see if 12/26 is optimal.
5. **Sanity checks:** 
   - Verify: Do buy signals occur before major up moves? Sell signals before major down moves?
   - Confirm: No look-ahead bias (EMA lag is natural)
   - Check: Per-fold decomposition if backtesting rolling windows
6. **Live paper automation (next phase):** Use the same logic in a daily scan like `coppock_daily_scan.py`

## Files to Keep in Sync

- `.env` — Alpaca credentials (already configured, no action needed)
- `data/raw/ema_etfs_primary_daily.parquet` — Data cache; re-fetch monthly
- `data/models/ema_*.csv` — Results; archive old runs if comparing variants

## Debugging / Common Issues

| Issue | Cause | Fix |
|---|---|---|
| "Missing ALPACA_API_KEY" | `.env` file missing or unset | Check `.env` exists at repo root; set vars |
| FileNotFoundError on .parquet | Data hasn't been fetched yet | Run `fetch_ema_etfs.py` first |
| All signals on same day | Data stale, market closed, data fetch partial | Re-fetch with current date |
| All symbols have Sharpe ≈ 0 | EMA giving random entries | Check signal logic, verify data quality |
| CAGR = 0%, trades = 0 | Symbol in universe but not on Alpaca | Verify ticker; remove from list if delisted |

## Extending This System

### Add new ETF
1. Edit `ema_etf_universe.py`, add ticker to appropriate dict
2. Re-run `fetch_ema_etfs.py` (auto-includes new symbol)
3. Re-run `backtest_ema_crossover.py`

### Test different EMA periods
Modify `backtest_ema_crossover.py` line:
```python
symbol_df = calculate_emas(symbol_df, ema_fast=9, ema_slow=21)
```
Then re-run backtest.

### Add position sizing
Extend backtest to weight by Sharpe or inverse volatility. See `backtest_rotation.py` for example.

### Add max holding period
In `backtest_symbol()`, add:
```python
if hold_days > 20:  # Exit after 20 days max
    # force exit logic
```

## Alignment with Your Project

This system is **intentionally simple** and separate from the ML pipeline (`walk_forward.py`, `build_features.py`) to avoid the lessons learned:

- ✓ Signal is **ranked on historical data**, not fit to prices
- ✓ Uses **independent universe** (not just SPY/QQQ pair)
- ✓ Per-symbol backtest results are **transparent** (not pooled OOS predictions)
- ✓ **Baseline is equal-weight universe**, not buy-hold SPY alone
- ✓ Transaction costs are **explicit** (0bp, 5bp, 10bp cases)
- ✓ No look-ahead, no future information leakage

If EMA gives durable edge on this broad universe with realistic costs, it can graduate to:
- A daily scan + alert system (like Coppock)
- Rotational portfolio (overweight top-IC symbols)
- Multi-timeframe ensemble (daily + weekly EMA)

## References

- EMA calculation: Pandas `.ewm(span=n, adjust=False).mean()` (recursive exponential smoothing)
- Crossover logic: Previous bar state vs. current bar state
- Performance metrics: Standard backtest calculations (CAGR, Sharpe, win rate)
- Your existing infrastructure: Alpaca fetch, daily bars, parquet storage, paper trading
