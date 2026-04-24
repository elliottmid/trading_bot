#!/usr/bin/env python3
"""
Grid search over EMA crossover parameters.

Entry signal : entry_fast crosses ABOVE entry_slow  → BUY
Exit  signal : exit_fast  crosses BELOW exit_slow   → SELL

Search space
  entry_fast : 3–12  (must be < entry_slow)
  entry_slow : 8–26
  exit_fast  : 3–12  (must be < exit_slow)
  exit_slow  : 8–26

Strategy: each combo is evaluated on every symbol; universe-level metrics
(mean CAGR, mean Sharpe, win-rate, etc.) are aggregated and ranked.

Usage
-----
    python3 backtest_ema_grid_search.py [--tc-bps 5] [--jobs -1] [--top-n 20]
"""
from __future__ import annotations

import sys
import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_etf_data(filename: str = "ema_etfs_primary_daily.parquet") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} bars for {df['symbol'].nunique()} symbols")
    return df


# ---------------------------------------------------------------------------
# Precompute all EMA series we'll ever need
# ---------------------------------------------------------------------------

def precompute_emas(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """
    Add columns ema_<period> for every period in `periods`.
    Computed per symbol; result is a copy of df.
    """
    logger.info(f"Precomputing EMAs for periods: {sorted(periods)}")
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("timestamp")
        for p in periods:
            g[f"ema_{p}"] = g["close"].ewm(span=p, adjust=False).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Signal generation (vectorised, operates on one symbol's data)
# ---------------------------------------------------------------------------

def _crossover_signal(fast_col: pd.Series, slow_col: pd.Series) -> pd.Series:
    """
    Returns a Series with:
      +1 where fast crosses above slow (buy)
      -1 where fast crosses below slow (sell)
       0 otherwise
    """
    above     = (fast_col > slow_col).astype(int)
    above_lag = above.shift(1)
    sig = pd.Series(0, index=fast_col.index, dtype=int)
    sig[(above_lag == 0) & (above == 1)] =  1   # bullish cross
    sig[(above_lag == 1) & (above == 0)] = -1   # bearish cross
    return sig


# ---------------------------------------------------------------------------
# Single-symbol backtest for one parameter combo
# ---------------------------------------------------------------------------

def backtest_symbol_combo(
    sym_df: pd.DataFrame,
    ef: int, es: int,   # entry fast / slow
    xf: int, xs: int,   # exit  fast / slow
    tc_bps: float = 0,
) -> dict:
    """
    Backtest one symbol with separate entry and exit EMA pairs.

    Returns a dict of scalar metrics (or None if too few bars).
    """
    df = sym_df.copy().reset_index(drop=True)
    if len(df) < max(es, xs) + 5:
        return None

    symbol = df["symbol"].iloc[0]

    entry_sig = _crossover_signal(df[f"ema_{ef}"], df[f"ema_{es}"])
    exit_sig  = _crossover_signal(df[f"ema_{xf}"], df[f"ema_{xs}"])

    # Walk through bars, tracking trades
    in_position = False
    entry_price = None
    entry_idx   = None
    pnls        = []
    hold_days_list = []
    tc_frac = tc_bps / 10_000

    for i in range(len(df)):
        if not in_position and entry_sig.iloc[i] == 1:
            in_position = True
            entry_price = df["close"].iloc[i]
            entry_idx   = i
        elif in_position and exit_sig.iloc[i] == -1:
            exit_price = df["close"].iloc[i]
            pnl = (exit_price - entry_price) / entry_price - 2 * tc_frac
            pnls.append(pnl)
            hold_days_list.append(i - entry_idx)
            in_position = False
            entry_price = None
            entry_idx   = None

    # ---- metrics ----
    n_trades = len(pnls)
    if n_trades == 0:
        return {
            "symbol": symbol,
            "trades": 0,
            "strategy_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_hold_days": 0.0,
            "buy_hold_return": (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0],
        }

    pnls_arr    = np.array(pnls)
    strat_ret   = float(np.prod(1 + pnls_arr) - 1)
    win_mask    = pnls_arr > 0
    win_rate    = win_mask.mean()
    avg_win     = pnls_arr[win_mask].mean()  if win_mask.any()  else 0.0
    avg_loss    = pnls_arr[~win_mask].mean() if (~win_mask).any() else 0.0
    avg_hold    = float(np.mean(hold_days_list))
    bh_ret      = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]

    years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25
    cagr  = float((1 + strat_ret) ** (1 / years) - 1) if years > 0 else 0.0

    # Daily equity for Sharpe — mark-to-market while in position
    daily_ret = np.zeros(len(df))
    in_pos  = False
    ep      = None
    prev_eq = 0.0
    for i in range(len(df)):
        if not in_pos and entry_sig.iloc[i] == 1:
            in_pos = True
            ep = df["close"].iloc[i]
            prev_eq = 0.0
        elif in_pos and exit_sig.iloc[i] == -1:
            in_pos = False
        if in_pos and ep is not None:
            eq = (df["close"].iloc[i] - ep) / ep
            daily_ret[i] = eq - prev_eq
            prev_eq = eq

    vol    = daily_ret.std()
    sharpe = float(daily_ret.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    return {
        "symbol":          symbol,
        "trades":          n_trades,
        "strategy_return": strat_ret,
        "cagr":            cagr,
        "sharpe":          sharpe,
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_hold_days":   avg_hold,
        "buy_hold_return": float(bh_ret),
    }


# ---------------------------------------------------------------------------
# One full universe run for a single (ef, es, xf, xs) combo
# ---------------------------------------------------------------------------

def run_combo(
    sym_groups: dict,           # {symbol: DataFrame}  — precomputed EMAs already in df
    ef: int, es: int,
    xf: int, xs: int,
    tc_bps: float,
) -> dict | None:
    """
    Run all symbols for one parameter combo and return aggregated metrics.
    Returns None if no valid symbols.
    """
    rows = []
    for sym_df in sym_groups.values():
        r = backtest_symbol_combo(sym_df, ef, es, xf, xs, tc_bps)
        if r is not None:
            rows.append(r)

    if not rows:
        return None

    mdf = pd.DataFrame(rows)

    return {
        "entry_fast":     ef,
        "entry_slow":     es,
        "exit_fast":      xf,
        "exit_slow":      xs,
        # universe averages
        "mean_cagr":      mdf["cagr"].mean(),
        "mean_sharpe":    mdf["sharpe"].mean(),
        "mean_win_rate":  mdf["win_rate"].mean(),
        "mean_trades":    mdf["trades"].mean(),
        "mean_hold_days": mdf["avg_hold_days"].mean(),
        "median_cagr":    mdf["cagr"].median(),
        "median_sharpe":  mdf["sharpe"].median(),
        # fraction of symbols with positive strategy return
        "pct_profitable": (mdf["strategy_return"] > 0).mean(),
        "symbols_tested": len(mdf),
    }


# ---------------------------------------------------------------------------
# Grid search orchestration
# ---------------------------------------------------------------------------

def build_param_grid(
    fast_range: tuple[int, int] = (3, 12),
    slow_range: tuple[int, int] = (8, 26),
) -> list[tuple[int, int, int, int]]:
    """
    All valid (entry_fast, entry_slow, exit_fast, exit_slow) combos
    where fast < slow for each pair.
    """
    fasts = range(fast_range[0], fast_range[1] + 1)
    slows = range(slow_range[0], slow_range[1] + 1)
    entry_pairs = [(f, s) for f, s in product(fasts, slows) if f < s]
    exit_pairs  = entry_pairs  # same search space
    combos = [(ef, es, xf, xs) for (ef, es), (xf, xs) in product(entry_pairs, exit_pairs)]
    logger.info(f"Grid: {len(entry_pairs)} entry pairs × {len(exit_pairs)} exit pairs "
                f"= {len(combos):,} combos")
    return combos


def run_grid_search(
    df: pd.DataFrame,
    fast_range: tuple[int, int] = (3, 12),
    slow_range: tuple[int, int] = (8, 26),
    tc_bps: float = 0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    combos = build_param_grid(fast_range, slow_range)

    # Precompute every unique EMA period needed
    all_periods = sorted(set(
        p for combo in combos for p in combo
    ))
    df = precompute_emas(df, all_periods)

    # Split into per-symbol dict so workers don't re-filter the full frame
    sym_groups = {sym: g for sym, g in df.groupby("symbol", sort=False)}
    logger.info(f"Running {len(combos):,} combos on {len(sym_groups)} symbols "
                f"(jobs={n_jobs}, tc={tc_bps}bps) …")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_combo)(sym_groups, ef, es, xf, xs, tc_bps)
        for ef, es, xf, xs in combos
    )

    rows = [r for r in results if r is not None]
    if not rows:
        logger.error("No results returned — check your data file.")
        sys.exit(1)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("mean_sharpe", ascending=False).reset_index(drop=True)
    results_df.insert(0, "rank", results_df.index + 1)
    return results_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_top_n(results_df: pd.DataFrame, n: int = 20) -> None:
    cols = [
        "rank", "entry_fast", "entry_slow", "exit_fast", "exit_slow",
        "mean_sharpe", "mean_cagr", "mean_win_rate", "mean_trades",
        "mean_hold_days", "pct_profitable",
    ]
    fmt = {
        "mean_sharpe":    "{:.3f}".format,
        "mean_cagr":      "{:.2%}".format,
        "mean_win_rate":  "{:.2%}".format,
        "mean_trades":    "{:.1f}".format,
        "mean_hold_days": "{:.1f}".format,
        "pct_profitable": "{:.2%}".format,
    }
    top = results_df.head(n)[cols].copy()
    for col, fn in fmt.items():
        top[col] = top[col].map(fn)

    print("\n" + "=" * 110)
    print(f"TOP {n} PARAMETER COMBOS  (ranked by mean Sharpe across universe)")
    print("=" * 110)
    print(top.to_string(index=False))

    # Also surface best by mean CAGR
    top_cagr = results_df.nlargest(5, "mean_cagr")[cols].copy()
    for col, fn in fmt.items():
        top_cagr[col] = top_cagr[col].map(fn)
    print("\n" + "-" * 110)
    print("TOP 5 BY MEAN CAGR")
    print("-" * 110)
    print(top_cagr.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search: EMA crossover with separate entry/exit periods"
    )
    parser.add_argument("--data",       default="ema_etfs_primary_daily.parquet")
    parser.add_argument("--tc-bps",     type=float, default=0,
                        help="Transaction cost in basis points (default 0)")
    parser.add_argument("--jobs",       type=int,   default=-1,
                        help="Parallel workers; -1 = all CPUs (default)")
    parser.add_argument("--top-n",      type=int,   default=20,
                        help="Number of top combos to print (default 20)")
    parser.add_argument("--entry-fast-min", type=int, default=3)
    parser.add_argument("--entry-fast-max", type=int, default=12)
    parser.add_argument("--slow-min",       type=int, default=8)
    parser.add_argument("--slow-max",       type=int, default=26)
    parser.add_argument("--output",     default="ema_grid_search_results.csv",
                        help="Output filename inside data/models/")
    args = parser.parse_args()

    df = load_etf_data(args.data)

    results_df = run_grid_search(
        df,
        fast_range=(args.entry_fast_min, args.entry_fast_max),
        slow_range=(args.slow_min,       args.slow_max),
        tc_bps=args.tc_bps,
        n_jobs=args.jobs,
    )

    print_top_n(results_df, n=args.top_n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.output
    results_df.to_csv(out_path, index=False)
    logger.info(f"Full results saved → {out_path}")


if __name__ == "__main__":
    main()
