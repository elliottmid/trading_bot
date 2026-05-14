#!/usr/bin/env python3
"""
EMA crossover grid search on QQQ (2000-present) with trailing stop sweep.

Mirrors the SPY optimization: exhaustive search over entry/exit EMA pairs
and trailing stop percentages. Ranks by Sharpe; also reports CAGR, max
drawdown, Calmar, win rate, and exit-type split (trail vs EMA).

Usage
-----
    python3 backtest_ema_qqq_grid.py
    python3 backtest_ema_qqq_grid.py --tc-bps 5 --top-n 30
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "models"

# ── grid ─────────────────────────────────────────────────────────────────────
ENTRY_FASTS   = list(range(3, 13))       # 3–12
ENTRY_SLOWS   = list(range(8, 27))       # 8–26
EXIT_FASTS    = list(range(3, 13))
EXIT_SLOWS    = list(range(8, 27))
TRAIL_STOPS   = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]   # 1–6 %
TC_BPS        = 5


# ── data ─────────────────────────────────────────────────────────────────────

def load_qqq() -> pd.DataFrame:
    path = DATA_DIR / "spy_qqq_2000_daily.parquet"
    if not path.exists():
        logger.error(f"Not found: {path}  — run the fetch step first")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    qqq = df[df["symbol"] == "QQQ"].sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"QQQ: {len(qqq):,} bars  "
        f"{qqq['timestamp'].min().date()} → {qqq['timestamp'].max().date()}"
    )
    return qqq


# ── EMA precompute ────────────────────────────────────────────────────────────

def add_emas(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


# ── single backtest ───────────────────────────────────────────────────────────

def backtest(
    close: np.ndarray,
    ema_fast_entry: np.ndarray,
    ema_slow_entry: np.ndarray,
    ema_fast_exit:  np.ndarray,
    ema_slow_exit:  np.ndarray,
    trail_pct: float,
    tc_frac:   float,
) -> dict:
    n = len(close)
    in_pos      = False
    entry_price = 0.0
    peak        = 0.0
    daily_rets  = np.zeros(n)
    prev_eq     = 0.0
    pnls        = []
    trail_exits = 0
    ema_exits   = 0

    for i in range(1, n):
        sig = i - 1   # signal fires at end of bar sig; exec at close of bar i

        entry_cross = (ema_fast_entry[sig] > ema_slow_entry[sig] and
                       ema_fast_entry[sig - 1] <= ema_slow_entry[sig - 1]) if sig > 0 else False
        exit_cross  = (ema_fast_exit[sig] < ema_slow_exit[sig] and
                       ema_fast_exit[sig - 1] >= ema_slow_exit[sig - 1]) if sig > 0 else False

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close[i]
            peak        = close[i]
            prev_eq     = 0.0
            daily_rets[i] -= tc_frac

        if in_pos:
            peak = max(peak, close[i])
            eq   = (close[i] - entry_price) / entry_price
            daily_rets[i] += eq - prev_eq
            prev_eq = eq

            trail_hit = close[i] <= peak * (1 - trail_pct)
            if trail_hit or exit_cross:
                daily_rets[i] -= tc_frac
                pnls.append(eq - 2 * tc_frac)
                if trail_hit:
                    trail_exits += 1
                else:
                    ema_exits += 1
                in_pos  = False
                prev_eq = 0.0

    n_trades = len(pnls)
    if n_trades == 0:
        return None

    pnls_arr = np.array(pnls)
    cum      = np.cumprod(1 + daily_rets)
    peak_eq  = np.maximum.accumulate(cum)
    dd       = (cum - peak_eq) / peak_eq
    max_dd   = float(dd.min())

    years    = (n - 1) / 252.0
    total    = float(np.prod(1 + pnls_arr) - 1)
    base     = 1 + total
    cagr     = float(base ** (1 / years) - 1) if years > 0 and base > 0 else 0.0

    vol    = daily_rets.std()
    sharpe = float(daily_rets.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0

    win_mask = pnls_arr > 0
    return {
        "trades":       n_trades,
        "cagr":         cagr,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "calmar":       calmar,
        "win_rate":     float(win_mask.mean()),
        "avg_win":      float(pnls_arr[win_mask].mean()) if win_mask.any() else 0.0,
        "avg_loss":     float(pnls_arr[~win_mask].mean()) if (~win_mask).any() else 0.0,
        "trail_exits":  trail_exits,
        "ema_exits":    ema_exits,
        "trail_pct_exits": trail_exits / n_trades if n_trades else 0.0,
    }


# ── worker ────────────────────────────────────────────────────────────────────

def _run_batch(
    close:          np.ndarray,
    ema_cache:      dict[int, np.ndarray],
    combos:         list[tuple[int, int, int, int, float]],
    tc_frac:        float,
) -> list[dict | None]:
    results = []
    for ef, es, xf, xs, trail in combos:
        r = backtest(
            close,
            ema_cache[ef], ema_cache[es],
            ema_cache[xf], ema_cache[xs],
            trail, tc_frac,
        )
        if r is not None:
            r.update({"entry_fast": ef, "entry_slow": es,
                      "exit_fast": xf,  "exit_slow": xs,
                      "trail_stop_pct": trail})
        results.append(r)
    return results


# ── grid search ───────────────────────────────────────────────────────────────

def run_grid(qqq: pd.DataFrame, tc_bps: float = 5, n_jobs: int = -1) -> pd.DataFrame:
    close = qqq["close"].to_numpy()

    all_periods = sorted(set(ENTRY_FASTS + ENTRY_SLOWS + EXIT_FASTS + EXIT_SLOWS))
    logger.info(f"Precomputing {len(all_periods)} EMA series …")
    ema_cache = {p: qqq["close"].ewm(span=p, adjust=False).mean().to_numpy()
                 for p in all_periods}

    entry_pairs = [(ef, es) for ef, es in product(ENTRY_FASTS, ENTRY_SLOWS) if ef < es]
    exit_pairs  = [(xf, xs) for xf, xs in product(EXIT_FASTS,  EXIT_SLOWS)  if xf < xs]
    combos = [
        (ef, es, xf, xs, trail)
        for (ef, es), (xf, xs), trail in product(entry_pairs, exit_pairs, TRAIL_STOPS)
    ]
    logger.info(
        f"{len(entry_pairs)} entry × {len(exit_pairs)} exit × {len(TRAIL_STOPS)} stops "
        f"= {len(combos):,} combos"
    )

    tc_frac  = tc_bps / 10_000
    n_work   = cpu_count() if n_jobs < 0 else max(1, n_jobs)
    chunk    = math.ceil(len(combos) / n_work)
    batches  = [combos[i: i + chunk] for i in range(0, len(combos), chunk)]

    logger.info(f"Running {len(combos):,} combos across {len(batches)} workers …")
    batch_out = Parallel(n_jobs=n_jobs)(
        delayed(_run_batch)(close, ema_cache, b, tc_frac) for b in batches
    )

    rows = [r for batch in batch_out for r in batch if r is not None]
    if not rows:
        logger.error("No results — check data.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# ── reporting ─────────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame, n: int = 20, bh_cagr: float = 0.0) -> None:
    cols = [
        "rank", "entry_fast", "entry_slow", "exit_fast", "exit_slow",
        "trail_stop_pct", "sharpe", "cagr", "max_dd", "calmar",
        "win_rate", "trades", "trail_pct_exits",
    ]
    fmt = {
        "trail_stop_pct":  "{:.0%}".format,
        "sharpe":          "{:.3f}".format,
        "cagr":            "{:.2%}".format,
        "max_dd":          "{:.2%}".format,
        "calmar":          "{:.2f}".format,
        "win_rate":        "{:.2%}".format,
        "trades":          "{:.0f}".format,
        "trail_pct_exits": "{:.1%}".format,
    }

    print("\n" + "=" * 120)
    print(f"TOP {n} BY SHARPE  —  QQQ 2000→2026  (buy-hold CAGR ≈ {bh_cagr:.2%})")
    print("=" * 120)
    top = df.head(n)[cols].copy()
    for col, fn in fmt.items():
        top[col] = top[col].map(fn)
    print(top.to_string(index=False))

    print("\n" + "-" * 120)
    print("TOP 10 BY CALMAR  (best risk-adjusted)")
    print("-" * 120)
    top_cal = df.nlargest(10, "calmar")[cols].copy()
    for col, fn in fmt.items():
        top_cal[col] = top_cal[col].map(fn)
    print(top_cal.to_string(index=False))

    print("\n" + "-" * 120)
    print("TOP 10 BY CAGR  (highest raw return)")
    print("-" * 120)
    top_cagr = df.nlargest(10, "cagr")[cols].copy()
    for col, fn in fmt.items():
        top_cagr[col] = top_cagr[col].map(fn)
    print(top_cagr.to_string(index=False))

    # trailing stop sensitivity
    print("\n" + "-" * 80)
    print("TRAILING STOP SENSITIVITY  (median Sharpe per stop level)")
    print("-" * 80)
    ts = df.groupby("trail_stop_pct")["sharpe"].median().reset_index()
    ts["trail_stop_pct"] = ts["trail_stop_pct"].map("{:.0%}".format)
    ts["sharpe"]         = ts["sharpe"].map("{:.3f}".format)
    print(ts.to_string(index=False))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-bps",  type=float, default=TC_BPS)
    parser.add_argument("--top-n",   type=int,   default=20)
    parser.add_argument("--jobs",    type=int,   default=-1)
    parser.add_argument("--output",  default="ema_qqq_grid_results.csv")
    args = parser.parse_args()

    qqq    = load_qqq()
    close  = qqq["close"].to_numpy()
    bh_ret = (close[-1] - close[0]) / close[0]
    years  = (qqq["timestamp"].iloc[-1] - qqq["timestamp"].iloc[0]).days / 365.25
    bh_cagr = (1 + bh_ret) ** (1 / years) - 1

    results = run_grid(qqq, tc_bps=args.tc_bps, n_jobs=args.jobs)

    print_results(results, n=args.top_n, bh_cagr=bh_cagr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / args.output
    results.to_csv(out, index=False)
    logger.info(f"Full results → {out}")


if __name__ == "__main__":
    main()
