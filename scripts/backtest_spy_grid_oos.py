#!/usr/bin/env python3
# Author: Elliott Middleton, assisted by Claude
# Date: 2026-05-14
# Description: SPY EMA grid search on 2000-2019 (IS), evaluate top params on 2020-2026 (OOS).

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

TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"

ENTRY_FASTS = list(range(3, 13))
ENTRY_SLOWS = list(range(8, 27))
EXIT_FASTS  = list(range(3, 13))
EXIT_SLOWS  = list(range(8, 27))
TRAIL_STOPS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

# Current SPY params (from CLAUDE.md) — included as reference row
CURRENT_PARAMS = {"entry_fast": 7, "entry_slow": 11, "exit_fast": 17, "exit_slow": 20, "trail_stop_pct": 0.04}


def load_spy() -> pd.DataFrame:
    path = DATA_DIR / "spy_qqq_2000_daily.parquet"
    if not path.exists():
        logger.error(f"Not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    spy = df[df["symbol"] == "SPY"].sort_values("timestamp").reset_index(drop=True)
    spy["date"] = spy["timestamp"].dt.normalize()
    logger.info(f"SPY: {len(spy):,} bars  {spy['date'].min().date()} → {spy['date'].max().date()}")
    return spy


def precompute_emas(close: np.ndarray, periods: list[int]) -> dict[int, np.ndarray]:
    # Computed on FULL history so test-period EMAs reflect training history
    s = pd.Series(close)
    return {p: s.ewm(span=p, adjust=False).mean().to_numpy() for p in periods}


def backtest_slice(
    close:          np.ndarray,
    ema_ef:         np.ndarray,
    ema_es:         np.ndarray,
    ema_xf:         np.ndarray,
    ema_xs:         np.ndarray,
    trail_pct:      float,
    tc_frac:        float,
    start_idx:      int,
    end_idx:        int,
) -> dict | None:
    """
    Evaluate a parameter set on bars [start_idx, end_idx).
    Begins with no open position regardless of prior history.
    EMA arrays span the full dataset, so signal quality at start_idx
    reflects the full warmup.
    """
    n = end_idx - start_idx
    if n < 2:
        return None

    in_pos      = False
    entry_price = 0.0
    peak        = 0.0
    daily_rets  = np.zeros(n)
    prev_eq     = 0.0
    pnls        = []
    trail_exits = 0
    ema_exits   = 0

    for k in range(1, n):
        i   = start_idx + k       # absolute index
        sig = i - 1               # signal bar

        if sig >= 1:
            entry_cross = (ema_ef[sig] > ema_es[sig] and ema_ef[sig - 1] <= ema_es[sig - 1])
            exit_cross  = (ema_xf[sig] < ema_xs[sig] and ema_xf[sig - 1] >= ema_xs[sig - 1])
        else:
            entry_cross = exit_cross = False

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close[i]
            peak        = close[i]
            prev_eq     = 0.0
            daily_rets[k] -= tc_frac

        if in_pos:
            peak = max(peak, close[i])
            eq   = (close[i] - entry_price) / entry_price
            daily_rets[k] += eq - prev_eq
            prev_eq = eq

            trail_hit = close[i] <= peak * (1 - trail_pct)
            if trail_hit or exit_cross:
                daily_rets[k] -= tc_frac
                pnls.append(eq - 2 * tc_frac)
                trail_exits += (1 if trail_hit else 0)
                ema_exits   += (0 if trail_hit else 1)
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
    base     = float(cum[-1])
    cagr     = base ** (1 / years) - 1 if years > 0 and base > 0 else 0.0
    vol      = daily_rets.std()
    sharpe   = float(daily_rets.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0
    calmar   = abs(cagr / max_dd) if max_dd < 0 else 0.0
    win_mask = pnls_arr > 0

    return {
        "trades":          n_trades,
        "cagr":            cagr,
        "sharpe":          sharpe,
        "max_dd":          max_dd,
        "calmar":          calmar,
        "win_rate":        float(win_mask.mean()),
        "avg_win":         float(pnls_arr[win_mask].mean())  if win_mask.any()  else 0.0,
        "avg_loss":        float(pnls_arr[~win_mask].mean()) if (~win_mask).any() else 0.0,
        "trail_exits":     trail_exits,
        "ema_exits":       ema_exits,
        "trail_pct_exits": trail_exits / n_trades,
    }


def _run_batch(
    close:      np.ndarray,
    ema_cache:  dict[int, np.ndarray],
    combos:     list[tuple],
    tc_frac:    float,
    start_idx:  int,
    end_idx:    int,
) -> list[dict | None]:
    out = []
    for ef, es, xf, xs, trail in combos:
        r = backtest_slice(
            close, ema_cache[ef], ema_cache[es],
            ema_cache[xf], ema_cache[xs],
            trail, tc_frac, start_idx, end_idx,
        )
        if r is not None:
            r.update({"entry_fast": ef, "entry_slow": es,
                      "exit_fast": xf,  "exit_slow": xs,
                      "trail_stop_pct": trail})
        out.append(r)
    return out


def run_grid(
    close:      np.ndarray,
    ema_cache:  dict[int, np.ndarray],
    start_idx:  int,
    end_idx:    int,
    tc_bps:     float,
    n_jobs:     int = -1,
) -> pd.DataFrame:
    tc_frac  = tc_bps / 10_000

    entry_pairs = [(ef, es) for ef, es in product(ENTRY_FASTS, ENTRY_SLOWS) if ef < es]
    exit_pairs  = [(xf, xs) for xf, xs in product(EXIT_FASTS,  EXIT_SLOWS)  if xf < xs]
    combos = [
        (ef, es, xf, xs, trail)
        for (ef, es), (xf, xs), trail in product(entry_pairs, exit_pairs, TRAIL_STOPS)
    ]
    logger.info(
        f"{len(entry_pairs)} entry × {len(exit_pairs)} exit × {len(TRAIL_STOPS)} stops "
        f"= {len(combos):,} combos  (bars {start_idx}–{end_idx})"
    )

    n_work  = cpu_count() if n_jobs < 0 else max(1, n_jobs)
    chunk   = math.ceil(len(combos) / n_work)
    batches = [combos[i: i + chunk] for i in range(0, len(combos), chunk)]

    batch_out = Parallel(n_jobs=n_jobs)(
        delayed(_run_batch)(close, ema_cache, b, tc_frac, start_idx, end_idx)
        for b in batches
    )
    rows = [r for batch in batch_out for r in batch if r is not None]
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def bh_metrics(close: np.ndarray, start_idx: int, end_idx: int) -> dict:
    c    = close[start_idx:end_idx]
    rets = np.diff(np.log(c))
    years = (end_idx - start_idx - 1) / 252.0
    cagr  = (c[-1] / c[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol   = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    peak   = np.maximum.accumulate(c / c[0])
    dd     = ((c / c[0]) - peak) / peak
    max_dd = float(dd.min())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "vol": vol}


def print_comparison(
    is_df:      pd.DataFrame,
    oos_rows:   list[dict],
    train_bh:   dict,
    test_bh:    dict,
    top_n:      int,
    current_is: dict | None,
    current_oos: dict | None,
) -> None:
    cols = ["rank", "entry_fast", "entry_slow", "exit_fast", "exit_slow",
            "trail_stop_pct", "sharpe", "cagr", "max_dd", "calmar", "win_rate", "trades"]
    fmt  = {
        "trail_stop_pct": "{:.0%}".format,
        "sharpe":         "{:.3f}".format,
        "cagr":           "{:.2%}".format,
        "max_dd":         "{:.2%}".format,
        "calmar":         "{:.2f}".format,
        "win_rate":       "{:.1%}".format,
        "trades":         "{:.0f}".format,
    }

    print("\n" + "=" * 120)
    print(f"IN-SAMPLE  2000–2019  |  buy-hold SPY CAGR {train_bh['cagr']:.2%}  Sharpe {train_bh['sharpe']:.3f}")
    print("=" * 120)
    top = is_df.head(top_n)[cols].copy()
    for col, fn in fmt.items():
        top[col] = top[col].map(fn)
    print(top.to_string(index=False))

    if current_is:
        print(f"\n  ── Current params IS ──  "
              f"entry({current_is.get('entry_fast')}/{current_is.get('entry_slow')})  "
              f"exit({current_is.get('exit_fast')}/{current_is.get('exit_slow')})  "
              f"trail {current_is.get('trail_stop_pct'):.0%}  |  "
              f"Sharpe {current_is.get('sharpe', float('nan')):.3f}  "
              f"CAGR {current_is.get('cagr', float('nan')):.2%}  "
              f"MaxDD {current_is.get('max_dd', float('nan')):.2%}  "
              f"Trades {current_is.get('trades', '?'):.0f}")

    print("\n" + "=" * 120)
    print(f"OUT-OF-SAMPLE  2020–2026  |  buy-hold SPY CAGR {test_bh['cagr']:.2%}  Sharpe {test_bh['sharpe']:.3f}")
    print("  (same parameter sets, evaluated on held-out data — no refitting)")
    print("=" * 120)
    oos_df = pd.DataFrame(oos_rows).sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    oos_df.insert(0, "oos_rank", oos_df.index + 1)
    oos_cols = ["oos_rank", "is_rank", "entry_fast", "entry_slow", "exit_fast", "exit_slow",
                "trail_stop_pct", "oos_sharpe", "oos_cagr", "oos_max_dd", "oos_win_rate", "oos_trades"]
    oos_fmt = {
        "trail_stop_pct": "{:.0%}".format,
        "oos_sharpe":     "{:.3f}".format,
        "oos_cagr":       "{:.2%}".format,
        "oos_max_dd":     "{:.2%}".format,
        "oos_win_rate":   "{:.1%}".format,
        "oos_trades":     "{:.0f}".format,
    }
    oos_top = oos_df[oos_cols].copy()
    for col, fn in oos_fmt.items():
        if col in oos_top.columns:
            oos_top[col] = oos_top[col].map(fn)
    print(oos_top.to_string(index=False))

    if current_oos:
        print(f"\n  ── Current params OOS ──  "
              f"entry({CURRENT_PARAMS['entry_fast']}/{CURRENT_PARAMS['entry_slow']})  "
              f"exit({CURRENT_PARAMS['exit_fast']}/{CURRENT_PARAMS['exit_slow']})  "
              f"trail {CURRENT_PARAMS['trail_stop_pct']:.0%}  |  "
              f"Sharpe {current_oos.get('sharpe', float('nan')):.3f}  "
              f"CAGR {current_oos.get('cagr', float('nan')):.2%}  "
              f"MaxDD {current_oos.get('max_dd', float('nan')):.2%}  "
              f"Trades {current_oos.get('trades', '?'):.0f}")

    print("\n" + "-" * 80)
    print("TRAILING STOP SENSITIVITY — IS median Sharpe per level")
    print("-" * 80)
    ts = is_df.groupby("trail_stop_pct")["sharpe"].median().reset_index()
    ts["trail_stop_pct"] = ts["trail_stop_pct"].map("{:.0%}".format)
    ts["sharpe"]         = ts["sharpe"].map("{:.3f}".format)
    print(ts.to_string(index=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SPY EMA grid search: IS 2000-2019 → OOS 2020-2026"
    )
    parser.add_argument("--tc-bps",  type=float, default=5)
    parser.add_argument("--top-n",   type=int,   default=20)
    parser.add_argument("--jobs",    type=int,   default=-1)
    args = parser.parse_args()

    spy   = load_spy()
    close = spy["close"].to_numpy()
    dates = spy["date"]

    train_mask = dates <= TRAIN_END
    test_mask  = dates >= TEST_START
    train_end_idx = int(train_mask.sum())   # exclusive upper bound for train
    test_start_idx = int((~test_mask).sum()) # first test bar

    logger.info(
        f"Train: bars 0–{train_end_idx - 1}  "
        f"({dates.iloc[0].date()} → {dates.iloc[train_end_idx - 1].date()})"
    )
    logger.info(
        f"Test:  bars {test_start_idx}–{len(close) - 1}  "
        f"({dates.iloc[test_start_idx].date()} → {dates.iloc[-1].date()})"
    )

    all_periods = sorted(set(ENTRY_FASTS + ENTRY_SLOWS + EXIT_FASTS + EXIT_SLOWS))
    logger.info(f"Precomputing {len(all_periods)} EMA series on full history …")
    ema_cache = precompute_emas(close, all_periods)

    # IS grid
    logger.info("Running in-sample grid search …")
    is_df = run_grid(close, ema_cache, 0, train_end_idx, args.tc_bps, args.jobs)

    # Buy-hold benchmarks
    train_bh = bh_metrics(close, 0, train_end_idx)
    test_bh  = bh_metrics(close, test_start_idx, len(close))

    # OOS evaluation for top N IS params
    tc_frac = args.tc_bps / 10_000
    oos_rows = []
    for _, row in is_df.head(args.top_n).iterrows():
        ef, es = int(row["entry_fast"]), int(row["entry_slow"])
        xf, xs = int(row["exit_fast"]),  int(row["exit_slow"])
        trail  = float(row["trail_stop_pct"])
        r = backtest_slice(
            close, ema_cache[ef], ema_cache[es],
            ema_cache[xf], ema_cache[xs],
            trail, tc_frac, test_start_idx, len(close),
        )
        if r is None:
            r = {"trades": 0, "cagr": 0, "sharpe": 0, "max_dd": 0,
                 "win_rate": 0, "calmar": 0}
        oos_rows.append({
            "is_rank":        int(row["rank"]),
            "is_sharpe":      float(row["sharpe"]),
            "entry_fast":     ef, "entry_slow": es,
            "exit_fast":      xf, "exit_slow":  xs,
            "trail_stop_pct": trail,
            "oos_sharpe":     r["sharpe"],
            "oos_cagr":       r["cagr"],
            "oos_max_dd":     r["max_dd"],
            "oos_calmar":     r["calmar"],
            "oos_win_rate":   r["win_rate"],
            "oos_trades":     r["trades"],
        })

    # Current params IS and OOS
    cp = CURRENT_PARAMS
    current_is = backtest_slice(
        close, ema_cache[cp["entry_fast"]], ema_cache[cp["entry_slow"]],
        ema_cache[cp["exit_fast"]], ema_cache[cp["exit_slow"]],
        cp["trail_stop_pct"], tc_frac, 0, train_end_idx,
    )
    if current_is:
        current_is.update(cp)
    current_oos = backtest_slice(
        close, ema_cache[cp["entry_fast"]], ema_cache[cp["entry_slow"]],
        ema_cache[cp["exit_fast"]], ema_cache[cp["exit_slow"]],
        cp["trail_stop_pct"], tc_frac, test_start_idx, len(close),
    )

    print_comparison(is_df, oos_rows, train_bh, test_bh,
                     args.top_n, current_is, current_oos)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_df.to_csv(OUT_DIR / "ema_spy_grid_is.csv", index=False)
    pd.DataFrame(oos_rows).to_csv(OUT_DIR / "ema_spy_grid_oos.csv", index=False)
    logger.info("Saved: ema_spy_grid_is.csv  |  ema_spy_grid_oos.csv")


if __name__ == "__main__":
    main()
