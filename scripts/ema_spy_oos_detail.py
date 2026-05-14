#!/usr/bin/env python3
# Author: Elliott Middleton, assisted by Claude
# Date: 2026-05-14
# Description: Year-by-year OOS breakdown (2020-2026) for candidate SPY EMA parameter sets vs buy-hold.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
TEST_START = "2020-01-01"
TC_BPS     = 5
TC_FRAC    = TC_BPS / 10_000

CANDIDATES = [
    {"label": "IS Winner  EMA(12/16)+EMA(8/10)+2%trail",  "ef": 12, "es": 16, "xf": 8,  "xs": 10, "trail": 0.02},
    {"label": "IS <30%stp EMA(12/16)+EMA(8/10)+6%trail",  "ef": 12, "es": 16, "xf": 8,  "xs": 10, "trail": 0.06},
    {"label": "Wider stop EMA(12/16)+EMA(8/10)+8%trail",  "ef": 12, "es": 16, "xf": 8,  "xs": 10, "trail": 0.08},
    {"label": "Current    EMA(7/11)+EMA(17/20)+4%trail",  "ef": 7,  "es": 11, "xf": 17, "xs": 20, "trail": 0.04},
]


def load_spy() -> pd.DataFrame:
    path = DATA_DIR / "spy_qqq_2000_daily.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    spy = df[df["symbol"] == "SPY"].sort_values("timestamp").reset_index(drop=True)
    spy["date"] = spy["timestamp"].dt.normalize()
    return spy


def add_ema(close: pd.Series, p: int) -> np.ndarray:
    return close.ewm(span=p, adjust=False).mean().to_numpy()


def backtest_full(
    close:     np.ndarray,
    dates:     np.ndarray,
    ef_arr:    np.ndarray,
    es_arr:    np.ndarray,
    xf_arr:    np.ndarray,
    xs_arr:    np.ndarray,
    trail_pct: float,
    start_idx: int,
) -> tuple[np.ndarray, list[dict]]:
    """Returns equity curve (same length as close[start_idx:]) and trade log."""
    n     = len(close)
    n_sub = n - start_idx
    eq    = np.ones(n_sub)
    running_eq  = 1.0
    in_pos      = False
    entry_price = 0.0
    entry_date  = None
    peak        = 0.0
    prev_rel    = 0.0
    trades      = []

    for k in range(1, n_sub):
        i   = start_idx + k
        sig = i - 1

        entry_cross = (sig > 0 and ef_arr[sig] > es_arr[sig] and ef_arr[sig-1] <= es_arr[sig-1])
        exit_cross  = (sig > 0 and xf_arr[sig] < xs_arr[sig] and xf_arr[sig-1] >= xs_arr[sig-1])

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close[i]
            entry_date  = dates[i]
            peak        = close[i]
            prev_rel    = 0.0
            running_eq *= (1 - TC_FRAC)

        if in_pos:
            peak = max(peak, close[i])
            rel  = (close[i] - entry_price) / entry_price
            running_eq *= (1 + rel - prev_rel)
            prev_rel = rel

            trail_hit = close[i] <= peak * (1 - trail_pct)
            if trail_hit or exit_cross:
                running_eq *= (1 - TC_FRAC)
                trades.append({
                    "entry": str(entry_date)[:10],
                    "exit":  str(dates[i])[:10],
                    "pnl":   rel - 2 * TC_FRAC,
                    "type":  "trail" if trail_hit else "ema",
                })
                in_pos = False
                prev_rel = 0.0

        eq[k] = running_eq

    return eq, trades


def year_by_year_eq(eq: np.ndarray, dates: pd.Series, start_idx: int) -> dict[int, float]:
    """Returns annual return keyed by year."""
    sub_dates = dates.iloc[start_idx:].reset_index(drop=True)
    years_by_bar = sub_dates.dt.year
    out = {}
    for yr in sorted(years_by_bar.unique()):
        mask = years_by_bar == yr
        idx  = np.where(mask)[0]
        out[yr] = float(eq[idx[-1]] / eq[idx[0]] - 1)
    return out


def overall_metrics(eq: np.ndarray, n_bars: int) -> dict:
    rets   = np.diff(np.log(np.maximum(eq, 1e-12)))
    years  = (n_bars - 1) / 252.0
    cagr   = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else 0.0
    vol    = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak) / peak
    max_dd = float(dd.min())
    return {"cagr": cagr, "sharpe": sharpe, "vol": vol, "max_dd": max_dd}


def bh_year_by_year(close: np.ndarray, dates: pd.Series, start_idx: int) -> dict[int, float]:
    sub_close = close[start_idx:]
    sub_dates = dates.iloc[start_idx:].reset_index(drop=True)
    out = {}
    for yr in sorted(sub_dates.dt.year.unique()):
        mask = sub_dates.dt.year == yr
        idx  = np.where(mask)[0]
        out[yr] = float(sub_close[idx[-1]] / sub_close[idx[0]] - 1)
    return out


def main() -> None:
    spy        = load_spy()
    close      = spy["close"].to_numpy()
    dates_s    = spy["date"]
    dates_arr  = dates_s.to_numpy()
    test_mask  = dates_s >= TEST_START
    start_idx  = int((~test_mask).sum())

    print(f"\nOOS period: {dates_s.iloc[start_idx].date()} → {dates_s.iloc[-1].date()}"
          f"  ({len(close) - start_idx} bars)")
    print(f"TC: {TC_BPS}bps each side\n")

    # Pre-compute all needed EMA periods
    needed_periods = set()
    for c in CANDIDATES:
        needed_periods.update([c["ef"], c["es"], c["xf"], c["xs"]])
    ema_cache = {p: add_ema(spy["close"], p) for p in needed_periods}

    # Run each candidate
    results = []
    for c in CANDIDATES:
        eq, trades = backtest_full(
            close, dates_arr,
            ema_cache[c["ef"]], ema_cache[c["es"]],
            ema_cache[c["xf"]], ema_cache[c["xs"]],
            c["trail"], start_idx,
        )
        m      = overall_metrics(eq, len(eq))
        yy     = year_by_year_eq(eq, dates_s, start_idx)
        n_t    = len(trades)
        wins   = sum(1 for t in trades if t["pnl"] > 0)
        trails = sum(1 for t in trades if t["type"] == "trail")
        results.append({
            "label":      c["label"],
            "eq":         eq,
            "yy":         yy,
            "metrics":    m,
            "n_trades":   n_t,
            "win_rate":   wins / n_t if n_t else 0.0,
            "pct_trail":  trails / n_t if n_t else 0.0,
            "trades":     trades,
        })

    # Buy-hold baseline
    bh_close = close[start_idx:]
    bh_eq    = bh_close / bh_close[0]
    bh_m     = overall_metrics(bh_eq, len(bh_eq))
    bh_yy    = bh_year_by_year(close, dates_s, start_idx)
    all_years = sorted(bh_yy.keys())

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    WIDTH = 20
    header_labels = [r["label"][:WIDTH] for r in results] + ["Buy-Hold SPY"]
    col_w = 16

    print("=" * (10 + col_w * (len(results) + 1)))
    print("SUMMARY METRICS  —  OOS 2020–2026")
    print("=" * (10 + col_w * (len(results) + 1)))
    print(f"{'':10}" + "".join(f"{h:>{col_w}}" for h in header_labels))
    print("-" * (10 + col_w * (len(results) + 1)))

    for metric, label, fmt in [
        ("cagr",   "CAGR",       "{:.2%}"),
        ("vol",    "Ann. vol",    "{:.2%}"),
        ("sharpe", "Sharpe",      "{:.3f}"),
        ("max_dd", "Max DD",      "{:.2%}"),
    ]:
        row = f"{label:<10}"
        for r in results:
            row += f"{fmt.format(r['metrics'][metric]):>{col_w}}"
        row += f"{fmt.format(bh_m[metric]):>{col_w}}"
        print(row)

    print("-" * (10 + col_w * (len(results) + 1)))
    row = f"{'Trades':<10}"
    for r in results:
        row += f"{r['n_trades']:>{col_w}}"
    row += f"{'—':>{col_w}}"
    print(row)

    row = f"{'Win rate':<10}"
    for r in results:
        s = f"{r['win_rate']:.1%}"
        row += f"{s:>{col_w}}"
    row += f"{'—':>{col_w}}"
    print(row)

    row = f"{'% trail':<10}"
    for r in results:
        s = f"{r['pct_trail']:.0%}"
        row += f"{s:>{col_w}}"
    row += f"{'—':>{col_w}}"
    print(row)

    # ── YEAR-BY-YEAR TABLE ────────────────────────────────────────────────────
    print()
    print("=" * (8 + col_w * (len(results) + 1)))
    print("YEAR-BY-YEAR RETURNS  —  OOS 2020–2026")
    print("=" * (8 + col_w * (len(results) + 1)))
    print(f"{'Year':>8}" + "".join(f"{h:>{col_w}}" for h in header_labels))
    print("-" * (8 + col_w * (len(results) + 1)))

    for yr in all_years:
        row = f"{yr:>8}"
        vals = [r["yy"].get(yr, float("nan")) for r in results]
        bh_v = bh_yy.get(yr, float("nan"))
        for v in vals:
            if not np.isnan(v):
                row += f"{v:>+{col_w}.2%}"
            else:
                row += f"{'—':>{col_w}}"
        row += f"{bh_v:>+{col_w}.2%}"
        print(row)

    # ── FULL PERIOD ───────────────────────────────────────────────────────────
    print("-" * (8 + col_w * (len(results) + 1)))
    row = f"{'Full':>8}"
    for r in results:
        total = float(r["eq"][-1] - 1)
        row += f"{total:>+{col_w}.2%}"
    bh_total = float(bh_eq[-1] - 1)
    row += f"{bh_total:>+{col_w}.2%}"
    print(row)

    print()


if __name__ == "__main__":
    main()
