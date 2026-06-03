#!/usr/bin/env python3
# Author: Elliott Middleton, assisted by Claude
# Date: 2026-06-03
# Description: Sweep IS lookback window (60–180 months, step 12) in walk-forward MA
#              optimization to find the optimal lookback for SPY and QQQ (2010–2026 OOS).
#              Grid: MA entry/exit pairs × trailing stop 3–6%. IS objective: Sharpe.
#              MA type selectable: ema (default) or sma.
#              Outputs: CSV, PNG chart, and date-stamped markdown report to results/.

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data" / "raw"
RESULT_DIR = Path(__file__).parent.parent / "results"

SYMBOLS         = ["SPY", "QQQ"]
WF_YEARS        = list(range(2010, 2027))
LOOKBACK_MONTHS = list(range(60, 181, 12))   # 60, 72, 84, …, 180 (11 values)
TC_BPS          = 5
TRAIL_STOPS     = [0.03, 0.04, 0.05, 0.06]
MIN_IS_BARS     = 200


# ── data ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = DATA_DIR / "spy_qqq_2000_daily.parquet"
    if not path.exists():
        logger.error(f"Not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        f"Loaded {len(df):,} bars · {df['symbol'].nunique()} symbols · "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    )
    return df


# ── parameter grid ────────────────────────────────────────────────────────────

def build_combos() -> list[tuple[int, int, int, int, float]]:
    pairs = [(f, s) for f in range(3, 13) for s in range(8, 27) if f < s]
    combos = [
        (ef, es, xf, xs, ts)
        for (ef, es), (xf, xs), ts in product(pairs, pairs, TRAIL_STOPS)
    ]
    logger.info(
        f"Grid: {len(pairs)} entry pairs × {len(pairs)} exit pairs "
        f"× {len(TRAIL_STOPS)} stops = {len(combos):,} combos"
    )
    return combos


# ── MA computation ─────────────────────────────────────────────────────────────

def add_mas(df: pd.DataFrame, periods: list[int], ma_type: str) -> pd.DataFrame:
    df = df.copy()
    for p in sorted(set(periods)):
        if ma_type == "ema":
            df[f"ma_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        else:
            df[f"ma_{p}"] = df["close"].rolling(window=p, min_periods=p).mean()
    return df


# ── fast IS backtest — Sharpe only ────────────────────────────────────────────

def _sharpe_only(
    close_arr: np.ndarray,
    ef_arr: np.ndarray,
    es_arr: np.ndarray,
    xf_arr: np.ndarray,
    xs_arr: np.ndarray,
    trail_pct: float,
    tc_frac: float,
) -> float:
    in_pos = False
    entry_price = peak = prev_eq = 0.0
    daily_sum = daily_sum_sq = 0.0
    n_days = 0

    for i in range(2, len(close_arr)):
        sig = i - 1
        if (np.isnan(ef_arr[sig]) or np.isnan(es_arr[sig]) or
                np.isnan(xf_arr[sig]) or np.isnan(xs_arr[sig]) or
                np.isnan(ef_arr[sig - 1]) or np.isnan(es_arr[sig - 1]) or
                np.isnan(xf_arr[sig - 1]) or np.isnan(xs_arr[sig - 1])):
            continue
        entry_cross = ef_arr[sig] > es_arr[sig] and ef_arr[sig - 1] <= es_arr[sig - 1]
        exit_cross  = xf_arr[sig] < xs_arr[sig] and xf_arr[sig - 1] >= xs_arr[sig - 1]
        daily_ret   = 0.0

        if not in_pos and entry_cross:
            in_pos = True
            entry_price = peak = close_arr[i]
            prev_eq = 0.0
            daily_ret -= tc_frac

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            eq         = (close_arr[i] - entry_price) / entry_price
            daily_ret += eq - prev_eq
            prev_eq    = eq
            if close_arr[i] <= peak * (1.0 - trail_pct) or exit_cross:
                daily_ret -= tc_frac
                in_pos  = False
                prev_eq = 0.0

        daily_sum    += daily_ret
        daily_sum_sq += daily_ret * daily_ret
        n_days       += 1

    if n_days < 2:
        return 0.0
    mean = daily_sum / n_days
    var  = daily_sum_sq / n_days - mean * mean
    return float(mean / var ** 0.5 * 252 ** 0.5) if var > 0 else 0.0


def _run_is_batch(
    close_arr:   np.ndarray,
    ma_dict:     dict[int, np.ndarray],
    combo_batch: list[tuple[int, int, int, int, float]],
    tc_frac:     float,
) -> list[tuple[float, tuple[int, int, int, int, float]]]:
    return [
        (
            _sharpe_only(
                close_arr,
                ma_dict[ef], ma_dict[es],
                ma_dict[xf], ma_dict[xs],
                ts, tc_frac,
            ),
            (ef, es, xf, xs, ts),
        )
        for ef, es, xf, xs, ts in combo_batch
    ]


# ── full OOS backtest — all metrics ──────────────────────────────────────────

def backtest_full(
    sym_df: pd.DataFrame,
    ef: int, es: int,
    xf: int, xs: int,
    trail_pct: float,
    tc_frac: float,
) -> dict:
    close_arr = sym_df["close"].to_numpy()
    ef_arr    = sym_df[f"ma_{ef}"].to_numpy()
    es_arr    = sym_df[f"ma_{es}"].to_numpy()
    xf_arr    = sym_df[f"ma_{xf}"].to_numpy()
    xs_arr    = sym_df[f"ma_{xs}"].to_numpy()
    n         = len(close_arr)

    in_pos = False
    entry_price = peak = prev_eq = 0.0
    pnls:       list[float] = []
    daily_rets  = np.zeros(n)
    trail_exits = ma_exits = 0

    for i in range(2, n):
        sig = i - 1
        if (np.isnan(ef_arr[sig]) or np.isnan(es_arr[sig]) or
                np.isnan(xf_arr[sig]) or np.isnan(xs_arr[sig]) or
                np.isnan(ef_arr[sig - 1]) or np.isnan(es_arr[sig - 1]) or
                np.isnan(xf_arr[sig - 1]) or np.isnan(xs_arr[sig - 1])):
            continue
        entry_cross = ef_arr[sig] > es_arr[sig] and ef_arr[sig - 1] <= es_arr[sig - 1]
        exit_cross  = xf_arr[sig] < xs_arr[sig] and xf_arr[sig - 1] >= xs_arr[sig - 1]

        if not in_pos and entry_cross:
            in_pos = True
            entry_price = peak = close_arr[i]
            prev_eq = 0.0
            daily_rets[i] -= tc_frac

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            eq             = (close_arr[i] - entry_price) / entry_price
            daily_rets[i] += eq - prev_eq
            prev_eq        = eq
            trail_hit      = close_arr[i] <= peak * (1.0 - trail_pct)
            if trail_hit or exit_cross:
                daily_rets[i] -= tc_frac
                pnls.append(eq - tc_frac)
                trail_exits += int(trail_hit)
                ma_exits    += int(not trail_hit)
                in_pos  = False
                prev_eq = 0.0

    pnls_arr  = np.array(pnls) if pnls else np.zeros(1)
    n_trades  = len(pnls)
    strat_ret = float(np.prod(1.0 + pnls_arr) - 1.0) if pnls else 0.0

    days  = (sym_df["timestamp"].iloc[-1] - sym_df["timestamp"].iloc[0]).days
    years = days / 365.25
    base  = 1.0 + strat_ret
    cagr  = float(base ** (1.0 / years) - 1.0) if years > 0 and base > 0 else 0.0

    vol    = daily_rets.std()
    sharpe = float(daily_rets.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    eq_curve    = np.cumprod(1.0 + daily_rets)
    running_max = np.maximum.accumulate(eq_curve)
    max_dd      = float((eq_curve / running_max - 1.0).min())

    win_rate = float((pnls_arr > 0).mean()) if n_trades > 0 else 0.0
    bh_ret   = float((close_arr[-1] - close_arr[0]) / close_arr[0]) if len(close_arr) > 1 else 0.0

    return {
        "trades":        n_trades,
        "oos_strat_ret": strat_ret,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "win_rate":      win_rate,
        "trail_exits":   trail_exits,
        "ma_exits":      ma_exits,
        "bh_return":     bh_ret,
    }


# ── lookback sweep ────────────────────────────────────────────────────────────

def run_lookback_sweep(df: pd.DataFrame, ma_type: str, n_jobs: int = -1) -> pd.DataFrame:
    combos  = build_combos()
    tc_frac = TC_BPS / 10_000

    all_periods = sorted({p for c in combos for p in c[:4]})
    n_workers   = cpu_count() if n_jobs < 0 else max(1, n_jobs)
    chunk       = math.ceil(len(combos) / n_workers)
    batches     = [combos[i : i + chunk] for i in range(0, len(combos), chunk)]
    logger.info(f"MA type={ma_type.upper()}  Parallel workers={n_workers}  batch size={chunk}")

    n_total = len(LOOKBACK_MONTHS) * len(WF_YEARS) * len(SYMBOLS)
    logger.info(
        f"Sweep: {len(LOOKBACK_MONTHS)} lookbacks × {len(WF_YEARS)} OOS years "
        f"× {len(SYMBOLS)} symbols = {n_total} IS grid searches"
    )

    rows = []

    for lb_idx, lb_months in enumerate(LOOKBACK_MONTHS):
        logger.info(
            f"\n{'='*70}\n"
            f"Lookback {lb_months} months ({lb_months/12:.1f} yr)  "
            f"[{lb_idx + 1}/{len(LOOKBACK_MONTHS)}]"
        )

        for oos_year in WF_YEARS:
            is_end_date   = pd.Timestamp(f"{oos_year - 1}-12-31")
            is_start_date = is_end_date - pd.DateOffset(months=lb_months)

            for symbol in SYMBOLS:
                sym_all = (
                    df[df["symbol"] == symbol]
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                is_mask = (
                    (sym_all["timestamp"] >= is_start_date) &
                    (sym_all["timestamp"] <= is_end_date)
                )
                is_df = add_mas(sym_all[is_mask].copy(), all_periods, ma_type)

                if len(is_df) < MIN_IS_BARS:
                    logger.debug(
                        f"  {symbol} {oos_year} lb={lb_months}m: "
                        f"only {len(is_df)} IS bars — skip"
                    )
                    continue

                is_close  = is_df["close"].to_numpy()
                is_ma_dict = {p: is_df[f"ma_{p}"].to_numpy() for p in all_periods}

                batch_res = Parallel(n_jobs=n_jobs)(
                    delayed(_run_is_batch)(is_close, is_ma_dict, batch, tc_frac)
                    for batch in batches
                )
                flat = [item for sub in batch_res for item in sub]
                best_is_sharpe, best_params = max(flat, key=lambda x: x[0])
                ef, es, xf, xs, ts = best_params

                logger.info(
                    f"  {symbol} {oos_year} lb={lb_months}m: "
                    f"IS Sharpe={best_is_sharpe:.3f}  "
                    f"{ma_type.upper()}({ef}/{es}) x {ma_type.upper()}({xf}/{xs}) trail={ts:.0%}"
                )

                combined_mask = (
                    (sym_all["timestamp"] >= is_start_date) &
                    (sym_all["timestamp"].dt.year <= oos_year)
                )
                combined_df = add_mas(sym_all[combined_mask].copy(), [ef, es, xf, xs], ma_type)
                oos_df      = combined_df[combined_df["timestamp"].dt.year == oos_year].copy()

                if len(oos_df) < 20:
                    logger.warning(
                        f"  {symbol} {oos_year} lb={lb_months}m: "
                        f"only {len(oos_df)} OOS bars — skip"
                    )
                    continue

                oos_metrics = backtest_full(oos_df, ef, es, xf, xs, ts, tc_frac)

                rows.append({
                    "lookback_months": lb_months,
                    "year":            oos_year,
                    "symbol":          symbol,
                    "entry_fast":      ef,
                    "entry_slow":      es,
                    "exit_fast":       xf,
                    "exit_slow":       xs,
                    "trailing_stop":   ts,
                    "is_sharpe":       best_is_sharpe,
                    "oos_strat_ret":   oos_metrics["oos_strat_ret"],
                    **{f"oos_{k}": v for k, v in oos_metrics.items()},
                })

    return pd.DataFrame(rows)


# ── aggregate summary ─────────────────────────────────────────────────────────

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lb in LOOKBACK_MONTHS:
        for sym in SYMBOLS:
            sub = results_df[
                (results_df["lookback_months"] == lb) &
                (results_df["symbol"] == sym)
            ]
            if sub.empty:
                continue
            hprs = sub["oos_strat_ret"].to_numpy()
            n    = len(sub)
            chain_ret  = float(np.prod(1.0 + hprs) - 1.0)
            chain_cagr = float((1.0 + chain_ret) ** (1.0 / n) - 1.0) if n > 0 else 0.0
            hit_rate   = float((hprs > 0).mean())
            rows.append({
                "lookback_months": lb,
                "lookback_years":  lb / 12,
                "symbol":          sym,
                "n_oos_years":     n,
                "chain_cagr":      chain_cagr,
                "hit_rate":        hit_rate,
                "avg_sharpe":      sub["oos_sharpe"].mean(),
                "avg_max_dd":      sub["oos_max_dd"].mean(),
                "avg_trades":      sub["oos_trades"].mean(),
                "avg_win_rate":    sub["oos_win_rate"].mean(),
                "avg_is_sharpe":   sub["is_sharpe"].mean(),
            })
    return pd.DataFrame(rows)


# ── console output ────────────────────────────────────────────────────────────

def print_console_results(
    results_df: pd.DataFrame, summary_df: pd.DataFrame, ma_type: str
) -> None:
    sep = "=" * 110
    mt  = ma_type.upper()
    print(f"\n{sep}")
    print(
        f"{mt} LOOKBACK SWEEP — AGGREGATE OOS RESULTS  "
        f"(Sharpe-optimized IS → OOS, 2010–2026, {TC_BPS} bps TC)"
    )
    print(sep)

    for sym in SYMBOLS:
        print(f"\n{'─' * 45}  {sym}  {'─' * 45}")
        sub = summary_df[summary_df["symbol"] == sym].copy()
        disp = sub[[
            "lookback_months", "lookback_years", "n_oos_years",
            "chain_cagr", "hit_rate", "avg_sharpe",
            "avg_max_dd", "avg_trades", "avg_win_rate", "avg_is_sharpe",
        ]].copy()
        disp["lookback_years"] = disp["lookback_years"].map("{:.0f}yr".format)
        disp["chain_cagr"]     = disp["chain_cagr"].map("{:+.2%}".format)
        disp["hit_rate"]       = disp["hit_rate"].map("{:.0%}".format)
        disp["avg_sharpe"]     = disp["avg_sharpe"].map("{:.3f}".format)
        disp["avg_max_dd"]     = disp["avg_max_dd"].map("{:.1%}".format)
        disp["avg_trades"]     = disp["avg_trades"].map("{:.1f}".format)
        disp["avg_win_rate"]   = disp["avg_win_rate"].map("{:.1%}".format)
        disp["avg_is_sharpe"]  = disp["avg_is_sharpe"].map("{:.3f}".format)
        print(disp.to_string(index=False))

        best_sharpe = sub.loc[sub["avg_sharpe"].idxmax()]
        best_cagr   = sub.loc[sub["chain_cagr"].idxmax()]
        print(
            f"\n  ★ Best avg OOS Sharpe : {int(best_sharpe['lookback_months'])} months "
            f"({best_sharpe['lookback_months']/12:.0f} yr)  "
            f"Sharpe={best_sharpe['avg_sharpe']:.3f}  "
            f"Chain CAGR={best_sharpe['chain_cagr']:+.2%}  "
            f"Hit={best_sharpe['hit_rate']:.0%}"
        )
        print(
            f"  ★ Best chain CAGR     : {int(best_cagr['lookback_months'])} months "
            f"({best_cagr['lookback_months']/12:.0f} yr)  "
            f"CAGR={best_cagr['chain_cagr']:+.2%}  "
            f"Sharpe={best_cagr['avg_sharpe']:.3f}"
        )

    print(f"\n{sep}\n")

    print(f"COMBINED RANKING  (avg OOS Sharpe across SPY + QQQ, equal-weight)")
    print("─" * 60)
    combined = (
        summary_df.groupby("lookback_months")["avg_sharpe"]
        .mean()
        .reset_index()
        .rename(columns={"avg_sharpe": "combined_avg_sharpe"})
        .sort_values("combined_avg_sharpe", ascending=False)
    )
    for rank, (_, row) in enumerate(combined.iterrows(), 1):
        star = "★" if rank == 1 else " "
        print(
            f"  {star} #{rank:2d}  {int(row['lookback_months']):3d} mo "
            f"({row['lookback_months']/12:.0f} yr)  "
            f"combined Sharpe = {row['combined_avg_sharpe']:.3f}"
        )
    print(f"\n{sep}\n")


# ── markdown report ───────────────────────────────────────────────────────────

def write_markdown(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ma_type:    str,
    today_str:  str,
) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mt      = ma_type.upper()
    md_path = RESULT_DIR / f"{ma_type}_lookback_sweep_{today_str}.md"

    lines: list[str] = []

    lines += [
        f"# {mt} IS Lookback Sweep — Results",
        f"",
        f"**Generated:** {today_str}  ",
        f"**OOS period:** 2010–2026  |  **IS objective:** Sharpe maximization  |  **TC:** {TC_BPS} bps  ",
        f"**Lookbacks tested:** {', '.join(str(lb) for lb in LOOKBACK_MONTHS)} months  ",
        f"**Grid:** {mt} entry (fast 3–12, slow 8–26) × {mt} exit (same) × trailing stop "
        f"{', '.join(f'{ts:.0%}' for ts in TRAIL_STOPS)}  ",
        f"",
    ]

    lines += ["## Executive Summary", ""]
    for sym in SYMBOLS:
        sub         = summary_df[summary_df["symbol"] == sym]
        best_sharpe = sub.loc[sub["avg_sharpe"].idxmax()]
        best_cagr   = sub.loc[sub["chain_cagr"].idxmax()]
        lines += [
            f"**{sym}:**",
            f"- Best avg OOS Sharpe: **{int(best_sharpe['lookback_months'])} months "
            f"({best_sharpe['lookback_months']/12:.0f} yr)** → "
            f"Sharpe={best_sharpe['avg_sharpe']:.3f}, "
            f"Chain CAGR={best_sharpe['chain_cagr']:+.2%}, "
            f"Hit rate={best_sharpe['hit_rate']:.0%}",
            f"- Best chain CAGR: **{int(best_cagr['lookback_months'])} months "
            f"({best_cagr['lookback_months']/12:.0f} yr)** → "
            f"CAGR={best_cagr['chain_cagr']:+.2%}, "
            f"Sharpe={best_cagr['avg_sharpe']:.3f}",
            f"",
        ]

    combined = (
        summary_df.groupby("lookback_months")["avg_sharpe"]
        .mean()
        .reset_index()
        .rename(columns={"avg_sharpe": "combined_avg_sharpe"})
        .sort_values("combined_avg_sharpe", ascending=False)
        .reset_index(drop=True)
    )
    lines += ["## Combined Ranking (avg OOS Sharpe across SPY + QQQ)", ""]
    lines += ["| Rank | Lookback (mo) | Lookback (yr) | Combined Avg Sharpe |"]
    lines += ["|---|---|---|---|"]
    for i, row in combined.iterrows():
        star = "★ " if i == 0 else ""
        lines.append(
            f"| {star}{i+1} | {int(row['lookback_months'])} | "
            f"{row['lookback_months']/12:.0f} | "
            f"{row['combined_avg_sharpe']:.3f} |"
        )
    lines += [""]

    for sym in SYMBOLS:
        lines += [f"## {sym} — Aggregate Results by Lookback", ""]
        lines += [
            "| Lookback (mo) | Lookback (yr) | OOS Years | Chain CAGR | Hit Rate | "
            "Avg Sharpe | Avg MaxDD | Avg Trades | Avg Win% | Avg IS Sharpe |"
        ]
        lines += ["|---|---|---|---|---|---|---|---|---|---|"]
        sub = summary_df[summary_df["symbol"] == sym].copy()
        best_lb = int(sub.loc[sub["avg_sharpe"].idxmax(), "lookback_months"])
        for _, row in sub.iterrows():
            tag = " ★" if int(row["lookback_months"]) == best_lb else ""
            lines.append(
                f"| {int(row['lookback_months'])}{tag} | {row['lookback_months']/12:.0f} | "
                f"{int(row['n_oos_years'])} | {row['chain_cagr']:+.2%} | {row['hit_rate']:.0%} | "
                f"{row['avg_sharpe']:.3f} | {row['avg_max_dd']:.1%} | "
                f"{row['avg_trades']:.1f} | {row['avg_win_rate']:.1%} | "
                f"{row['avg_is_sharpe']:.3f} |"
            )
        lines += [""]

    lines += ["## Year-by-Year OOS Performance — Best Lookback per Symbol", ""]
    for sym in SYMBOLS:
        sub_sum = summary_df[summary_df["symbol"] == sym]
        best_lb = int(sub_sum.loc[sub_sum["avg_sharpe"].idxmax(), "lookback_months"])
        lines += [
            f"### {sym} — {best_lb}-month IS window (best avg OOS Sharpe)", "",
            f"| Year | Entry {mt} | Exit {mt} | Trail | IS Sharpe | OOS CAGR | "
            "OOS Sharpe | OOS MaxDD | Trades | Win% | Buy-Hold |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        sub = results_df[
            (results_df["symbol"] == sym) &
            (results_df["lookback_months"] == best_lb)
        ].sort_values("year")
        for _, row in sub.iterrows():
            lines.append(
                f"| {int(row['year'])} | "
                f"{int(row['entry_fast'])}/{int(row['entry_slow'])} | "
                f"{int(row['exit_fast'])}/{int(row['exit_slow'])} | "
                f"{row['trailing_stop']:.0%} | "
                f"{row['is_sharpe']:.3f} | "
                f"{row['oos_cagr']:+.1%} | "
                f"{row['oos_sharpe']:.2f} | "
                f"{row['oos_max_dd']:.1%} | "
                f"{int(row['oos_trades'])} | "
                f"{row['oos_win_rate']:.1%} | "
                f"{row['oos_bh_return']:+.1%} |"
            )
        lines += [""]

    lines += [
        "## Methodology Notes", "",
        "- **IS window:** Exact calendar months (DateOffset) ending Dec 31 of (OOS year − 1). "
        "Not year-aligned — a 60-month window ending 2014-12-31 starts ~2009-12-31 regardless "
        "of how many trading years that spans.",
        f"- **OOS window:** One full calendar year. {mt}s are initialised on the IS window then "
        "the OOS year is sliced out, preventing look-ahead contamination.",
        f"- **Grid:** 122,500 combinations ({mt} entry pairs × {mt} exit pairs × 4 trailing stops). "
        "IS objective: Sharpe ratio maximization. TC: 5 bps round-trip.",
        "- **Chain CAGR:** Geometric chain-link of raw OOS holding-period returns across all "
        "OOS years, annualised by number of OOS years. Not arithmetic average of annual CAGRs.",
        "- **Ranking:** Primary = avg OOS Sharpe (robust to single-year outliers). "
        "Secondary = chain CAGR.",
        "- **Data:** SPY + QQQ daily OHLCV from 2000-01-03. Lookbacks > data history are "
        f"automatically skipped if IS bars < {MIN_IS_BARS}.",
        "",
    ]

    md_path.write_text("\n".join(lines))
    return md_path


# ── chart ─────────────────────────────────────────────────────────────────────

def plot_sweep(summary_df: pd.DataFrame, ma_type: str, today_str: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mt       = ma_type.upper()
    out_path = RESULT_DIR / f"{ma_type}_lookback_sweep_{today_str}.png"

    colors = {"SPY": "#1f77b4", "QQQ": "#ff7f0e"}
    x_lbs  = LOOKBACK_MONTHS

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{mt} IS Lookback Sweep — SPY & QQQ (2010–2026 OOS, Sharpe-optimized IS, {TC_BPS} bps TC)\n"
        "IS window: 60–180 months (step 12)",
        fontsize=12,
    )

    panels = [
        ("avg_sharpe", "Avg OOS Sharpe",           False),
        ("chain_cagr", "Chain-Linked CAGR",         True),
        ("hit_rate",   "Hit Rate (positive years)", True),
        ("avg_max_dd", "Avg Max Drawdown",           True),
    ]

    for ax, (col, title, pct_fmt) in zip(axes.flat, panels):
        for sym in SYMBOLS:
            sub  = summary_df[summary_df["symbol"] == sym].sort_values("lookback_months")
            vals = sub[col].tolist()
            lbs  = sub["lookback_months"].tolist()
            ax.plot(lbs, vals, marker="o", label=sym, color=colors[sym], linewidth=2, markersize=6)

            best_idx = int(np.argmin(vals)) if col == "avg_max_dd" else int(np.argmax(vals))
            ax.plot(
                lbs[best_idx], vals[best_idx],
                marker="*", markersize=14, color=colors[sym], zorder=5,
            )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("IS Lookback (months)")
        ax.set_xticks(x_lbs)
        ax.set_xticklabels([str(lb) for lb in x_lbs], rotation=45, ha="right")
        if pct_fmt:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep IS lookback (60–180 months, step 12) in walk-forward MA optimization. "
            f"Runs {len(LOOKBACK_MONTHS)} × {len(WF_YEARS)} OOS years × {len(SYMBOLS)} symbols "
            f"= {len(LOOKBACK_MONTHS)*len(WF_YEARS)*len(SYMBOLS)} IS grid searches."
        )
    )
    parser.add_argument(
        "--ma-type", choices=("ema", "sma"), default="ema",
        help="Moving average type: ema (default) or sma"
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers for grid search (default: all CPUs)"
    )
    args = parser.parse_args()

    today_str  = date.today().isoformat()
    df         = load_data()
    results_df = run_lookback_sweep(df, ma_type=args.ma_type, n_jobs=args.jobs)

    if results_df.empty:
        logger.error("No results — check data coverage (need 2000-present for SPY+QQQ).")
        sys.exit(1)

    summary_df = compute_summary(results_df)

    print_console_results(results_df, summary_df, ma_type=args.ma_type)
    plot_sweep(summary_df, ma_type=args.ma_type, today_str=today_str)

    md_path  = write_markdown(results_df, summary_df, ma_type=args.ma_type, today_str=today_str)
    csv_path = RESULT_DIR / f"{args.ma_type}_lookback_sweep_{today_str}.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info(f"Results CSV → {csv_path}")
    logger.info(f"Markdown   → {md_path}")


if __name__ == "__main__":
    main()
