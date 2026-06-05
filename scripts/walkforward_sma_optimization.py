#!/usr/bin/env python3
# Description: Annual walk-forward SMA optimization — each year finds Sharpe-maximizing
#              entry/exit SMA + trailing stop on prior 14-year IS window (168-month lookback-sweep optimum), then applies
#              those params OOS to the following calendar year (2010–2026).
#
# Direct SMA analogue of walkforward_ema_optimization.py. The only substantive change
# is that moving averages use simple rolling means instead of exponential weighting.
# Grid ranges, backtest logic, walk-forward structure, and output format are identical
# so EMA vs SMA results can be compared directly.
#
# Grid: entry SMA (fast 3-12, slow 8-26), exit SMA (same), trailing stop 3-6%.
# Objective: maximize IS Sharpe ratio. OOS = next calendar year (partial for 2026).
# Output: CSV + two-panel chart (annual OOS CAGR bars + equity curve).

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

SYMBOLS     = ["SPY", "QQQ"]
WF_YEARS    = list(range(2010, 2027))   # OOS years
IS_WINDOW   = 14                         # calendar years of in-sample history (168-month lookback-sweep optimum)
TC_BPS      = 5
TRAIL_STOPS = [0.03, 0.04, 0.05, 0.06]  # 3–6 %


# ── data ──────────────────────────────────────────────────────────────────────

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


# ── parameter grid ─────────────────────────────────────────────────────────────

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


# ── SMA precomputation (replaces EMA ewm) ─────────────────────────────────────

def add_smas(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    df = df.copy()
    for p in sorted(set(periods)):
        df[f"sma_{p}"] = df["close"].rolling(window=p, min_periods=p).mean()
    return df


# ── fast IS backtest — returns Sharpe only ────────────────────────────────────

def _sharpe_only(
    close_arr: np.ndarray,
    sf_arr: np.ndarray,   # SMA fast entry
    ss_arr: np.ndarray,   # SMA slow entry
    xf_arr: np.ndarray,   # SMA fast exit
    xs_arr: np.ndarray,   # SMA slow exit
    trail_pct: float,
    tc_frac: float,
) -> float:
    in_pos      = False
    entry_price = 0.0
    peak        = 0.0
    prev_eq     = 0.0
    n           = len(close_arr)

    daily_sum    = 0.0
    daily_sum_sq = 0.0
    n_days       = 0

    for i in range(2, n):
        sig = i - 1

        # Skip NaN warm-up bars
        if (np.isnan(sf_arr[sig]) or np.isnan(ss_arr[sig]) or
                np.isnan(xf_arr[sig]) or np.isnan(xs_arr[sig])):
            continue
        if (np.isnan(sf_arr[sig - 1]) or np.isnan(ss_arr[sig - 1]) or
                np.isnan(xf_arr[sig - 1]) or np.isnan(xs_arr[sig - 1])):
            continue

        entry_cross = sf_arr[sig] > ss_arr[sig] and sf_arr[sig - 1] <= ss_arr[sig - 1]
        exit_cross  = xf_arr[sig] < xs_arr[sig] and xf_arr[sig - 1] >= xs_arr[sig - 1]

        daily_ret = 0.0

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close_arr[i]
            peak        = close_arr[i]
            prev_eq     = 0.0
            daily_ret  -= tc_frac

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            eq           = (close_arr[i] - entry_price) / entry_price
            daily_ret    += eq - prev_eq
            prev_eq      = eq
            trail_hit  = close_arr[i] <= peak * (1.0 - trail_pct)
            if trail_hit or exit_cross:
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
    if var <= 0:
        return 0.0
    return float(mean / var ** 0.5 * 252 ** 0.5)


def _run_is_batch(
    close_arr:   np.ndarray,
    sma_dict:    dict[int, np.ndarray],
    combo_batch: list[tuple[int, int, int, int, float]],
    tc_frac:     float,
) -> list[tuple[float, tuple[int, int, int, int, float]]]:
    out = []
    for ef, es, xf, xs, ts in combo_batch:
        sharpe = _sharpe_only(
            close_arr,
            sma_dict[ef], sma_dict[es], sma_dict[xf], sma_dict[xs],
            ts, tc_frac,
        )
        out.append((sharpe, (ef, es, xf, xs, ts)))
    return out


# ── full OOS backtest — all metrics ───────────────────────────────────────────

def backtest_full(
    sym_df:    pd.DataFrame,
    ef: int, es: int,
    xf: int, xs: int,
    trail_pct: float,
    tc_frac:   float,
) -> dict:
    close_arr = sym_df["close"].to_numpy()
    ef_arr    = sym_df[f"sma_{ef}"].to_numpy()
    es_arr    = sym_df[f"sma_{es}"].to_numpy()
    xf_arr    = sym_df[f"sma_{xf}"].to_numpy()
    xs_arr    = sym_df[f"sma_{xs}"].to_numpy()
    n         = len(close_arr)

    in_pos      = False
    entry_price = 0.0
    peak        = 0.0
    prev_eq     = 0.0
    pnls:        list[float] = []
    daily_rets   = np.zeros(n)
    trail_exits  = 0
    sma_exits    = 0

    for i in range(2, n):
        sig = i - 1

        if (np.isnan(ef_arr[sig]) or np.isnan(es_arr[sig]) or
                np.isnan(xf_arr[sig]) or np.isnan(xs_arr[sig])):
            continue
        if (np.isnan(ef_arr[sig - 1]) or np.isnan(es_arr[sig - 1]) or
                np.isnan(xf_arr[sig - 1]) or np.isnan(xs_arr[sig - 1])):
            continue

        entry_cross = ef_arr[sig] > es_arr[sig] and ef_arr[sig - 1] <= es_arr[sig - 1]
        exit_cross  = xf_arr[sig] < xs_arr[sig] and xf_arr[sig - 1] >= xs_arr[sig - 1]

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close_arr[i]
            peak        = close_arr[i]
            prev_eq     = 0.0
            daily_rets[i] -= tc_frac

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            eq = (close_arr[i] - entry_price) / entry_price
            daily_rets[i] += eq - prev_eq
            prev_eq = eq

            trail_hit = close_arr[i] <= peak * (1.0 - trail_pct)
            if trail_hit or exit_cross:
                daily_rets[i] -= tc_frac
                pnls.append(eq - tc_frac)
                trail_exits += int(trail_hit)
                sma_exits   += int(not trail_hit)
                in_pos  = False
                prev_eq = 0.0

    n_trades  = len(pnls)
    pnls_arr  = np.array(pnls) if pnls else np.zeros(1)
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

    win_mask = pnls_arr > 0
    win_rate = float(win_mask.mean()) if n_trades > 0 else 0.0
    bh_ret   = float((close_arr[-1] - close_arr[0]) / close_arr[0]) if len(close_arr) > 1 else 0.0

    return {
        "trades":        n_trades,
        "oos_strat_ret": strat_ret,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "win_rate":      win_rate,
        "trail_exits":   trail_exits,
        "sma_exits":     sma_exits,
        "bh_return":     bh_ret,
    }


# ── walk-forward loop ──────────────────────────────────────────────────────────

def run_walkforward(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    combos  = build_combos()
    tc_frac = TC_BPS / 10_000

    all_periods = sorted({p for c in combos for p in c[:4]})

    n_workers = cpu_count() if n_jobs < 0 else max(1, n_jobs)
    chunk     = math.ceil(len(combos) / n_workers)
    batches   = [combos[i : i + chunk] for i in range(0, len(combos), chunk)]
    logger.info(f"Parallel jobs={n_workers}  batch size={chunk}")

    rows = []

    for oos_year in WF_YEARS:
        is_start = oos_year - IS_WINDOW
        is_end   = oos_year - 1

        for symbol in SYMBOLS:
            sym_all = (
                df[df["symbol"] == symbol]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # IS slice: precompute SMAs
            is_mask = (
                (sym_all["timestamp"].dt.year >= is_start) &
                (sym_all["timestamp"].dt.year <= is_end)
            )
            is_df = add_smas(sym_all[is_mask].copy(), all_periods)

            if len(is_df) < 200:
                logger.warning(f"{symbol} {oos_year}: IS only {len(is_df)} bars — skip")
                continue

            is_close    = is_df["close"].to_numpy()
            is_sma_dict = {p: is_df[f"sma_{p}"].to_numpy() for p in all_periods}

            # Grid search: maximise IS Sharpe in parallel
            batch_res = Parallel(n_jobs=n_jobs)(
                delayed(_run_is_batch)(is_close, is_sma_dict, batch, tc_frac)
                for batch in batches
            )
            flat             = [item for batch in batch_res for item in batch]
            best_is_sharpe, best_params = max(flat, key=lambda x: x[0])
            ef, es, xf, xs, ts = best_params

            logger.info(
                f"  {symbol} {oos_year}: IS Sharpe={best_is_sharpe:.3f}  "
                f"entry SMA({ef}/{es})  exit SMA({xf}/{xs})  trail={ts:.0%}"
            )

            # OOS: compute SMAs on combined [is_start → oos_year] window to
            # avoid warm-up NaN bleed into the first OOS bars.
            combined_mask = (
                (sym_all["timestamp"].dt.year >= is_start) &
                (sym_all["timestamp"].dt.year <= oos_year)
            )
            combined_df = add_smas(sym_all[combined_mask].copy(), [ef, es, xf, xs])
            oos_df      = combined_df[combined_df["timestamp"].dt.year == oos_year].copy()

            if len(oos_df) < 20:
                logger.warning(f"{symbol} {oos_year}: OOS only {len(oos_df)} bars — skip")
                continue

            oos_metrics = backtest_full(oos_df, ef, es, xf, xs, ts, tc_frac)

            rows.append({
                "year":          oos_year,
                "symbol":        symbol,
                "entry_fast":    ef,
                "entry_slow":    es,
                "exit_fast":     xf,
                "exit_slow":     xs,
                "trailing_stop": ts,
                "is_sharpe":     best_is_sharpe,
                "oos_strat_ret": oos_metrics["oos_strat_ret"],
                **{f"oos_{k}": v for k, v in oos_metrics.items()},
            })

    return pd.DataFrame(rows)


# ── summary printing ───────────────────────────────────────────────────────────

def compile_summary(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 130)
    print("WALK-FORWARD SMA OPTIMIZATION — OOS RESULTS  "
          "(Sharpe-optimized IS params applied to next calendar year, 5 bps TC)")
    print("=" * 130)

    for symbol in SYMBOLS:
        sub = results_df[results_df["symbol"] == symbol].copy()
        if sub.empty:
            continue

        print(f"\n{'─' * 55}  {symbol}  {'─' * 55}")
        cols = [
            "year", "entry_fast", "entry_slow", "exit_fast", "exit_slow",
            "trailing_stop", "is_sharpe",
            "oos_cagr", "oos_sharpe", "oos_max_dd",
            "oos_trades", "oos_win_rate", "oos_bh_return",
        ]
        fmt = {
            "trailing_stop": "{:.0%}".format,
            "is_sharpe":     "{:.3f}".format,
            "oos_cagr":      "{:+.1%}".format,
            "oos_sharpe":    "{:.2f}".format,
            "oos_max_dd":    "{:.1%}".format,
            "oos_win_rate":  "{:.1%}".format,
            "oos_bh_return": "{:+.1%}".format,
        }
        tbl = sub[cols].copy()
        for col, fn in fmt.items():
            tbl[col] = tbl[col].map(fn)
        print(tbl.to_string(index=False))

        hprs       = sub["oos_strat_ret"].to_numpy()
        chain_ret  = float(np.prod(1.0 + hprs) - 1.0)
        n_years    = len(sub)
        chain_cagr = float((1.0 + chain_ret) ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0
        hit_rate   = float((sub["oos_strat_ret"] > 0).mean())

        print(f"\n  Chain-linked CAGR (HPR basis): {chain_cagr:+.1%}  |  "
              f"Hit rate (positive OOS years): {hit_rate:.0%}  |  "
              f"Avg OOS Sharpe: {sub['oos_sharpe'].mean():.2f}  |  "
              f"Avg OOS MaxDD: {sub['oos_max_dd'].mean():.1%}")


# ── chart ──────────────────────────────────────────────────────────────────────

def plot_results(results_df: pd.DataFrame, today_str: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"walkforward_sma_equity_curve_{today_str}.png"

    colors = {"SPY": "#1f77b4", "QQQ": "#ff7f0e"}
    years  = sorted(results_df["year"].unique())
    x      = np.arange(len(years))
    width  = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "Walk-Forward SMA Optimization — Annual IS/OOS Results (2010–2026)\n"
        f"IS window: 14 years | Grid: SMA entry/exit × trailing stop 3–6% | TC: {TC_BPS} bps | IS objective: Sharpe",
        fontsize=12,
    )

    ax1 = axes[0]
    for i, sym in enumerate(SYMBOLS):
        sub  = results_df[results_df["symbol"] == sym].set_index("year")
        vals = [sub.loc[y, "oos_cagr"] if y in sub.index else 0.0 for y in years]
        bars = ax1.bar(
            x + (i - 0.5) * width, vals, width,
            label=sym, color=colors[sym], alpha=0.85,
        )
        for bar, y_val, yr in zip(bars, vals, years):
            if sym in results_df[results_df["year"] == yr]["symbol"].values:
                row = results_df[(results_df["symbol"] == sym) & (results_df["year"] == yr)].iloc[0]
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 if y_val >= 0 else bar.get_height() - 0.025,
                    f"{row['trailing_stop']:.0%}",
                    ha="center", va="bottom", fontsize=6, color="grey",
                )

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(years, rotation=45, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title("Annual OOS CAGR  (Sharpe-optimized IS parameters)", fontsize=11)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    for sym in SYMBOLS:
        sub  = results_df[results_df["symbol"] == sym].sort_values("year")
        hprs = sub["oos_strat_ret"].to_numpy()
        eq   = np.cumprod(1.0 + hprs)
        eq   = np.insert(eq, 0, 1.0)
        ax2.plot(range(len(eq)), eq, label=sym, color=colors[sym], linewidth=2, marker="o", markersize=4)

    ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    tick_labels = [str(years[0] - 1)] + [str(y) for y in years]
    ax2.set_xticks(range(len(years) + 1))
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax2.set_title("Chain-Linked OOS Equity Curve  ($1 invested Jan 2010)", fontsize=11)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart → {out_path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annual walk-forward SMA optimization (Sharpe-maximizing IS → OOS)"
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers for grid search (default: all CPUs)"
    )
    args = parser.parse_args()

    today_str = date.today().isoformat()

    df         = load_data()
    results_df = run_walkforward(df, n_jobs=args.jobs)

    if results_df.empty:
        logger.error("No results — check data coverage (need 2000-present for SPY+QQQ).")
        sys.exit(1)

    compile_summary(results_df)
    plot_results(results_df, today_str)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / f"walkforward_sma_results_{today_str}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results → {csv_path}")


if __name__ == "__main__":
    main()
