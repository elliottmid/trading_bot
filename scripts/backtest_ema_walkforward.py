#!/usr/bin/env python3
"""
Walk-forward backtest of the EMA crossover rule from ema_daily_scan.py.

Parameters (grid-search winner):
    Entry: EMA(6) crosses above EMA(10)  → BUY
    Exit:  EMA(11) crosses below EMA(24) → SELL

Fold structure:
    Initial train : TRAIN_MONTHS of EMA warmup (no trades counted)
    Test windows  : TEST_MONTHS each, rolling forward
    Warmup carry  : EMAs are computed on all data up to end of test window,
                    so state at the start of each test window is fully warmed up.

Usage
-----
    python3 backtest_ema_walkforward.py
    python3 backtest_ema_walkforward.py --tc-bps 10
    python3 backtest_ema_walkforward.py --train-months 18 --test-months 3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── fixed strategy params ────────────────────────────────────────────────────
ENTRY_FAST = 6
ENTRY_SLOW = 10
EXIT_FAST  = 11
EXIT_SLOW  = 24

# ── walk-forward defaults ────────────────────────────────────────────────────
TRAIN_MONTHS = 24
TEST_MONTHS  = 6
TC_BPS       = 5

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "models"


# ── data ────────────────────────────────────────────────────────────────────

def load_data(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        logger.error(f"Data file not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        f"Loaded {len(df):,} bars · {df['symbol'].nunique()} symbols · "
        f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"
    )
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required EMA columns per symbol (uses full history — no look-ahead)."""
    periods = [ENTRY_FAST, ENTRY_SLOW, EXIT_FAST, EXIT_SLOW]
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("timestamp")
        for p in periods:
            g[f"ema_{p}"] = g["close"].ewm(span=p, adjust=False).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ── signal generation ────────────────────────────────────────────────────────

def crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    above = (fast > slow).astype(int)
    sig = pd.Series(0, index=fast.index, dtype=int)
    sig[(above.shift(1) == 0) & (above == 1)] =  1   # bullish cross
    sig[(above.shift(1) == 1) & (above == 0)] = -1   # bearish cross
    return sig


# ── per-symbol, per-fold backtest ────────────────────────────────────────────

def backtest_symbol_fold(
    sym_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    tc_bps: float,
) -> dict:
    """
    Run one symbol over one test window.
    EMAs use full history through test_end (warmup is automatic).
    Trades are only opened if the entry signal fires *during* the test window.
    Positions opened in the test window are closed at test_end if still open.
    """
    symbol = sym_df["symbol"].iloc[0]

    sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)
    entry_sig = crossover_signal(sym_df[f"ema_{ENTRY_FAST}"], sym_df[f"ema_{ENTRY_SLOW}"])
    exit_sig  = crossover_signal(sym_df[f"ema_{EXIT_FAST}"],  sym_df[f"ema_{EXIT_SLOW}"])

    close_arr  = sym_df["close"].to_numpy()
    ts_arr     = sym_df["timestamp"].to_numpy()
    entry_arr  = entry_sig.to_numpy()
    exit_arr   = exit_sig.to_numpy()
    n          = len(close_arr)
    tc_frac    = tc_bps / 10_000

    test_start_np = np.datetime64(test_start)
    test_end_np   = np.datetime64(test_end)
    in_test        = (ts_arr >= test_start_np) & (ts_arr <= test_end_np)

    in_position = False
    entry_price = 0.0
    entry_idx   = 0
    prev_eq     = 0.0
    pnls:            list[float] = []
    hold_days_list:  list[int]   = []
    daily_ret        = np.zeros(n)

    for i in range(1, n):
        sig_i = i - 1

        # Open new position only if entry fires during test window
        if not in_position and entry_arr[sig_i] == 1 and in_test[i]:
            in_position = True
            entry_price = close_arr[i]
            entry_idx   = i
            prev_eq     = 0.0
            daily_ret[i] -= tc_frac

        if in_position and in_test[i]:
            eq = (close_arr[i] - entry_price) / entry_price
            daily_ret[i] += eq - prev_eq
            prev_eq = eq

            # Close on exit signal
            if exit_arr[sig_i] == -1:
                daily_ret[i] -= tc_frac
                pnls.append(eq - 2 * tc_frac)
                hold_days_list.append(i - entry_idx)
                in_position = False
                prev_eq = 0.0

            # Force-close at end of test window
            elif ts_arr[i] == ts_arr[in_test].max():
                daily_ret[i] -= tc_frac
                pnls.append(eq - 2 * tc_frac)
                hold_days_list.append(i - entry_idx)
                in_position = False
                prev_eq = 0.0

    test_ret  = daily_ret[in_test]
    n_trades  = len(pnls)
    pnls_arr  = np.array(pnls) if pnls else np.array([0.0])
    strat_ret = float(np.prod(1 + pnls_arr) - 1) if pnls else 0.0

    # CAGR
    test_dates = sym_df["timestamp"][in_test]
    years = (test_dates.iloc[-1] - test_dates.iloc[0]).days / 365.25 if len(test_dates) > 1 else 0
    base = 1 + strat_ret
    cagr = float(base ** (1 / years) - 1) if years > 0 and base > 0 else 0.0

    # Sharpe
    vol    = test_ret.std()
    sharpe = float(test_ret.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    # Max drawdown (on test-window daily return series)
    eq_curve    = np.cumprod(1 + test_ret)
    running_max = np.maximum.accumulate(eq_curve)
    max_dd      = float((eq_curve / running_max - 1).min())

    # Calmar (CAGR / |max_dd|) — 0 if flat
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # Win rate, avg win/loss
    win_mask = pnls_arr > 0
    win_rate = float(win_mask.mean()) if n_trades > 0 else 0.0
    avg_win  = float(pnls_arr[win_mask].mean())   if win_mask.any()  else 0.0
    avg_loss = float(pnls_arr[~win_mask].mean())  if (~win_mask).any() else 0.0
    avg_hold = float(np.mean(hold_days_list))       if hold_days_list else 0.0

    # Buy-hold
    test_close = sym_df["close"][in_test]
    bh_ret = float(
        (test_close.iloc[-1] - test_close.iloc[0]) / test_close.iloc[0]
    ) if len(test_close) > 1 else 0.0

    # Time in market
    bars_in    = sum(in_test)
    time_in_mkt = float(np.sum(test_ret != 0) / bars_in) if bars_in > 0 else 0.0

    return {
        "symbol":        symbol,
        "trades":        n_trades,
        "strat_ret":     strat_ret,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "calmar":        calmar,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "avg_hold_days": avg_hold,
        "time_in_mkt":   time_in_mkt,
        "bh_ret":        bh_ret,
        "excess_ret":    strat_ret - bh_ret,
    }


# ── fold definitions ─────────────────────────────────────────────────────────

def build_folds(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    train_months: int,
    test_months: int,
) -> list[dict]:
    """Return list of {fold, train_start, train_end, test_start, test_end}."""
    folds = []
    test_start = min_date + relativedelta(months=train_months)
    fold_num   = 1
    while test_start < max_date:
        test_end = min(test_start + relativedelta(months=test_months), max_date)
        if (test_end - test_start).days < 30:
            break
        folds.append({
            "fold":        fold_num,
            "train_start": min_date,
            "train_end":   test_start,
            "test_start":  test_start,
            "test_end":    test_end,
        })
        test_start = test_end
        fold_num  += 1
    return folds


# ── walk-forward orchestration ───────────────────────────────────────────────

def run_walkforward(
    df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    test_months:  int = TEST_MONTHS,
    tc_bps:       float = TC_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_symbol_fold_results, per_fold_summary).
    """
    folds = build_folds(
        df["timestamp"].min(),
        df["timestamp"].max(),
        train_months,
        test_months,
    )
    logger.info(f"Walk-forward: {len(folds)} folds · train={train_months}mo · test={test_months}mo · tc={tc_bps}bps")

    sym_groups = {sym: g for sym, g in df.groupby("symbol", sort=False)}

    all_rows  = []
    fold_rows = []

    for fold_info in folds:
        fold     = fold_info["fold"]
        t_start  = pd.Timestamp(fold_info["test_start"])
        t_end    = pd.Timestamp(fold_info["test_end"])

        logger.info(f"  Fold {fold}: test {t_start.date()} → {t_end.date()}")

        sym_rows = []
        for sym, sym_df in sym_groups.items():
            # Use only data through test_end (no future look-ahead)
            sym_slice = sym_df[sym_df["timestamp"] <= t_end].copy()
            if len(sym_slice) < EXIT_SLOW * 4:
                continue
            r = backtest_symbol_fold(sym_slice, t_start, t_end, tc_bps)
            r["fold"]       = fold
            r["test_start"] = t_start.date()
            r["test_end"]   = t_end.date()
            all_rows.append(r)
            sym_rows.append(r)

        if not sym_rows:
            continue

        mdf = pd.DataFrame(sym_rows)

        # Equal-weight buy-hold benchmark for this fold
        bh_ew = float(mdf["bh_ret"].mean())

        fold_rows.append({
            "fold":           fold,
            "test_start":     t_start.date(),
            "test_end":       t_end.date(),
            "n_symbols":      len(mdf),
            "mean_cagr":      mdf["cagr"].mean(),
            "median_cagr":    mdf["cagr"].median(),
            "mean_sharpe":    mdf["sharpe"].mean(),
            "median_sharpe":  mdf["sharpe"].median(),
            "mean_max_dd":    mdf["max_dd"].mean(),
            "mean_calmar":    mdf["calmar"].mean(),
            "mean_win_rate":  mdf["win_rate"].mean(),
            "mean_trades":    mdf["trades"].mean(),
            "mean_hold_days": mdf["avg_hold_days"].mean(),
            "mean_time_in":   mdf["time_in_mkt"].mean(),
            "pct_profitable": (mdf["strat_ret"] > 0).mean(),
            "bh_ew_ret":      bh_ew,
            "mean_excess":    mdf["excess_ret"].mean(),
            "mean_strat_ret": mdf["strat_ret"].mean(),
        })

    sym_df_out  = pd.DataFrame(all_rows)
    fold_df_out = pd.DataFrame(fold_rows)
    return sym_df_out, fold_df_out


# ── printing ─────────────────────────────────────────────────────────────────

def print_fold_summary(fold_df: pd.DataFrame) -> None:
    cols = [
        "fold", "test_start", "test_end",
        "mean_cagr", "mean_sharpe", "mean_max_dd", "mean_calmar",
        "mean_win_rate", "mean_trades", "pct_profitable",
        "bh_ew_ret", "mean_excess",
    ]
    fmt = {
        "mean_cagr":      "{:.1%}".format,
        "mean_sharpe":    "{:.2f}".format,
        "mean_max_dd":    "{:.1%}".format,
        "mean_calmar":    "{:.2f}".format,
        "mean_win_rate":  "{:.1%}".format,
        "mean_trades":    "{:.1f}".format,
        "pct_profitable": "{:.1%}".format,
        "bh_ew_ret":      "{:.1%}".format,
        "mean_excess":    "{:.1%}".format,
    }
    tbl = fold_df[cols].copy()
    for col, fn in fmt.items():
        tbl[col] = tbl[col].map(fn)

    print("\n" + "=" * 120)
    print("WALK-FORWARD FOLD RESULTS  "
          f"(EMA {ENTRY_FAST}/{ENTRY_SLOW} entry  |  EMA {EXIT_FAST}/{EXIT_SLOW} exit)")
    print("=" * 120)
    print(tbl.to_string(index=False))


def print_aggregate_summary(fold_df: pd.DataFrame) -> None:
    # (metric_name, format_fn)
    metrics: list[tuple[str, str]] = [
        ("mean_cagr",      "pct"),
        ("mean_sharpe",    "float"),
        ("mean_max_dd",    "pct"),
        ("mean_calmar",    "float"),
        ("mean_win_rate",  "pct"),
        ("pct_profitable", "pct"),
        ("bh_ew_ret",      "pct"),
        ("mean_excess",    "pct"),
    ]
    print("\n" + "-" * 80)
    print("AGGREGATE  (mean ± std across folds)")
    print("-" * 80)
    for m, fmt in metrics:
        vals = fold_df[m]
        if fmt == "pct":
            print(f"  {m:<22}  {vals.mean():>+.1%}  ±  {vals.std():.1%}   "
                  f"[min {vals.min():>+.1%}  max {vals.max():>+.1%}]")
        else:
            print(f"  {m:<22}  {vals.mean():>+.3f}  ±  {vals.std():.3f}   "
                  f"[min {vals.min():>+.3f}  max {vals.max():>+.3f}]")

    n_pos = int((fold_df["mean_excess"] > 0).sum())
    print(f"\n  Folds with positive excess return: {n_pos} / {len(fold_df)}")
    print(f"  Mean IC between fold excess and fold sharpe: "
          f"{fold_df['mean_excess'].corr(fold_df['mean_sharpe']):.3f}")
    print("-" * 70)


def print_symbol_summary(sym_df: pd.DataFrame) -> None:
    agg = (
        sym_df.groupby("symbol")
        .agg(
            folds       =("fold", "count"),
            mean_cagr   =("cagr", "mean"),
            mean_sharpe =("sharpe", "mean"),
            mean_max_dd =("max_dd", "mean"),
            total_trades=("trades", "sum"),
            mean_win_rt =("win_rate", "mean"),
            mean_excess =("excess_ret", "mean"),
            mean_bh     =("bh_ret", "mean"),
        )
        .sort_values("mean_sharpe", ascending=False)
    )
    fmt = {
        "mean_cagr":    "{:.1%}".format,
        "mean_sharpe":  "{:.2f}".format,
        "mean_max_dd":  "{:.1%}".format,
        "mean_win_rt":  "{:.1%}".format,
        "mean_excess":  "{:.1%}".format,
        "mean_bh":      "{:.1%}".format,
    }
    tbl = agg.copy()
    for col, fn in fmt.items():
        tbl[col] = tbl[col].map(fn)

    print("\n" + "-" * 100)
    print("PER-SYMBOL SUMMARY  (averaged across folds, ranked by mean Sharpe)")
    print("-" * 100)
    print(tbl.to_string())
    print("-" * 100)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global ENTRY_FAST, ENTRY_SLOW, EXIT_FAST, EXIT_SLOW
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest of EMA crossover rule"
    )
    parser.add_argument("--data",          default="ema_etfs_primary_daily.parquet")
    parser.add_argument("--train-months",  type=int,   default=TRAIN_MONTHS)
    parser.add_argument("--test-months",   type=int,   default=TEST_MONTHS)
    parser.add_argument("--tc-bps",        type=float, default=TC_BPS)
    parser.add_argument("--entry-fast",    type=int,   default=ENTRY_FAST)
    parser.add_argument("--entry-slow",    type=int,   default=ENTRY_SLOW)
    parser.add_argument("--exit-fast",     type=int,   default=EXIT_FAST)
    parser.add_argument("--exit-slow",     type=int,   default=EXIT_SLOW)
    parser.add_argument("--out-prefix",    default="ema_wf",
                        help="Output filename prefix inside data/models/")
    parser.add_argument("--exclude", nargs="*", default=[],
                        metavar="SYM", help="Symbols to drop before backtesting")
    args = parser.parse_args()

    ENTRY_FAST = args.entry_fast
    ENTRY_SLOW = args.entry_slow
    EXIT_FAST  = args.exit_fast
    EXIT_SLOW  = args.exit_slow

    raw = load_data(args.data)
    if args.exclude:
        raw = raw[~raw["symbol"].isin(args.exclude)]
        logger.info(f"Excluded: {args.exclude}  ({raw['symbol'].nunique()} symbols remain)")
    df  = add_emas(raw)

    sym_df, fold_df = run_walkforward(
        df,
        train_months=args.train_months,
        test_months=args.test_months,
        tc_bps=args.tc_bps,
    )

    print_fold_summary(fold_df)
    print_aggregate_summary(fold_df)
    print_symbol_summary(sym_df)

    # ── save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    folds_path  = OUT_DIR / f"{args.out_prefix}_folds.csv"
    detail_path = OUT_DIR / f"{args.out_prefix}_symbol_folds.csv"
    fold_df.to_csv(folds_path,  index=False)
    sym_df.to_csv(detail_path,  index=False)

    print(f"\nOutputs written:")
    print(f"  {folds_path}")
    print(f"  {detail_path}")


if __name__ == "__main__":
    main()
