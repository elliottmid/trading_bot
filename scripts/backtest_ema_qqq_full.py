#!/usr/bin/env python3
"""
Full backtest: QQQ EMA(12/24) entry → EMA(10/23) exit + 5% trailing stop
2000-01-03 → present, 5bps TC each side.

Reports: equity curve summary, year-by-year, trade log, vs buy-hold QQQ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR  = Path(__file__).parent.parent / "data" / "models"

# ── params ────────────────────────────────────────────────────────────────────
ENTRY_FAST  = 12
ENTRY_SLOW  = 24
EXIT_FAST   = 10
EXIT_SLOW   = 23
TRAIL_PCT   = 0.05
TC_BPS      = 5
TC_FRAC     = TC_BPS / 10_000


def load_qqq() -> pd.DataFrame:
    path = DATA_DIR / "spy_qqq_2000_daily.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    qqq = df[df["symbol"] == "QQQ"].sort_values("timestamp").reset_index(drop=True)
    qqq["date"] = qqq["timestamp"].dt.normalize()
    return qqq


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    for p in [ENTRY_FAST, ENTRY_SLOW, EXIT_FAST, EXIT_SLOW]:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    return df


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    close      = df["close"].to_numpy()
    dates      = df["date"].to_numpy()
    ef = df[f"ema_{ENTRY_FAST}"].to_numpy()
    es = df[f"ema_{ENTRY_SLOW}"].to_numpy()
    xf = df[f"ema_{EXIT_FAST}"].to_numpy()
    xs = df[f"ema_{EXIT_SLOW}"].to_numpy()
    n  = len(close)

    equity      = np.ones(n)   # starts at 1.0
    in_pos      = False
    entry_price = 0.0
    entry_date  = None
    peak        = 0.0
    prev_eq     = 0.0
    running_eq  = 1.0
    trades      = []

    for i in range(1, n):
        s = i - 1  # signal bar

        entry_cross = (s > 0 and
                       ef[s] > es[s] and ef[s - 1] <= es[s - 1])
        exit_cross  = (s > 0 and
                       xf[s] < xs[s] and xf[s - 1] >= xs[s - 1])

        if not in_pos and entry_cross:
            in_pos      = True
            entry_price = close[i]
            entry_date  = dates[i]
            peak        = close[i]
            prev_eq     = 0.0
            running_eq *= (1 - TC_FRAC)

        if in_pos:
            peak = max(peak, close[i])
            eq   = (close[i] - entry_price) / entry_price
            running_eq *= (1 + eq - prev_eq)
            prev_eq = eq

            trail_hit = close[i] <= peak * (1 - TRAIL_PCT)
            if trail_hit or exit_cross:
                running_eq *= (1 - TC_FRAC)
                exit_type = "trail" if trail_hit else "ema"
                trades.append({
                    "entry_date":  str(entry_date)[:10],
                    "exit_date":   str(dates[i])[:10],
                    "hold_days":   int((pd.Timestamp(dates[i]) - pd.Timestamp(entry_date)).days),
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(close[i], 4),
                    "pnl_pct":     round(eq * 100, 2),
                    "exit_type":   exit_type,
                    "peak_price":  round(peak, 4),
                })
                in_pos  = False
                prev_eq = 0.0

        equity[i] = running_eq

    df = df.copy()
    df["equity"] = equity
    return df, trades


def metrics(equity: np.ndarray, dates: pd.DatetimeIndex, label: str) -> dict:
    rets = np.diff(np.log(np.maximum(equity, 1e-12)))
    total   = equity[-1] / equity[0] - 1
    years   = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    cagr    = (equity[-1] / equity[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol     = rets.std() * np.sqrt(252)
    sharpe  = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    peak_eq = np.maximum.accumulate(equity)
    dd      = (equity - peak_eq) / peak_eq
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "label":       label,
        "total_ret":   total,
        "cagr":        cagr,
        "ann_vol":     vol,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "calmar":      calmar,
    }


def year_by_year(df: pd.DataFrame, equity_col: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    rows = []
    for yr, g in df.groupby("year"):
        start = g[equity_col].iloc[0]
        end   = g[equity_col].iloc[-1]
        ret   = end / start - 1
        rows.append({"year": yr, "return": ret})
    return pd.DataFrame(rows)


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main() -> None:
    qqq = load_qqq()
    qqq = add_emas(qqq)
    qqq, trades = run_backtest(qqq)

    # buy-hold equity curve
    qqq["bh_equity"] = qqq["close"] / qqq["close"].iloc[0]
    dates = pd.to_datetime(qqq["date"])

    strat = metrics(qqq["equity"].to_numpy(),   dates, "EMA strategy")
    bhold = metrics(qqq["bh_equity"].to_numpy(), dates, "Buy-hold QQQ")

    # ── summary ──────────────────────────────────────────────────────────────
    print_section(
        f"QQQ  EMA({ENTRY_FAST}/{ENTRY_SLOW})→({EXIT_FAST}/{EXIT_SLOW}) "
        f"+ {TRAIL_PCT:.0%} trail  |  {TC_BPS}bps TC  |  "
        f"{qqq['date'].min().date()} → {qqq['date'].max().date()}"
    )
    print(f"\n{'Metric':<22} {'Strategy':>14} {'Buy-Hold QQQ':>14}")
    print("-" * 52)
    for key, label, fmt in [
        ("total_ret", "Total return",  "{:.1%}"),
        ("cagr",      "CAGR",          "{:.2%}"),
        ("ann_vol",   "Ann. vol",       "{:.2%}"),
        ("sharpe",    "Sharpe ratio",  "{:.3f}"),
        ("max_dd",    "Max drawdown",  "{:.2%}"),
        ("calmar",    "Calmar ratio",  "{:.2f}"),
    ]:
        print(f"{label:<22} {fmt.format(strat[key]):>14} {fmt.format(bhold[key]):>14}")

    # ── trade stats ───────────────────────────────────────────────────────────
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        wins    = (tdf["pnl_pct"] > 0).sum()
        losses  = (tdf["pnl_pct"] <= 0).sum()
        trails  = (tdf["exit_type"] == "trail").sum()
        emas    = (tdf["exit_type"] == "ema").sum()
        print_section("TRADE STATISTICS")
        print(f"  Total trades    : {len(tdf)}")
        print(f"  Win / Loss      : {wins} / {losses}  ({wins/len(tdf):.1%} win rate)")
        print(f"  Avg win         : {tdf.loc[tdf['pnl_pct']>0,'pnl_pct'].mean():.2f}%")
        print(f"  Avg loss        : {tdf.loc[tdf['pnl_pct']<=0,'pnl_pct'].mean():.2f}%")
        print(f"  Avg hold        : {tdf['hold_days'].mean():.0f} days")
        print(f"  Trailing exits  : {trails} ({trails/len(tdf):.1%})")
        print(f"  EMA exits       : {emas} ({emas/len(tdf):.1%})")
        print(f"  Longest hold    : {tdf['hold_days'].max()} days")
        print(f"  Shortest hold   : {tdf['hold_days'].min()} days")

    # ── year by year ─────────────────────────────────────────────────────────
    print_section("YEAR-BY-YEAR RETURNS")
    strat_yy = year_by_year(qqq, "equity")
    bh_yy    = year_by_year(qqq, "bh_equity")
    yy = strat_yy.merge(bh_yy, on="year", suffixes=("_strat", "_bh"))
    print(f"\n{'Year':>6} {'Strategy':>12} {'Buy-Hold':>12} {'Delta':>10}")
    print("-" * 44)
    for _, row in yy.iterrows():
        delta = row["return_strat"] - row["return_bh"]
        marker = " ▲" if delta > 0 else " ▼"
        print(
            f"{int(row['year']):>6} "
            f"{row['return_strat']:>11.2%} "
            f"{row['return_bh']:>11.2%} "
            f"{delta:>+9.2%}{marker}"
        )

    # ── trade log (abbreviated) ───────────────────────────────────────────────
    if not tdf.empty:
        print_section("TRADE LOG  (all trades)")
        print(tdf.to_string(index=False))

    # save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(OUT_DIR / "ema_qqq_trades.csv", index=False)
    qqq[["date", "close", "equity", "bh_equity"]].to_csv(
        OUT_DIR / "ema_qqq_equity_curve.csv", index=False
    )
    print(f"\nSaved: data/models/ema_qqq_trades.csv  |  ema_qqq_equity_curve.csv")


if __name__ == "__main__":
    main()
