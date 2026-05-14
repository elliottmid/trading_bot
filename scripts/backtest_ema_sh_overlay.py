#!/usr/bin/env python3
"""
Backtest: 4-asset EMA strategy (SPY/QQQ/XLF/XLY) with optional SH overlay.

Baseline
--------
  4 assets, equal 25% allocation each.
  Entry : EMA(9)  crosses above EMA(11)
  Exit  : EMA(10) crosses below EMA(20), OR 4% trailing stop

SH Overlay variant
------------------
  Same as baseline, but when SPY fires its EXIT signal (EMA10 < EMA20 cross),
  SPY's 25% slice moves into SH (ProShares Short S&P500) instead of cash.
  SH position exits when SPY fires its ENTRY signal (EMA9 > EMA11 cross).

Benchmark: SPY buy-and-hold over the same window.

Usage
-----
    python3 backtest_ema_sh_overlay.py
    python3 backtest_ema_sh_overlay.py --tc-bps 5
    python3 backtest_ema_sh_overlay.py --start 2022-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── strategy params (grid-search winner) ────────────────────────────────────
ENTRY_FAST    = 9
ENTRY_SLOW    = 11
EXIT_FAST     = 10
EXIT_SLOW     = 20
TRAILING_STOP = 0.04   # 4% trailing stop (optimal from prior backtest)

SYMBOLS   = ["SPY", "QQQ", "XLF", "XLY"]
ALLOC     = 1.0 / len(SYMBOLS)   # 25% each

DATA_DIR  = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR   = Path(__file__).parent.parent / "data" / "models"


# ── data ────────────────────────────────────────────────────────────────────

def load_primary(start: str) -> pd.DataFrame:
    path = DATA_DIR / "ema_etfs_primary_daily.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["symbol"].isin(SYMBOLS)]
    df = df[df["timestamp"] >= start].copy()
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def fetch_sh(start: str, end: str) -> pd.Series:
    """Fetch SH close prices via yfinance; return a date-indexed Series."""
    logger.info("Fetching SH from Yahoo Finance…")
    raw = yf.download("SH", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        logger.error("Could not fetch SH data.")
        sys.exit(1)
    s = raw["Close"].squeeze()
    s.index = pd.to_datetime(s.index)
    s.name = "SH"
    return s


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    periods = sorted({ENTRY_FAST, ENTRY_SLOW, EXIT_FAST, EXIT_SLOW})
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("timestamp")
        for p in periods:
            g[f"ema_{p}"] = g["close"].ewm(span=p, adjust=False).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ── signal helpers ───────────────────────────────────────────────────────────

def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    above     = (fast > slow).astype(int)
    above_lag = above.shift(1)
    sig = pd.Series(0, index=fast.index, dtype=int)
    sig[(above_lag == 0) & (above == 1)] =  1
    sig[(above_lag == 1) & (above == 0)] = -1
    return sig


# ── per-symbol state machine ─────────────────────────────────────────────────

def simulate_symbol(sym_df: pd.DataFrame, tc_frac: float) -> pd.DataFrame:
    """
    Return a frame with columns: date, close, daily_ret, in_position.
    daily_ret is the strategy's MTM return for that day (0 when flat).
    """
    sym_df = sym_df.sort_values("timestamp").reset_index(drop=True)

    entry_sig = _crossover(sym_df[f"ema_{ENTRY_FAST}"], sym_df[f"ema_{ENTRY_SLOW}"])
    exit_sig  = _crossover(sym_df[f"ema_{EXIT_FAST}"],  sym_df[f"ema_{EXIT_SLOW}"])

    close      = sym_df["close"].to_numpy()
    entry_arr  = entry_sig.to_numpy()
    exit_arr   = exit_sig.to_numpy()
    n          = len(close)

    in_pos      = False
    entry_px    = 0.0
    peak_px     = 0.0
    prev_eq     = 0.0
    daily_ret   = np.zeros(n)
    in_position = np.zeros(n, dtype=bool)

    for i in range(1, n):
        s = i - 1  # signal index (lag-1)

        if not in_pos and entry_arr[s] == 1:
            in_pos   = True
            entry_px = close[i]
            peak_px  = close[i]
            prev_eq  = 0.0
            daily_ret[i] -= tc_frac

        if in_pos:
            if close[i] > peak_px:
                peak_px = close[i]

            eq = (close[i] - entry_px) / entry_px
            daily_ret[i] += eq - prev_eq
            prev_eq = eq
            in_position[i] = True

            stop_hit = close[i] < peak_px * (1 - TRAILING_STOP)
            exit_hit = exit_arr[s] == -1

            if stop_hit or exit_hit:
                daily_ret[i] -= tc_frac
                in_pos   = False
                prev_eq  = 0.0
                in_position[i] = False  # flat at end of bar (already booked P&L)

    result = sym_df[["timestamp", "close"]].copy()
    result.columns = ["date", "close"]
    result["daily_ret"]   = daily_ret
    result["in_position"] = in_position
    # carry exit signal for SH logic
    result["entry_signal"] = entry_arr
    result["exit_signal"]  = exit_arr
    return result.reset_index(drop=True)


def simulate_sh_slice(spy_sim: pd.DataFrame, sh_prices: pd.Series, tc_frac: float) -> np.ndarray:
    """
    Simulate SPY's 25% slice going into SH when SPY fires EXIT, returning
    to cash when SPY fires ENTRY.
    Returns array of daily_ret (for the SH slice only, unscaled by allocation).
    """
    dates     = spy_sim["date"].values
    entry_arr = spy_sim["entry_signal"].values
    exit_arr  = spy_sim["exit_signal"].values
    n         = len(dates)

    in_sh       = False
    entry_sh_px = 0.0
    prev_eq_sh  = 0.0
    daily_ret   = np.zeros(n)

    for i in range(1, n):
        s = i - 1
        date_i = pd.Timestamp(dates[i])

        if not in_sh and exit_arr[s] == -1:
            # SPY exit signal → enter SH
            if date_i in sh_prices.index:
                in_sh       = True
                entry_sh_px = sh_prices.loc[date_i]
                prev_eq_sh  = 0.0
                daily_ret[i] -= tc_frac

        if in_sh:
            if date_i not in sh_prices.index:
                continue
            sh_px  = sh_prices.loc[date_i]
            eq     = (sh_px - entry_sh_px) / entry_sh_px
            daily_ret[i] += eq - prev_eq_sh
            prev_eq_sh = eq

            if entry_arr[s] == 1:
                # SPY entry signal → exit SH back to cash
                daily_ret[i] -= tc_frac
                in_sh      = False
                prev_eq_sh = 0.0

    return daily_ret


# ── portfolio aggregation ────────────────────────────────────────────────────

def build_portfolio(
    sim_map: dict[str, pd.DataFrame],
    sh_slice: np.ndarray | None,
    spy_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Equal-weight portfolio daily returns.
    If sh_slice provided, SPY's FLAT cash is replaced by SH P&L.
    """
    all_sims = list(sim_map.values())
    base_df  = all_sims[0][["date"]].copy()

    port_ret = np.zeros(len(base_df))
    for sym, sim in sim_map.items():
        port_ret += sim["daily_ret"].to_numpy() * ALLOC

    if sh_slice is not None:
        port_ret += sh_slice * ALLOC

    equity = pd.Series((1 + port_ret).cumprod(), index=base_df["date"])
    equity.index = pd.to_datetime(equity.index)
    return equity


# ── metrics ──────────────────────────────────────────────────────────────────

def metrics(equity: pd.Series, label: str) -> dict:
    daily_ret = equity.pct_change().dropna()
    years     = (equity.index[-1] - equity.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr      = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    vol       = daily_ret.std() * np.sqrt(252)
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    roll_max  = equity.cummax()
    dd        = (equity - roll_max) / roll_max
    max_dd    = dd.min()
    return {
        "label":    label,
        "cagr":     cagr,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
        "total_ret": total_ret,
        "vol":       vol,
    }


def yearly_returns(equity: pd.Series, label: str) -> pd.DataFrame:
    yr = equity.resample("YE").last()
    yr_ret = yr.pct_change().dropna()
    # also include first year from start
    first_yr = equity[equity.index.year == equity.index[0].year]
    first_ret = first_yr.iloc[-1] / first_yr.iloc[0] - 1
    rows = [{"year": equity.index[0].year, label: first_ret}]
    for dt, r in yr_ret.items():
        rows.append({"year": dt.year, label: r})
    return pd.DataFrame(rows).set_index("year")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-bps", type=float, default=5)
    parser.add_argument("--start",  default="2021-01-01")
    args = parser.parse_args()

    tc_frac = args.tc_bps / 10_000

    # ── load data ──
    df = load_primary(args.start)
    df = add_emas(df)
    logger.info(f"Data: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    sh_raw   = fetch_sh(args.start, end_date)

    # ── simulate each symbol ──
    sim_map: dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("symbol"):
        sim_map[sym] = simulate_symbol(g.reset_index(drop=True), tc_frac)

    # align all sims to same date index (inner join)
    common_dates = None
    for sim in sim_map.values():
        d = set(pd.to_datetime(sim["date"]))
        common_dates = d if common_dates is None else common_dates & d
    common_dates = sorted(common_dates)

    for sym in sim_map:
        sim_map[sym] = sim_map[sym][
            pd.to_datetime(sim_map[sym]["date"]).isin(common_dates)
        ].reset_index(drop=True)

    spy_sim  = sim_map["SPY"]
    sh_align = sh_raw.reindex(pd.to_datetime(spy_sim["date"].values)).ffill()

    # ── SH slice ──
    sh_daily = simulate_sh_slice(spy_sim, sh_align, tc_frac)

    # ── build equity curves ──
    equity_base = build_portfolio(sim_map, sh_slice=None,     spy_dates=None)
    equity_sh   = build_portfolio(sim_map, sh_slice=sh_daily, spy_dates=None)

    # SPY buy-and-hold benchmark
    spy_close = spy_sim.set_index(pd.to_datetime(spy_sim["date"]))["close"]
    equity_bh = spy_close / spy_close.iloc[0]

    # ── summary table ──
    rows = [
        metrics(equity_base, "4-asset EMA + trailing stop"),
        metrics(equity_sh,   "4-asset EMA + trailing stop + SH"),
        metrics(equity_bh,   "SPY buy-and-hold"),
    ]
    summary = pd.DataFrame(rows).set_index("label")

    print("\n" + "=" * 72)
    print(f"EMA STRATEGY — SH OVERLAY COMPARISON")
    print(f"Period : {common_dates[0].date()} → {common_dates[-1].date()}")
    print(f"TC     : {args.tc_bps}bps  |  Trailing stop: {TRAILING_STOP:.0%}")
    print("=" * 72)
    print(summary[["cagr", "sharpe", "max_dd", "vol"]].rename(columns={
        "cagr": "CAGR", "sharpe": "Sharpe", "max_dd": "MaxDD", "vol": "AnnVol"
    }).map(lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.3f}").to_string())

    # ── year-by-year ──
    y_base = yearly_returns(equity_base, "baseline")
    y_sh   = yearly_returns(equity_sh,   "sh_overlay")
    y_bh   = yearly_returns(equity_bh,   "spy_bh")
    yearly = y_base.join(y_sh).join(y_bh)

    print("\n" + "-" * 50)
    print("YEAR-BY-YEAR RETURNS")
    print("-" * 50)
    print(yearly.map(lambda x: f"{x:+.1%}" if pd.notna(x) else "—").to_string())

    # ── SH trade log ──
    n_sh_trades = int((np.diff(np.where(sh_daily != 0, 1, 0)) > 0).sum()) if sh_daily.any() else 0
    print(f"\nSH trades entered : {n_sh_trades}")
    print("=" * 72 + "\n")

    # save equity curves
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "date":     pd.to_datetime(spy_sim["date"]),
        "baseline": equity_base.values,
        "sh_overlay": equity_sh.values,
        "spy_bh":   equity_bh.values,
    })
    out_path = OUT_DIR / "ema_sh_overlay_equity.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Equity curves saved → {out_path}")


if __name__ == "__main__":
    main()
