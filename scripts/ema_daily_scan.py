#!/usr/bin/env python3
"""
Daily EMA(3/8) signal scan for the ETF universe.

For each ETF, computes EMA(3) and EMA(8) on recent daily closes and
reports today's signal:

  BUY  — EMA3 crossed ABOVE EMA8 today
  SELL — EMA3 crossed BELOW EMA8 today
  HOLD — EMA3 is above EMA8 (in position, no new cross)
  FLAT — EMA3 is below EMA8 (not in position, no new cross)

Run daily after market close:
    python3 ema_daily_scan.py
    python3 ema_daily_scan.py --universe full
    python3 ema_daily_scan.py --signals buy sell hold   # filter specific signals
    python3 ema_daily_scan.py --no-flat                 # suppress FLAT rows
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from ema_etf_universe import PRIMARY_UNIVERSE, UNIVERSE

logging.basicConfig(
    level=logging.WARNING,           # keep yfinance noise off by default
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FAST = 3
SLOW = 8
# Minimum bars needed for reliable EMA(8): warmup + a few extra
MIN_BARS = SLOW * 4


# ── data ─────────────────────────────────────────────────────────────────────

def fetch_recent(symbols: list, lookback_days: int = 90) -> pd.DataFrame:
    """
    Pull the last `lookback_days` of daily closes for all symbols at once.
    Returns long-form DataFrame with columns: symbol, date, close.
    """
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    tickers = [s for s in symbols if s not in ("CurrencyShares",)]  # skip placeholder

    raw = yf.download(
        tickers,
        start=start,
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        logger.error("yfinance returned no data.")
        sys.exit(1)

    # yfinance multi-ticker → MultiIndex columns (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # Single ticker
        close = raw[["Close"]]
        close.columns = tickers

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    long = (
        close.stack(future_stack=True)
        .reset_index()
    )
    long.columns = ["date", "symbol", "close"]
    long = long.dropna(subset=["close"])
    return long


# ── signal logic ─────────────────────────────────────────────────────────────

def compute_signals(long_df: pd.DataFrame, fast: int = FAST, slow: int = SLOW) -> pd.DataFrame:
    """
    Compute EMA(fast)/EMA(slow) for each symbol and return today's signal row.
    """
    rows = []
    for symbol, g in long_df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_BARS:
            logger.warning(f"{symbol}: only {len(g)} bars, skipping")
            continue

        g["ema_fast"] = g["close"].ewm(span=fast, adjust=False).mean()
        g["ema_slow"] = g["close"].ewm(span=slow, adjust=False).mean()

        today = g.iloc[-1]
        prev  = g.iloc[-2]

        fast_above_today = today["ema_fast"] > today["ema_slow"]
        fast_above_prev  = prev["ema_fast"]  > prev["ema_slow"]

        if not fast_above_prev and fast_above_today:
            signal = "BUY"
        elif fast_above_prev and not fast_above_today:
            signal = "SELL"
        elif fast_above_today:
            signal = "HOLD"
        else:
            signal = "FLAT"

        rows.append({
            "symbol":    symbol,
            "signal":    signal,
            "date":      today["date"].date(),
            "close":     round(today["close"], 2),
            "ema_fast":  round(today["ema_fast"], 4),
            "ema_slow":  round(today["ema_slow"], 4),
            "gap_pct":   round((today["ema_fast"] - today["ema_slow"])
                               / today["ema_slow"] * 100, 3),
        })

    return pd.DataFrame(rows)


# ── display ───────────────────────────────────────────────────────────────────

_SIGNAL_ORDER = {"BUY": 0, "SELL": 1, "HOLD": 2, "FLAT": 3}
_SIGNAL_LABEL = {
    "BUY":  "🟢 BUY",
    "SELL": "🔴 SELL",
    "HOLD": "🔵 HOLD",
    "FLAT": "⬜ FLAT",
}


def print_scan(df: pd.DataFrame, universe_desc: dict, show_flat: bool = True) -> None:
    if df.empty:
        print("No signals generated — check your data.")
        return

    df = df.copy()
    df["_order"] = df["signal"].map(_SIGNAL_ORDER)
    df = df.sort_values(["_order", "symbol"]).drop(columns="_order")

    # Attach descriptions
    df["description"] = df["symbol"].map(universe_desc).fillna("")

    as_of = df["date"].iloc[0]
    print(f"\n{'='*72}")
    print(f"  EMA({FAST}/{SLOW}) DAILY SIGNAL SCAN  —  {as_of}")
    print(f"{'='*72}")

    for signal in ["BUY", "SELL", "HOLD", "FLAT"]:
        if signal == "FLAT" and not show_flat:
            continue
        subset = df[df["signal"] == signal]
        if subset.empty:
            continue
        label = _SIGNAL_LABEL[signal]
        print(f"\n{label}  ({len(subset)})")
        print(f"  {'Symbol':<8}  {'Close':>7}  {'EMA3':>8}  {'EMA8':>8}  {'Gap%':>7}  Description")
        print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*30}")
        for _, row in subset.iterrows():
            print(
                f"  {row['symbol']:<8}  {row['close']:>7.2f}"
                f"  {row['ema_fast']:>8.4f}  {row['ema_slow']:>8.4f}"
                f"  {row['gap_pct']:>+7.3f}  {row['description']}"
            )

    print(f"\n{'='*72}")
    buys  = (df["signal"] == "BUY").sum()
    sells = (df["signal"] == "SELL").sum()
    holds = (df["signal"] == "HOLD").sum()
    flats = (df["signal"] == "FLAT").sum()
    print(f"  Universe: {len(df)} ETFs  |  "
          f"BUY {buys}  SELL {sells}  HOLD {holds}  FLAT {flats}")
    print(f"{'='*72}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily EMA(3/8) signal scan for ETF universe"
    )
    parser.add_argument(
        "--universe", choices=["primary", "full"], default="primary",
        help="ETF universe to scan (default: primary)"
    )
    parser.add_argument(
        "--signals", nargs="+",
        choices=["buy", "sell", "hold", "flat"],
        default=None,
        help="Only show these signal types (default: all)"
    )
    parser.add_argument(
        "--no-flat", action="store_true",
        help="Suppress FLAT (not in position, no cross) rows"
    )
    parser.add_argument(
        "--lookback", type=int, default=90,
        help="Days of history to fetch for EMA warmup (default: 90)"
    )
    parser.add_argument(
        "--fast", type=int, default=FAST,
        help=f"Fast EMA period (default: {FAST})"
    )
    parser.add_argument(
        "--slow", type=int, default=SLOW,
        help=f"Slow EMA period (default: {SLOW})"
    )
    args = parser.parse_args()

    # Override module-level constants if user passed custom periods
    global FAST, SLOW, MIN_BARS
    FAST     = args.fast
    SLOW     = args.slow
    MIN_BARS = SLOW * 4

    universe = PRIMARY_UNIVERSE if args.universe == "primary" else UNIVERSE
    symbols  = list(universe.keys())

    print(f"Fetching {len(symbols)} ETFs via Yahoo Finance…")
    long_df = fetch_recent(symbols, lookback_days=args.lookback)

    signals_df = compute_signals(long_df, fast=FAST, slow=SLOW)

    # Optional filter
    if args.signals:
        keep = {s.upper() for s in args.signals}
        signals_df = signals_df[signals_df["signal"].isin(keep)]

    show_flat = not args.no_flat
    print_scan(signals_df, universe, show_flat=show_flat)


if __name__ == "__main__":
    main()
