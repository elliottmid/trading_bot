#!/usr/bin/env python3
"""
Daily EMA signal scan — 4-asset universe (SPY, XLY, XLF, QQQ).

Entry: EMA(9)  crosses ABOVE EMA(11) → BUY
Exit:  EMA(10) crosses BELOW EMA(20) → SELL
HOLD  — entry fast above entry slow (in position, no new cross)
FLAT  — entry fast below entry slow (not in position, no new cross)

Usage:
    python3 ema_4assets.py
    python3 ema_4assets.py --no-flat
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SYMBOLS = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq 100 ETF",
    "XLF":  "Financials Select Sector SPDR",
    "XLY":  "Consumer Discretionary Select Sector SPDR",
}

ENTRY_FAST = 7
ENTRY_SLOW = 11
EXIT_FAST  = 17
EXIT_SLOW  = 20
MIN_BARS   = EXIT_SLOW * 4


# ── data ─────────────────────────────────────────────────────────────────────

def fetch_recent(symbols: list[str], lookback_days: int = 120) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=start, progress=False, auto_adjust=True)

    if raw.empty:
        logger.error("yfinance returned no data.")
        sys.exit(1)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = symbols

    close.index = pd.to_datetime(close.index)
    long = close.sort_index().stack(future_stack=True).reset_index()
    long.columns = ["date", "symbol", "close"]
    return long.dropna(subset=["close"])


# ── signal logic ─────────────────────────────────────────────────────────────

def compute_signals(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, g in long_df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < MIN_BARS:
            logger.warning(f"{symbol}: only {len(g)} bars, skipping")
            continue

        g["ema_ef"] = g["close"].ewm(span=ENTRY_FAST, adjust=False).mean()
        g["ema_es"] = g["close"].ewm(span=ENTRY_SLOW, adjust=False).mean()
        g["ema_xf"] = g["close"].ewm(span=EXIT_FAST,  adjust=False).mean()
        g["ema_xs"] = g["close"].ewm(span=EXIT_SLOW,  adjust=False).mean()

        today = g.iloc[-1]
        prev  = g.iloc[-2]

        entry_above_today = today["ema_ef"] > today["ema_es"]
        entry_above_prev  = prev["ema_ef"]  > prev["ema_es"]
        exit_above_today  = today["ema_xf"] > today["ema_xs"]
        exit_above_prev   = prev["ema_xf"]  > prev["ema_xs"]

        if not entry_above_prev and entry_above_today:
            signal = "BUY"
        elif exit_above_prev and not exit_above_today:
            signal = "SELL"
        elif entry_above_today:
            signal = "HOLD"
        else:
            signal = "FLAT"

        rows.append({
            "symbol":        symbol,
            "signal":        signal,
            "date":          today["date"].date(),
            "close":         round(today["close"], 2),
            "ema_ef":        round(today["ema_ef"], 4),
            "ema_es":        round(today["ema_es"], 4),
            "ema_xf":        round(today["ema_xf"], 4),
            "ema_xs":        round(today["ema_xs"], 4),
            "entry_gap_pct": round((today["ema_ef"] - today["ema_es"])
                                   / today["ema_es"] * 100, 3),
        })

    return pd.DataFrame(rows)


# ── display ───────────────────────────────────────────────────────────────────

_SIGNAL_ORDER = {"BUY": 0, "SELL": 1, "HOLD": 2, "FLAT": 3}
_SIGNAL_LABEL = {
    "BUY":  "🟢 BUY w 4% trailing stop",
    "SELL": "🔴 SELL",
    "HOLD": "🔵 HOLD",
    "FLAT": "⬜ FLAT",
}


def print_scan(df: pd.DataFrame, show_flat: bool = True) -> None:
    if df.empty:
        print("No signals generated — check your data.")
        return

    df = df.copy()
    df["_order"] = df["signal"].map(_SIGNAL_ORDER)
    df = df.sort_values(["_order", "symbol"]).drop(columns="_order")
    df["description"] = df["symbol"].map(SYMBOLS).fillna("")

    as_of = df["date"].iloc[0]
    print(f"\n{'='*80}")
    print(f"  EMA DAILY SIGNAL SCAN  —  {as_of}")
    print(f"  Entry EMA({ENTRY_FAST}/{ENTRY_SLOW})  |  Exit EMA({EXIT_FAST}/{EXIT_SLOW})")
    print(f"  Universe: SPY  QQQ  XLF  XLY")
    print(f"{'='*80}")

    for signal in ["BUY", "SELL", "HOLD", "FLAT"]:
        if signal == "FLAT" and not show_flat:
            continue
        subset = df[df["signal"] == signal]
        if subset.empty:
            continue
        print(f"\n{_SIGNAL_LABEL[signal]}  ({len(subset)})")
        print(f"  {'Symbol':<6}  {'Close':>7}  {'EF9':>8}  {'ES11':>8}  {'XF10':>8}  {'XS20':>8}  {'Gap%':>7}  Description")
        print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*35}")
        for _, row in subset.iterrows():
            print(
                f"  {row['symbol']:<6}  {row['close']:>7.2f}"
                f"  {row['ema_ef']:>8.4f}  {row['ema_es']:>8.4f}"
                f"  {row['ema_xf']:>8.4f}  {row['ema_xs']:>8.4f}"
                f"  {row['entry_gap_pct']:>+7.3f}  {row['description']}"
            )

    print(f"\n{'='*72}")
    buys  = (df["signal"] == "BUY").sum()
    sells = (df["signal"] == "SELL").sum()
    holds = (df["signal"] == "HOLD").sum()
    flats = (df["signal"] == "FLAT").sum()
    print(f"  BUY {buys}  SELL {sells}  HOLD {holds}  FLAT {flats}")
    print(f"{'='*72}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily EMA signal scan — SPY, QQQ, XLF, XLY"
    )
    parser.add_argument(
        "--no-flat", dest="show_flat", action="store_false",
        help="Suppress FLAT rows"
    )
    parser.add_argument(
        "--lookback", type=int, default=120,
        help="Days of history to fetch for EMA warmup (default: 120)"
    )
    args = parser.parse_args()

    symbols = list(SYMBOLS.keys())
    print(f"Fetching {', '.join(symbols)} via Yahoo Finance…")
    long_df = fetch_recent(symbols, lookback_days=args.lookback)
    signals_df = compute_signals(long_df)
    print_scan(signals_df, show_flat=args.show_flat)


if __name__ == "__main__":
    main()
