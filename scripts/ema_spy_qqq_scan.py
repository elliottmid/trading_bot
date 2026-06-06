#!/usr/bin/env python3
"""
Daily EMA signal scanner — SPY and QQQ with individually optimised parameters.

SPY: Entry EMA(12/16)  → Exit EMA(8/10) + 6% trailing stop
QQQ: Entry EMA(12/24) → Exit EMA(10/23) + 5% trailing stop

ML exit filter: suppresses EMA/trail exits when SP500/NDX MODERATE monthly
regressor predicts a return above ML_THRESHOLD for the current month.
Walk-forward validated (2010–2026): avg IS-optimal threshold −2.1% (SPY),
+0.2% (QQQ) — using 0.0% for both as practical simplification.

Reconstructs the current open trade from 2-year history so it can report:
  • HOLD  — entry date/price, peak price, live trailing stop level
  • HOLD* — exit fired but suppressed by ML filter (stays long)
  • BUY   — suggested initial stop price at today's close
  • SELL  — exit triggered, ML filter did not suppress
  • FLAT  — no position, no entry signal

Usage:
    python3 ema_spy_qqq_scan.py
    python3 ema_spy_qqq_scan.py --lookback 600   # more history for warmup
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── ML exit-filter configuration ─────────────────────────────────────────────
R_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/R"
ML_PATTERN = {
    "SPY": "sp500_moderate_results_*.csv",
    "QQQ": "ndx_moderate_results_*.csv",
}
# Suppress exit if ML predicted return > threshold.
# Walk-forward avg IS-optimal: SPY −2.1%, QQQ +0.2% (2010–2026).
ML_THRESHOLD = {"SPY": 0.0, "QQQ": 0.0}

# ── per-symbol strategy parameters ───────────────────────────────────────────
PARAMS = {
    "SPY": {
        "name":        "S&P 500 ETF",
        "entry_fast":  12,
        "entry_slow":  16,
        "exit_fast":   8,
        "exit_slow":   10,
        "trail_pct":   0.06,
    },
    "QQQ": {
        "name":        "Nasdaq-100 ETF",
        "entry_fast":  12,
        "entry_slow":  24,
        "exit_fast":   10,
        "exit_slow":   23,
        "trail_pct":   0.05,
    },
}

LOOKBACK_DAYS = 504   # ~2 trading years; enough to capture any open trade


# ── data ─────────────────────────────────────────────────────────────────────

def fetch(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    raw = yf.download(symbols, start=start, progress=False, auto_adjust=True)
    if raw.empty:
        print("ERROR: yfinance returned no data.")
        sys.exit(1)

    close = (
        raw["Close"]
        if isinstance(raw.columns, pd.MultiIndex)
        else raw[["Close"]].rename(columns={"Close": symbols[0]})
    )
    close.index = pd.to_datetime(close.index)
    out = {}
    for sym in symbols:
        s = close[sym].dropna().sort_index().reset_index()
        s.columns = ["date", "close"]
        out[sym] = s
    return out


# ── ML prediction loader ──────────────────────────────────────────────────────

def load_ml_predictions() -> dict[str, dict | None]:
    """Return most recent ML predicted return for each symbol.

    The MODERATE model is run monthly. File name encodes the generation date:
    sp500_moderate_results_YY-MM-DD.csv → forecast for that month.
    The scanner always uses the most recent file; glob picks up new files
    automatically each month. Forecast is flagged stale if > 1 month old.

    Result keys: pred (float %), gen_month (str), is_active (bool).
    Returns None for a symbol when no file or no valid prediction row is found.
    """
    import re
    today = pd.Timestamp.today()
    current_month = today.to_period("M")
    out: dict[str, dict | None] = {}
    for sym, pattern in ML_PATTERN.items():
        files = sorted(R_DIR.glob(pattern))
        if not files:
            out[sym] = None
            continue
        fname = files[-1].name
        m = re.search(r'(\d{2})-(\d{2})-\d{2}', fname)
        if not m:
            out[sym] = None
            continue
        yy, mm = int(m.group(1)), int(m.group(2))
        gen_month = pd.Timestamp(year=2000 + yy, month=mm, day=1).to_period("M")
        df = pd.read_csv(files[-1], parse_dates=["Date"])
        valid = df.dropna(subset=["Predicted_Return"])
        if valid.empty:
            out[sym] = None
            continue
        row = valid.iloc[-1]
        out[sym] = {
            "pred":      float(row["Predicted_Return"]),
            "gen_month": str(gen_month),
            # Stale if forecast is more than one month old
            "is_active": (current_month - gen_month).n <= 1,
        }
    return out


# ── EMA crossover helpers ─────────────────────────────────────────────────────

def _cross_up(fast: pd.Series, slow: pd.Series, i: int) -> bool:
    return i > 0 and fast.iloc[i] > slow.iloc[i] and fast.iloc[i - 1] <= slow.iloc[i - 1]


def _cross_dn(fast: pd.Series, slow: pd.Series, i: int) -> bool:
    return i > 0 and fast.iloc[i] < slow.iloc[i] and fast.iloc[i - 1] >= slow.iloc[i - 1]


# ── position reconstruction + today's signal ─────────────────────────────────

def analyse(sym: str, df: pd.DataFrame, p: dict, ml_info: dict | None = None) -> dict:
    ef, es = p["entry_fast"], p["entry_slow"]
    xf, xs = p["exit_fast"],  p["exit_slow"]
    trail  = p["trail_pct"]

    df = df.copy().reset_index(drop=True)
    df["ema_ef"] = df["close"].ewm(span=ef, adjust=False).mean()
    df["ema_es"] = df["close"].ewm(span=es, adjust=False).mean()
    df["ema_xf"] = df["close"].ewm(span=xf, adjust=False).mean()
    df["ema_xs"] = df["close"].ewm(span=xs, adjust=False).mean()

    in_pos      = False
    entry_price = 0.0
    entry_date  = None
    peak        = 0.0

    ml_pred      = ml_info["pred"] if ml_info else None
    ml_threshold = ML_THRESHOLD[sym]
    ml_filter_on = (ml_info is not None and ml_info.get("is_active", False))

    # Walk every bar to reconstruct current trade state.
    # Signal at bar i fires at close of bar i; execution notionally at
    # next open, but for end-of-day scanning we treat today's close as entry.
    for i in range(1, len(df)):
        close_i = df["close"].iloc[i]

        entry_signal = _cross_up(df["ema_ef"], df["ema_es"], i)
        exit_signal  = _cross_dn(df["ema_xf"], df["ema_xs"], i)

        if not in_pos and entry_signal:
            in_pos      = True
            entry_price = close_i
            entry_date  = df["date"].iloc[i]
            peak        = close_i

        if in_pos:
            peak = max(peak, close_i)
            trail_hit = close_i <= peak * (1 - trail)
            if trail_hit or exit_signal:
                in_pos      = False
                entry_price = 0.0
                entry_date  = None
                peak        = 0.0

    # ── today's bar ──────────────────────────────────────────────────────────
    last   = df.iloc[-1]
    prev   = df.iloc[-2]
    close  = last["close"]
    date   = last["date"].date() if hasattr(last["date"], "date") else last["date"]

    entry_now = _cross_up(df["ema_ef"], df["ema_es"], len(df) - 1)
    exit_now  = _cross_dn(df["ema_xf"], df["ema_xs"], len(df) - 1)

    ml_suppressed = False
    exit_reason   = None

    if in_pos:
        trail_stop = round(peak * (1 - trail), 2)
        unrealised = (close - entry_price) / entry_price
        trail_hit       = close <= trail_stop
        exit_triggered  = exit_now or trail_hit
        exit_reason     = ("trail" if trail_hit else "ema") if exit_triggered else None

        if exit_triggered and ml_filter_on and ml_pred > ml_threshold:
            signal       = "HOLD"
            ml_suppressed = True
        elif exit_triggered:
            signal = "SELL"
        else:
            signal = "HOLD"
    else:
        trail_stop  = None
        unrealised  = None
        if entry_now:
            signal = "BUY"
            # Seed the position for display purposes
            entry_price = close
            entry_date  = date
            peak        = close
            trail_stop  = round(close * (1 - trail), 2)
        else:
            signal = "FLAT"

    return {
        "symbol":       sym,
        "name":         p["name"],
        "signal":       signal,
        "date":         date,
        "close":        round(close, 2),
        "trail_pct":    trail,
        "entry_price":  round(entry_price, 2) if entry_price else None,
        "entry_date":   entry_date,
        "peak":         round(peak, 2) if peak else None,
        "trail_stop":   trail_stop,
        "unrealised":   unrealised,
        "exit_reason":  exit_reason,
        # ML filter
        "ml_pred":         ml_pred,
        "ml_month":        ml_info["gen_month"] if ml_info else None,
        "ml_suppressed":   ml_suppressed,
        "ml_threshold":    ml_threshold,
        "ml_filter_on":    ml_filter_on,
        # EMA values for diagnostics
        "ema_ef":       round(last["ema_ef"], 4),
        "ema_es":       round(last["ema_es"], 4),
        "ema_xf":       round(last["ema_xf"], 4),
        "ema_xs":       round(last["ema_xs"], 4),
        "entry_gap":    round((last["ema_ef"] - last["ema_es"]) / last["ema_es"] * 100, 3),
    }


# ── display ───────────────────────────────────────────────────────────────────

_ICONS = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵", "FLAT": "⬜"}


def _ml_status_line(r: dict) -> str:
    """One-line ML filter summary for the scan header."""
    if not r["ml_filter_on"]:
        if r["ml_month"]:
            return f"ML filter: OFF ({r['ml_month']} forecast is stale — re-run MODERATE model)"
        return "ML filter: OFF (no forecast file found)"
    sign = "+" if r["ml_pred"] >= 0 else ""
    return (f"ML filter: ON  |  {r['ml_month']} forecast {sign}{r['ml_pred']:.2f}%  "
            f"(threshold {r['ml_threshold']:+.1f}%)")


def print_scan(results: list[dict]) -> None:
    as_of = results[0]["date"]
    order = {"BUY": 0, "SELL": 1, "HOLD": 2, "FLAT": 3}
    results = sorted(results, key=lambda r: order[r["signal"]])

    print(f"\n{'='*72}")
    print(f"  SPY / QQQ DAILY SIGNAL SCAN  —  {as_of}")
    for sym, p in PARAMS.items():
        print(f"  {sym}: Entry EMA({p['entry_fast']}/{p['entry_slow']})  |  "
              f"Exit EMA({p['exit_fast']}/{p['exit_slow']})  |  "
              f"{p['trail_pct']:.0%} trail")
    print(f"{'='*72}")

    for r in results:
        sig   = r["signal"]
        icon  = "🟡" if r["ml_suppressed"] else _ICONS[sig]
        trail = r["trail_pct"]

        label = f"{sig} (exit suppressed by ML filter)" if r["ml_suppressed"] else sig
        print(f"\n{icon} {label}  —  {r['symbol']}  ({r['name']})")
        print(f"   Close : ${r['close']:.2f}   |   As of : {r['date']}")
        print(f"   {_ml_status_line(r)}")

        if sig == "BUY":
            print(f"   ── ACTION ─────────────────────────────────────────────────────")
            print(f"   Enter at tomorrow's open (or today's close after hours)")
            print(f"   Set a {trail:.0%} trailing stop — stop rises with price,")
            print(f"   triggered when price drops {trail:.0%} from its highest close since entry.")
            print(f"   Initial stop at today's close : ${r['trail_stop']:.2f}  "
                  f"({trail:.0%} below ${r['close']:.2f})")
            print(f"   ── EMAs ──────────────────────────────────────────────────────")
            p = PARAMS[r["symbol"]]
            print(f"   Entry EMA({p['entry_fast']}) {r['ema_ef']:.4f}  >  "
                  f"EMA({p['entry_slow']}) {r['ema_es']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")

        elif sig == "HOLD":
            pnl_str = f"{r['unrealised']:+.2%}" if r["unrealised"] is not None else "n/a"
            edate = str(r["entry_date"])[:10]
            if r["ml_suppressed"]:
                reason = r["exit_reason"].upper() if r["exit_reason"] else "SIGNAL"
                print(f"   ── EXIT SUPPRESSED ────────────────────────────────────────────")
                sign = "+" if r["ml_pred"] >= 0 else ""
                print(f"   {reason} exit fired — overridden: pred {sign}{r['ml_pred']:.2f}%  "
                      f">  threshold {r['ml_threshold']:+.1f}%  → staying long")
            print(f"   ── OPEN TRADE ─────────────────────────────────────────────────")
            print(f"   Entry : ${r['entry_price']:.2f}  on  {edate}")
            print(f"   Peak  : ${r['peak']:.2f}   |   Unrealised : {pnl_str}")
            print(f"   Trailing stop ({trail:.0%}) : ${r['trail_stop']:.2f}  "
                  f"({'%.1f' % ((r['close'] / r['trail_stop'] - 1) * 100)}% cushion)")
            print(f"   ── EMAs ──────────────────────────────────────────────────────")
            p = PARAMS[r["symbol"]]
            print(f"   Entry EMA({p['entry_fast']}) {r['ema_ef']:.4f}  vs  "
                  f"EMA({p['entry_slow']}) {r['ema_es']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")
            print(f"   Exit  EMA({p['exit_fast']}) {r['ema_xf']:.4f}  vs  "
                  f"EMA({p['exit_slow']}) {r['ema_xs']:.4f}")

        elif sig == "SELL":
            pnl_str = f"{r['unrealised']:+.2%}" if r["unrealised"] is not None else "n/a"
            edate = str(r["entry_date"])[:10]
            print(f"   ── ACTION ─────────────────────────────────────────────────────")
            print(f"   Exit at tomorrow's open")
            print(f"   Entry was : ${r['entry_price']:.2f}  on  {edate}")
            print(f"   Unrealised P&L at today's close : {pnl_str}")
            if r["ml_filter_on"] and r["ml_pred"] is not None:
                sign = "+" if r["ml_pred"] >= 0 else ""
                print(f"   ML filter did not suppress: pred {sign}{r['ml_pred']:.2f}%  "
                      f"≤  threshold {r['ml_threshold']:+.1f}%")
            elif not r["ml_filter_on"]:
                print(f"   ML filter inactive (no recent prediction) — exit proceeds")

        elif sig == "FLAT":
            p = PARAMS[r["symbol"]]
            print(f"   No position.  Watching for EMA({p['entry_fast']}) "
                  f"> EMA({p['entry_slow']}) crossover.")
            print(f"   Entry EMA({p['entry_fast']}) {r['ema_ef']:.4f}  vs  "
                  f"EMA({p['entry_slow']}) {r['ema_es']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")

    print(f"\n{'='*72}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily EMA signal scan — SPY (4% trail) and QQQ (5% trail)"
    )
    parser.add_argument(
        "--lookback", type=int, default=LOOKBACK_DAYS,
        help=f"Calendar days of history to fetch (default: {LOOKBACK_DAYS})"
    )
    args = parser.parse_args()

    symbols = list(PARAMS.keys())
    print(f"Fetching {', '.join(symbols)} — {args.lookback} days of history…")
    data = fetch(symbols, args.lookback)

    print("Loading ML predictions…")
    ml_preds = load_ml_predictions()
    for sym, info in ml_preds.items():
        if info is None:
            print(f"  {sym}: no prediction file found — filter inactive")
        elif not info["is_active"]:
            print(f"  {sym}: {info['gen_month']} forecast is stale — filter inactive")
        else:
            sign = "+" if info["pred"] >= 0 else ""
            print(f"  {sym}: {info['gen_month']} forecast {sign}{info['pred']:.2f}%  "
                  f"vs threshold {ML_THRESHOLD[sym]:+.1f}%")

    results = [analyse(sym, data[sym], PARAMS[sym], ml_preds.get(sym)) for sym in symbols]
    print_scan(results)


if __name__ == "__main__":
    main()
