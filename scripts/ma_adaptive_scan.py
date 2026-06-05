#!/usr/bin/env python3
"""
Adaptive daily MA signal scanner — SPY and QQQ.

Finds Sharpe-maximizing entry/exit MA parameters by running the same IS grid
search used in walkforward_ma_optimization.py on a lookback-sweep-optimal IS
window (EMA/Wilder: 9yr / 108m; SMA: 14yr / 168m), then applies those params
to reconstruct the current trade state and report today's signal.

Moving average type is selectable: sma (default), ema, or wilder.

Because the grid search takes ~3–6 minutes, optimized parameters are cached
to data/models/ma_adaptive_params.json. On subsequent runs the cache is reused
unless --reoptimize is passed or the cache is more than --cache-days old
(default 30 days — params are stable month to month).

Usage
-----
    python3 ma_adaptive_scan.py                    # SMA, use/refresh cache
    python3 ma_adaptive_scan.py --ma-type ema      # EMA
    python3 ma_adaptive_scan.py --ma-type wilder   # Wilder smoothing
    python3 ma_adaptive_scan.py --reoptimize       # force fresh grid search
    python3 ma_adaptive_scan.py --cache-days 0     # always reoptimize
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, cpu_count, delayed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_PATH  = Path(__file__).parent.parent / "data" / "models" / "ma_adaptive_params.json"
SYMBOLS     = ["SPY", "QQQ"]
IS_YEARS    = {"ema": 9, "sma": 14, "wilder": 9}   # lookback-sweep optima (EMA/Wilder: 108m, SMA: 168m)
TRAIL_STOPS = [0.03, 0.04, 0.05, 0.06]
TC_BPS      = 5
MA_TYPES    = ("ema", "sma", "wilder")

_ICONS = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵", "FLAT": "⬜"}


# ── data ──────────────────────────────────────────────────────────────────────

def fetch_history(symbols: list[str], years: int = 15) -> dict[str, pd.DataFrame]:
    """Fetch ~(years + buffer) of daily closes via yfinance."""
    start = (datetime.today() - timedelta(days=int(years * 365.25) + 60)).strftime("%Y-%m-%d")
    logger.info(f"Fetching {', '.join(symbols)} from {start} …")
    raw = yf.download(symbols, start=start, progress=False, auto_adjust=True)
    if raw.empty:
        logger.error("yfinance returned no data.")
        sys.exit(1)

    close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
             else raw[["Close"]].rename(columns={"Close": symbols[0]}))
    close.index = pd.to_datetime(close.index)

    out = {}
    for sym in symbols:
        s = close[sym].dropna().sort_index().reset_index()
        s.columns = ["date", "close"]
        out[sym] = s
    return out


# ── MA computation ─────────────────────────────────────────────────────────────

def compute_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "wilder":
        return series.ewm(alpha=1.0 / period, adjust=False).mean()
    elif ma_type == "sma":
        return series.rolling(window=period, min_periods=period).mean()
    raise ValueError(f"Unknown ma_type: {ma_type!r}")


# ── grid ───────────────────────────────────────────────────────────────────────

def build_combos() -> list[tuple[int, int, int, int, float]]:
    pairs = [(f, s) for f in range(3, 13) for s in range(8, 27) if f < s]
    return [
        (ef, es, xf, xs, ts)
        for (ef, es), (xf, xs), ts in product(pairs, pairs, TRAIL_STOPS)
    ]


# ── fast IS backtest (Sharpe only) ────────────────────────────────────────────

def _sharpe_only(
    close_arr: np.ndarray,
    fe: np.ndarray, se: np.ndarray,
    fx: np.ndarray, sx: np.ndarray,
    trail_pct: float,
    tc_frac:   float,
) -> float:
    in_pos = False
    entry_price = peak = prev_eq = 0.0
    s = ss = 0.0
    n_days = 0

    for i in range(2, len(close_arr)):
        sig = i - 1
        if (np.isnan(fe[sig]) or np.isnan(se[sig]) or
                np.isnan(fx[sig]) or np.isnan(sx[sig]) or
                np.isnan(fe[sig-1]) or np.isnan(se[sig-1]) or
                np.isnan(fx[sig-1]) or np.isnan(sx[sig-1])):
            continue

        entry_cross = fe[sig] > se[sig] and fe[sig-1] <= se[sig-1]
        exit_cross  = fx[sig] < sx[sig] and fx[sig-1] >= sx[sig-1]
        dr = 0.0

        if not in_pos and entry_cross:
            in_pos = True
            entry_price = close_arr[i]
            peak = close_arr[i]
            prev_eq = 0.0
            dr -= tc_frac

        if in_pos:
            if close_arr[i] > peak:
                peak = close_arr[i]
            eq = (close_arr[i] - entry_price) / entry_price
            dr += eq - prev_eq
            prev_eq = eq
            if close_arr[i] <= peak * (1 - trail_pct) or exit_cross:
                dr -= tc_frac
                in_pos = False
                prev_eq = 0.0

        s  += dr
        ss += dr * dr
        n_days += 1

    if n_days < 2:
        return 0.0
    mean = s / n_days
    var  = ss / n_days - mean * mean
    return float(mean / max(var, 1e-12) ** 0.5 * 252 ** 0.5)


def _run_batch(
    close_arr: np.ndarray,
    ma_dict:   dict[int, np.ndarray],
    combos:    list[tuple],
    tc_frac:   float,
) -> list[tuple[float, tuple]]:
    return [
        (_sharpe_only(close_arr, ma_dict[ef], ma_dict[es],
                      ma_dict[xf], ma_dict[xs], ts, tc_frac),
         (ef, es, xf, xs, ts))
        for ef, es, xf, xs, ts in combos
    ]


# ── IS optimization ───────────────────────────────────────────────────────────

def optimize(sym: str, df: pd.DataFrame, ma_type: str, n_jobs: int) -> dict:
    """Run full IS grid search on the last IS_YEARS of data. Returns best params."""
    today_year = date.today().year
    is_start   = today_year - IS_YEARS[ma_type]
    is_mask    = df["date"].dt.year.between(is_start, today_year - 1)
    is_df      = df[is_mask].copy().reset_index(drop=True)

    if len(is_df) < 252:
        logger.error(f"{sym}: only {len(is_df)} IS bars — need at least 1 year. Aborting.")
        sys.exit(1)

    logger.info(
        f"{sym}: IS window {is_start}–{today_year - 1}  "
        f"({len(is_df)} bars)  MA type: {ma_type.upper()}"
    )

    combos      = build_combos()
    all_periods = sorted({p for c in combos for p in c[:4]})
    tc_frac     = TC_BPS / 10_000

    ma_dict = {p: compute_ma(is_df["close"], p, ma_type).to_numpy() for p in all_periods}
    close_arr = is_df["close"].to_numpy()

    n_workers = cpu_count() if n_jobs < 0 else max(1, n_jobs)
    chunk     = math.ceil(len(combos) / n_workers)
    batches   = [combos[i: i + chunk] for i in range(0, len(combos), chunk)]

    logger.info(f"{sym}: searching {len(combos):,} combos across {n_workers} workers …")
    batch_res = Parallel(n_jobs=n_jobs)(
        delayed(_run_batch)(close_arr, ma_dict, b, tc_frac) for b in batches
    )
    flat = [item for batch in batch_res for item in batch]
    best_sharpe, best = max(flat, key=lambda x: x[0])
    ef, es, xf, xs, ts = best

    logger.info(
        f"{sym}: best IS Sharpe={best_sharpe:.3f}  "
        f"entry {ma_type.upper()}({ef}/{es})  "
        f"exit {ma_type.upper()}({xf}/{xs})  trail={ts:.0%}"
    )
    return {
        "symbol":      sym,
        "ma_type":     ma_type,
        "is_sharpe":   round(best_sharpe, 4),
        "entry_fast":  ef,
        "entry_slow":  es,
        "exit_fast":   xf,
        "exit_slow":   xs,
        "trail_pct":   ts,
        "optimized_on": date.today().isoformat(),
        "is_start":    is_start,
        "is_end":      today_year - 1,
    }


# ── param cache ───────────────────────────────────────────────────────────────

def load_cache(ma_type: str, max_age_days: int) -> dict | None:
    """Return cached params if they exist, are fresh, and match ma_type."""
    if not CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(CACHE_PATH.read_text())
    except Exception:
        return None

    if cache.get("ma_type") != ma_type:
        logger.info(f"Cache is for ma_type={cache.get('ma_type')!r}, requested {ma_type!r} — reoptimizing.")
        return None

    optimized_on = date.fromisoformat(cache.get("optimized_on", "2000-01-01"))
    age = (date.today() - optimized_on).days
    if age > max_age_days:
        logger.info(f"Cache is {age} days old (limit {max_age_days}) — reoptimizing.")
        return None

    logger.info(
        f"Using cached params from {optimized_on}  (age {age}d)  "
        f"— pass --reoptimize to force refresh."
    )
    return cache


def save_cache(params_by_symbol: dict[str, dict], ma_type: str) -> None:
    payload = {"ma_type": ma_type, "optimized_on": date.today().isoformat()}
    payload.update(params_by_symbol)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2))
    logger.info(f"Params cached → {CACHE_PATH}")


# ── trade reconstruction + signal ─────────────────────────────────────────────

def _cross_up(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return i > 0 and fast[i] > slow[i] and fast[i-1] <= slow[i-1]


def _cross_dn(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return i > 0 and fast[i] < slow[i] and fast[i-1] >= slow[i-1]


def analyse(sym: str, df: pd.DataFrame, p: dict) -> dict:
    """Reconstruct trade state and return today's signal."""
    ef, es   = p["entry_fast"], p["entry_slow"]
    xf, xs   = p["exit_fast"],  p["exit_slow"]
    trail    = p["trail_pct"]
    ma_type  = p["ma_type"]

    df = df.copy().reset_index(drop=True)
    df["ma_fe"] = compute_ma(df["close"], ef, ma_type)
    df["ma_se"] = compute_ma(df["close"], es, ma_type)
    df["ma_fx"] = compute_ma(df["close"], xf, ma_type)
    df["ma_sx"] = compute_ma(df["close"], xs, ma_type)

    fe = df["ma_fe"].to_numpy()
    se = df["ma_se"].to_numpy()
    fx = df["ma_fx"].to_numpy()
    sx = df["ma_sx"].to_numpy()

    in_pos      = False
    entry_price = 0.0
    entry_date  = None
    peak        = 0.0

    for i in range(1, len(df) - 1):   # walk up to but not including today
        if np.isnan(fe[i]) or np.isnan(se[i]):
            continue
        if _cross_up(fe, se, i) and not in_pos:
            in_pos      = True
            entry_price = df["close"].iloc[i]
            entry_date  = df["date"].iloc[i]
            peak        = entry_price
        if in_pos:
            peak = max(peak, df["close"].iloc[i])
            if _cross_dn(fx, sx, i) or df["close"].iloc[i] <= peak * (1 - trail):
                in_pos      = False
                entry_price = 0.0
                entry_date  = None
                peak        = 0.0

    # Today's bar
    n      = len(df) - 1
    close  = df["close"].iloc[n]
    today  = df["date"].iloc[n]
    if hasattr(today, "date"):
        today = today.date()

    entry_now = _cross_up(fe, se, n) and not in_pos
    exit_now  = _cross_dn(fx, sx, n) and in_pos

    if in_pos:
        peak = max(peak, close)
        trail_stop = round(peak * (1 - trail), 2)
        unrealised = (close - entry_price) / entry_price
        signal = "SELL" if (exit_now or close <= trail_stop) else "HOLD"
    elif entry_now:
        signal      = "BUY"
        entry_price = close
        entry_date  = today
        peak        = close
        trail_stop  = round(close * (1 - trail), 2)
        unrealised  = 0.0
    else:
        signal     = "FLAT"
        trail_stop = None
        unrealised = None

    entry_gap = (fe[n] - se[n]) / se[n] * 100 if se[n] > 0 else 0.0

    return {
        "symbol":      sym,
        "signal":      signal,
        "date":        today,
        "close":       round(close, 2),
        "ma_type":     ma_type,
        "entry_fast":  ef,
        "entry_slow":  es,
        "exit_fast":   xf,
        "exit_slow":   xs,
        "trail_pct":   trail,
        "is_sharpe":   p["is_sharpe"],
        "entry_price": round(entry_price, 2) if entry_price else None,
        "entry_date":  str(entry_date)[:10] if entry_date else None,
        "peak":        round(peak, 2) if peak else None,
        "trail_stop":  trail_stop,
        "unrealised":  unrealised,
        "ma_fe":       round(fe[n], 4),
        "ma_se":       round(se[n], 4),
        "ma_fx":       round(fx[n], 4),
        "ma_sx":       round(sx[n], 4),
        "entry_gap":   round(entry_gap, 3),
    }


# ── display ────────────────────────────────────────────────────────────────────

def print_scan(results: list[dict]) -> None:
    order   = {"BUY": 0, "SELL": 1, "HOLD": 2, "FLAT": 3}
    results = sorted(results, key=lambda r: order[r["signal"]])
    mt      = results[0]["ma_type"].upper()
    as_of   = results[0]["date"]

    print(f"\n{'=' * 72}")
    print(f"  SPY / QQQ ADAPTIVE {mt} SCAN  —  {as_of}")
    print(f"  Parameters: Sharpe-optimized on past {IS_YEARS[results[0]['ma_type']]} years  |  TC: {TC_BPS} bps")
    print(f"{'=' * 72}")

    for r in results:
        sig  = r["signal"]
        icon = _ICONS[sig]
        mt_  = r["ma_type"].upper()

        print(f"\n{icon} {sig}  —  {r['symbol']}")
        print(f"   Close  : ${r['close']:.2f}   |   As of : {r['date']}")
        print(f"   Params : entry {mt_}({r['entry_fast']}/{r['entry_slow']})  "
              f"exit {mt_}({r['exit_fast']}/{r['exit_slow']})  "
              f"trail {r['trail_pct']:.0%}  "
              f"[IS Sharpe {r['is_sharpe']:.3f}]")

        if sig == "BUY":
            print(f"   ── ACTION ──────────────────────────────────────────────────")
            print(f"   Enter at tomorrow's open (or today's close after hours)")
            print(f"   Set {r['trail_pct']:.0%} trailing stop — initial stop : "
                  f"${r['trail_stop']:.2f}  ({r['trail_pct']:.0%} below ${r['close']:.2f})")
            print(f"   ── {mt_} VALUES ─────────────────────────────────────────────")
            print(f"   Entry {mt_}({r['entry_fast']}) {r['ma_fe']:.4f}  >  "
                  f"{mt_}({r['entry_slow']}) {r['ma_se']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")

        elif sig == "HOLD":
            cushion = (r["close"] / r["trail_stop"] - 1) * 100 if r["trail_stop"] else 0.0
            print(f"   ── OPEN TRADE ──────────────────────────────────────────────")
            print(f"   Entry  : ${r['entry_price']:.2f}  on  {r['entry_date']}")
            print(f"   Peak   : ${r['peak']:.2f}   |   Unrealised : {r['unrealised']:+.2%}")
            print(f"   Stop   : ${r['trail_stop']:.2f}  ({cushion:.1f}% cushion)")
            print(f"   ── {mt_} VALUES ─────────────────────────────────────────────")
            print(f"   Entry {mt_}({r['entry_fast']}) {r['ma_fe']:.4f}  vs  "
                  f"{mt_}({r['entry_slow']}) {r['ma_se']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")
            print(f"   Exit  {mt_}({r['exit_fast']}) {r['ma_fx']:.4f}  vs  "
                  f"{mt_}({r['exit_slow']}) {r['ma_sx']:.4f}")

        elif sig == "SELL":
            print(f"   ── ACTION ──────────────────────────────────────────────────")
            print(f"   Exit at tomorrow's open")
            print(f"   Entry was : ${r['entry_price']:.2f}  on  {r['entry_date']}")
            print(f"   Unrealised P&L at today's close : {r['unrealised']:+.2%}")

        elif sig == "FLAT":
            print(f"   No position.  Watching for {mt_}({r['entry_fast']}) "
                  f"> {mt_}({r['entry_slow']}) crossover.")
            print(f"   Entry {mt_}({r['entry_fast']}) {r['ma_fe']:.4f}  vs  "
                  f"{mt_}({r['entry_slow']}) {r['ma_se']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")

    print(f"\n{'=' * 72}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive daily MA scanner — reoptimizes params from lookback-sweep-optimal IS window (EMA/Wilder: 9yr, SMA: 14yr)"
    )
    parser.add_argument("--ma-type",    choices=MA_TYPES, default="sma")
    parser.add_argument("--reoptimize", action="store_true",
                        help="Ignore cache and rerun grid search")
    parser.add_argument("--cache-days", type=int, default=30,
                        help="Max age of cached params in days (default 30; 0 = always reoptimize)")
    parser.add_argument("--jobs",       type=int, default=-1)
    args = parser.parse_args()

    # ── param optimization (cached or fresh) ──────────────────────────────────
    cache = None if args.reoptimize else load_cache(args.ma_type, args.cache_days)

    if cache:
        params = {sym: cache[sym] for sym in SYMBOLS}
    else:
        # Fetch IS data (IS window + 1yr buffer) and optimize
        data   = fetch_history(SYMBOLS, years=IS_YEARS[args.ma_type] + 1)
        params = {}
        for sym in SYMBOLS:
            params[sym] = optimize(sym, data[sym], args.ma_type, args.jobs)
        save_cache(params, args.ma_type)

    # ── fetch recent history for trade reconstruction ──────────────────────────
    # Need enough bars to warm up the longest MA period used by either symbol.
    max_period = max(
        max(p["entry_slow"], p["exit_slow"]) for p in params.values()
    )
    warmup_days = max(max_period * 3, 120)   # generous buffer for SMA NaN warm-up
    recent = fetch_history(SYMBOLS, years=3)  # 3yr is always enough for reconstruction

    # ── signal scan ───────────────────────────────────────────────────────────
    results = [analyse(sym, recent[sym], params[sym]) for sym in SYMBOLS]
    print_scan(results)


if __name__ == "__main__":
    main()
