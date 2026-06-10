#!/usr/bin/env python3
"""
Adaptive daily MA signal scanner — SPY and QQQ.

Finds Sharpe-maximizing entry/exit MA parameters by running the same IS grid
search used in walkforward_ma_optimization.py on a lookback-sweep-optimal IS
window (EMA/Wilder: 9yr / 108m; SMA: 14yr / 168m), then applies those params
to reconstruct the current trade state and report today's signal.

Moving average type is selectable: ema (default), sma, or wilder.

ML exit filter (EXIT-ONLY, same rule as ema_spy_qqq_scan.py): when an MA-cross
or trailing-stop EXIT fires, it is suppressed if the SP500/NDX MODERATE monthly
regressor forecasts a return above ML_THRESHOLD (SPY +0.5%, QQQ 0.0%). Entries
are never gated. Each MODERATE row is month-end data forecasting M+1…M+3, applied
to trading days in M+1 (+1-month shift — no look-ahead). The filter degrades
gracefully to OFF when no current forecast file is found.

Because the grid search takes ~3–6 minutes, optimized parameters are cached
to data/models/ma_adaptive_params.json. On subsequent runs the cache is reused
unless --reoptimize is passed or the cache is more than --cache-days old
(default 30 days — params are stable month to month).

Usage
-----
    python3 ma_adaptive_scan.py                    # EMA (default), use/refresh cache
    python3 ma_adaptive_scan.py --ma-type sma      # SMA
    python3 ma_adaptive_scan.py --ma-type wilder   # Wilder smoothing
    python3 ma_adaptive_scan.py --reoptimize       # force fresh grid search
    python3 ma_adaptive_scan.py --cache-days 0     # always reoptimize
    python3 ma_adaptive_scan.py --no-ml-filter     # raw signals (filter off)
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

# ── ML exit-filter configuration (mirrors ema_spy_qqq_scan.py) ────────────────
R_DIR      = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/R"
ML_PATTERN = {
    "SPY": "sp500_moderate_results_*.csv",
    "QQQ": "ndx_moderate_m2pi_results_*.csv",   # M2PI model (promoted 2026-06-10)
}
# Suppress an exit if the MODERATE predicted return exceeds this threshold.
# SPY +0.5%: measured better SPY MaxDD (−13.7% vs −19.5%, 2010–26) at the same
# Sharpe (1.15) — see results/adaptive_vs_static_ml_*spythr+0.5*. QQQ left at 0.0%.
ML_THRESHOLD = {"SPY": 0.5, "QQQ": 0.0}

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


# ── ML prediction loader (mirrors ema_spy_qqq_scan.py) ────────────────────────

def load_ml_predictions() -> dict[str, dict | None]:
    """Return the MODERATE predicted-return history for each symbol.

    Each CSV row is **month-end** data: a row dated month M is computed from M's
    close and forecasts the change over M+1…M+3. The forecast is only knowable
    after M closes, so it is applied to trading days in M+1 (+1-month shift).

    Result keys per symbol:
      pred          – latest forecast value (float %, most recent row)
      data_month    – vintage of that forecast (the row's own month, str)
      applies_month – first month the forecast applies to (data_month + 1, str)
      pred_by_month – {applies-period → pred} over full history, for replaying
                      the exit filter during trade reconstruction
      is_active     – True if the latest forecast still applies this month
    Returns None for a symbol when no file or no valid prediction row is found.
    """
    today         = pd.Timestamp.today()
    current_month = today.to_period("M")
    out: dict[str, dict | None] = {}
    for sym, pattern in ML_PATTERN.items():
        files = sorted(R_DIR.glob(pattern))
        if not files:
            out[sym] = None
            continue
        df = pd.read_csv(files[-1], parse_dates=["Date"])
        valid = df.dropna(subset=["Predicted_Return"])
        if valid.empty:
            out[sym] = None
            continue
        # Shift +1 month: the month-M forecast applies to trading month M+1.
        pred_by_month = {
            (d.to_period("M") + 1): float(r)
            for d, r in zip(valid["Date"], valid["Predicted_Return"])
        }
        data_month    = valid["Date"].iloc[-1].to_period("M")
        applies_month = data_month + 1
        pred          = float(valid["Predicted_Return"].iloc[-1])
        # Prefer the forecast that applies to the *current* month: a mid-month
        # model rerun appends a partial-month row whose forecast applies to
        # NEXT month — taking the last row blindly would apply it a month early.
        if current_month in pred_by_month and applies_month > current_month:
            pred          = pred_by_month[current_month]
            data_month    = current_month - 1
            applies_month = current_month
        out[sym] = {
            "pred":          pred,
            "data_month":    str(data_month),
            "applies_month": str(applies_month),
            "pred_by_month": pred_by_month,
            "is_active":     (current_month - applies_month).n <= 1,
        }
    return out


def _ml_status_line(r: dict) -> str:
    """One-line ML filter summary for the scan header."""
    if r.get("ml_disabled"):
        return "ML filter: OFF (disabled via --no-ml-filter — raw signals)"
    if not r.get("ml_filter_on"):
        if r.get("ml_month"):
            return f"ML filter: OFF ({r['ml_month']} forecast is stale — re-run MODERATE model)"
        return "ML filter: OFF (no forecast file found)"
    sign = "+" if r["ml_pred"] >= 0 else ""
    return (f"ML filter: ON  |  {r['ml_month']} month-end forecast {sign}{r['ml_pred']:.2f}%  "
            f"→ applies {r['ml_applies']}+  (threshold {r['ml_threshold']:+.1f}%)")


# ── trade reconstruction + signal ─────────────────────────────────────────────

def _cross_up(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return i > 0 and fast[i] > slow[i] and fast[i-1] <= slow[i-1]


def _cross_dn(fast: np.ndarray, slow: np.ndarray, i: int) -> bool:
    return i > 0 and fast[i] < slow[i] and fast[i-1] >= slow[i-1]


def analyse(sym: str, df: pd.DataFrame, p: dict, ml_info: dict | None = None,
            ml_enabled: bool = True) -> dict:
    """Reconstruct trade state (replaying the ML exit filter) and return today's signal."""
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

    # ── ML exit-filter state ──────────────────────────────────────────────────
    ml_pred       = ml_info["pred"] if ml_info else None
    ml_threshold  = ML_THRESHOLD[sym]
    ml_filter_on  = (ml_info is not None and ml_info.get("is_active", False))
    pred_by_month = ml_info.get("pred_by_month") if ml_info else None

    in_pos      = False
    entry_price = 0.0
    entry_date  = None
    peak        = 0.0

    # Reconstruct through the *prior* bar, replaying the ML exit filter so the
    # reported open trade matches the strategy actually run. Today's bar is
    # evaluated separately below so a live BUY/SELL can be emitted.
    for i in range(1, len(df) - 1):
        if np.isnan(fe[i]) or np.isnan(se[i]):
            continue
        close_i = df["close"].iloc[i]
        if _cross_up(fe, se, i) and not in_pos:
            in_pos      = True
            entry_price = close_i
            entry_date  = df["date"].iloc[i]
            peak        = close_i
        if in_pos:
            peak = max(peak, close_i)
            trail_hit = close_i <= peak * (1 - trail)
            if _cross_dn(fx, sx, i) or trail_hit:
                bar_month = df["date"].iloc[i].to_period("M")
                pr = pred_by_month.get(bar_month) if pred_by_month else None
                if pr is not None and pr > ml_threshold:
                    # Exit suppressed — stay long; reset the trail anchor so a hit
                    # stop doesn't re-fire every bar (mirrors the backtest).
                    if trail_hit:
                        peak = close_i
                else:
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

    ml_suppressed = False
    exit_reason   = None

    if in_pos:
        peak = max(peak, close)
        trail_stop = round(peak * (1 - trail), 2)
        unrealised = (close - entry_price) / entry_price
        trail_hit      = close <= trail_stop
        exit_triggered = exit_now or trail_hit
        exit_reason    = ("trail" if trail_hit else "ma") if exit_triggered else None
        if exit_triggered and ml_filter_on and ml_pred is not None and ml_pred > ml_threshold:
            signal        = "HOLD"
            ml_suppressed = True
            if trail_hit:
                # Mirror the replay's anchor reset: the effective stop going
                # forward re-anchors at today's close, so report that level
                # rather than the breached (stale) one.
                peak       = close
                trail_stop = round(close * (1 - trail), 2)
        elif exit_triggered:
            signal = "SELL"
        else:
            signal = "HOLD"
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
        "exit_reason": exit_reason,
        # ML filter
        "ml_pred":       ml_pred,
        "ml_month":      ml_info["data_month"] if ml_info else None,
        "ml_applies":    ml_info["applies_month"] if ml_info else None,
        "ml_suppressed": ml_suppressed,
        "ml_threshold":  ml_threshold,
        "ml_filter_on":  ml_filter_on,
        "ml_disabled":   not ml_enabled,
        # MA values for diagnostics
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
        sig   = r["signal"]
        icon  = "🟡" if r.get("ml_suppressed") else _ICONS[sig]
        mt_   = r["ma_type"].upper()
        label = f"{sig} (exit suppressed by ML filter)" if r.get("ml_suppressed") else sig

        print(f"\n{icon} {label}  —  {r['symbol']}")
        print(f"   Close  : ${r['close']:.2f}   |   As of : {r['date']}")
        print(f"   {_ml_status_line(r)}")
        print(f"   Params : entry {mt_}({r['entry_fast']}/{r['entry_slow']})  "
              f"exit {mt_}({r['exit_fast']}/{r['exit_slow']})  "
              f"trail {r['trail_pct']:.0%}  "
              f"[IS Sharpe {r['is_sharpe']:.3f}]")

        if sig == "BUY":
            print(f"   ── ACTION ────────────────────────────────────")
            print(f"   Enter at tomorrow's open (or today's close after hours)")
            print(f"   Set {r['trail_pct']:.0%} trailing stop — initial stop : "
                  f"${r['trail_stop']:.2f}  ({r['trail_pct']:.0%} below ${r['close']:.2f})")
            print(f"   ── {mt_} VALUES ──────────────────────────────")
            print(f"   Entry {mt_}({r['entry_fast']}) {r['ma_fe']:.4f}  >  "
                  f"{mt_}({r['entry_slow']}) {r['ma_se']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")

        elif sig == "HOLD":
            cushion = (r["close"] / r["trail_stop"] - 1) * 100 if r["trail_stop"] else 0.0
            if r.get("ml_suppressed"):
                reason = r["exit_reason"].upper() if r["exit_reason"] else "SIGNAL"
                sign   = "+" if r["ml_pred"] >= 0 else ""
                print(f"   ── EXIT SUPPRESSED ──────────────────────────-")
                print(f"   {reason} exit fired — overridden: pred {sign}{r['ml_pred']:.2f}%  "
                      f">  threshold {r['ml_threshold']:+.1f}%  → staying long")
                if r["exit_reason"] == "trail":
                    print(f"   Trail re-anchored at today's close — stop below is the new effective level")
            print(f"   ── OPEN TRADE ───────────────────────────────-")
            print(f"   Entry  : ${r['entry_price']:.2f}  on  {r['entry_date']}")
            print(f"   Peak   : ${r['peak']:.2f}   |   Unrealised : {r['unrealised']:+.2%}")
            print(f"   Stop   : ${r['trail_stop']:.2f}  ({cushion:.1f}% cushion)")
            print(f"   ── {mt_} VALUES ──────────────────────────────")
            print(f"   Entry {mt_}({r['entry_fast']}) {r['ma_fe']:.4f}  vs  "
                  f"{mt_}({r['entry_slow']}) {r['ma_se']:.4f}  "
                  f"(gap {r['entry_gap']:+.3f}%)")
            print(f"   Exit  {mt_}({r['exit_fast']}) {r['ma_fx']:.4f}  vs  "
                  f"{mt_}({r['exit_slow']}) {r['ma_sx']:.4f}")

        elif sig == "SELL":
            print(f"   ── ACTION ───────────────────────────────────")
            print(f"   Exit at tomorrow's open")
            print(f"   Trigger : {'trailing stop' if r['exit_reason'] == 'trail' else mt_ + ' cross-down'}"
                  f"  (stop ${r['trail_stop']:.2f})")
            print(f"   Entry was : ${r['entry_price']:.2f}  on  {r['entry_date']}")
            print(f"   Unrealised P&L at today's close : {r['unrealised']:+.2%}")
            if r["ml_filter_on"] and r["ml_pred"] is not None:
                sign = "+" if r["ml_pred"] >= 0 else ""
                print(f"   ML filter did not suppress: pred {sign}{r['ml_pred']:.2f}%  "
                      f"≤  threshold {r['ml_threshold']:+.1f}%")
            elif not r["ml_filter_on"]:
                print(f"   ML filter inactive (no recent forecast) — exit proceeds")

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
    parser.add_argument("--ma-type",    choices=MA_TYPES, default="ema")
    parser.add_argument("--reoptimize", action="store_true",
                        help="Ignore cache and rerun grid search")
    parser.add_argument("--cache-days", type=int, default=30,
                        help="Max age of cached params in days (default 30; 0 = always reoptimize)")
    parser.add_argument("--no-ml-filter", action="store_true",
                        help="Disable the ML exit filter — show raw MA-crossover signals")
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

    # ── ML exit-filter forecasts ───────────────────────────────────────────────
    if args.no_ml_filter:
        logger.info("ML exit filter disabled (--no-ml-filter) — showing raw MA signals")
        ml_preds = {sym: None for sym in SYMBOLS}
    else:
        logger.info("Loading ML predictions …")
        ml_preds = load_ml_predictions()
        for sym in SYMBOLS:
            info = ml_preds.get(sym)
            if info is None:
                logger.info(f"  {sym}: no forecast file found — ML filter inactive")
            elif not info["is_active"]:
                logger.info(f"  {sym}: {info['data_month']} forecast no longer applies — ML filter inactive")
            else:
                sign = "+" if info["pred"] >= 0 else ""
                logger.info(f"  {sym}: {info['data_month']} month-end forecast {sign}{info['pred']:.2f}% "
                            f"(applies {info['applies_month']}+) vs threshold {ML_THRESHOLD[sym]:+.1f}%")

    # ── signal scan ───────────────────────────────────────────────────────────
    results = [
        analyse(sym, recent[sym], params[sym], ml_preds.get(sym),
                ml_enabled=not args.no_ml_filter)
        for sym in SYMBOLS
    ]
    print_scan(results)


if __name__ == "__main__":
    main()
