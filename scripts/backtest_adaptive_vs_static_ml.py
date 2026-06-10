# Author: Elliott Middleton, assisted by Claude
# Date: 2026-06-09
# Description: Head-to-head walk-forward backtest isolating ONE question —
#   does the adaptive (per-year re-optimized) EMA parameter layer in
#   ma_adaptive_scan.py actually beat the validated STATIC parameters of
#   ema_spy_qqq_scan.py, or is it just overfitting?
#
#   Both arms run the SAME annual walk-forward structure, the SAME OOS years
#   (2010–2026), and the SAME ML exit filter — only the MA parameters differ:
#     • adaptive  → per-year Sharpe-optimal params from walkforward_ema_results
#                   (the params ma_adaptive_scan would have used each year)
#     • static    → fixed IS-2000-2019 params (ema_spy_qqq_scan.py production)
#
#   The ML exit filter is held identical across arms so the comparison isolates
#   the parameter choice. Raw (no-ML) variants are also reported to show the
#   filter's marginal contribution.
#
#   `_capN` arms (added 2026-06-10): identical to the +ML arms but trail-hit
#   suppressions are limited to N per month (--cap, default 1). Tests whether
#   the 2nd+ same-month suppressions — each of which re-anchors the trail
#   another trail% lower — are net positive or just unbounded downside in
#   months the MODERATE forecast is wrong.
#
#   Convention (matches the existing EMA walk-forward): each OOS year is
#   simulated starting flat on Jan 1 with that year's params; daily returns are
#   concatenated across years for chain-linked CAGR / Sharpe / true-daily MaxDD.

import argparse
import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
R_DIR       = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/R"
WF_PARAMS   = sorted(RESULTS_DIR.glob("walkforward_ema_results_*.csv"))[-1]

ML_PATTERN = {"SPY": "sp500_moderate_results_*.csv",
              "QQQ": "ndx_moderate_m2pi_results_*.csv"}   # M2PI model (promoted 2026-06-10)
TC_FRAC    = 5 / 10_000          # 5 bps per side
TRADING    = 252

# Validated static production params (ema_spy_qqq_scan.py / CLAUDE.md).
STATIC = {
    "SPY": dict(entry_fast=12, entry_slow=16, exit_fast=8,  exit_slow=10, trail_pct=0.06),
    "QQQ": dict(entry_fast=12, entry_slow=24, exit_fast=10, exit_slow=23, trail_pct=0.05),
}
# ML exit-filter threshold — held IDENTICAL across arms to isolate the param effect.
ML_THRESHOLD = {"SPY": 0.0, "QQQ": 0.0}


# ── inputs ──────────────────────────────────────────────────────────────────────

def load_adaptive_params() -> dict:
    df = pd.read_csv(WF_PARAMS)
    out = {}
    for _, r in df.iterrows():
        out[(int(r["year"]), r["symbol"])] = dict(
            entry_fast=int(r["entry_fast"]), entry_slow=int(r["entry_slow"]),
            exit_fast=int(r["exit_fast"]),   exit_slow=int(r["exit_slow"]),
            trail_pct=float(r["trailing_stop"]))
    years = sorted({y for (y, _) in out})
    return out, years


def load_ml() -> dict:
    """pred_by_month[symbol][applies-Period] = MODERATE forecast (%), +1m shift."""
    out = {}
    for sym, pat in ML_PATTERN.items():
        files = sorted(glob.glob(str(R_DIR / pat)))
        if not files:
            out[sym] = {}
            continue
        m = pd.read_csv(files[-1], parse_dates=["Date"]).dropna(subset=["Predicted_Return"])
        out[sym] = {(d.to_period("M") + 1): float(v)
                    for d, v in zip(m["Date"], m["Predicted_Return"])}
    return out


def load_prices(start="2009-01-01") -> dict:
    raw = yf.download(["SPY", "QQQ"], start=start, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    out = {}
    for sym in ("SPY", "QQQ"):
        s = close[sym].dropna()
        out[sym] = (s.to_numpy(), list(pd.to_datetime(s.index)))
    return out


# ── simulator ───────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


def simulate_year(close, dates, p, pred_by_month, threshold, year, use_ml, cap=None):
    """Daily returns + held flags for one OOS calendar year (start flat).

    cap: max trail-hit suppressions per calendar month (None = unlimited).
    Once the cap is reached, the next trail hit exits even if the ML forecast
    says hold — by then price has fallen ~(cap+1)×trail from the month's peak
    and the market outvotes the model. MA-cross suppressions are never capped
    (they can't re-fire until the exit EMA crosses back up).
    """
    fe = _ema(close, p["entry_fast"]); se = _ema(close, p["entry_slow"])
    fx = _ema(close, p["exit_fast"]);  sx = _ema(close, p["exit_slow"])
    trail = p["trail_pct"]

    in_pos = False
    entry = peak = prev_eq = 0.0
    rets, held, ds, trades = [], [], [], 0
    suppr = capped = suppr_count = 0
    suppr_month = None

    for i in range(2, len(close)):
        if dates[i].year != year:
            continue
        sig = i - 1
        entry_cross = fe[sig] > se[sig] and fe[sig - 1] <= se[sig - 1]
        exit_cross  = fx[sig] < sx[sig] and fx[sig - 1] >= sx[sig - 1]
        dr = 0.0

        if not in_pos and entry_cross:
            in_pos, entry, peak, prev_eq = True, close[i], close[i], 0.0
            dr -= TC_FRAC
            trades += 1

        if in_pos:
            if close[i] > peak:
                peak = close[i]
            eq = (close[i] - entry) / entry
            dr += eq - prev_eq
            prev_eq = eq
            trail_hit = close[i] <= peak * (1 - trail)
            if trail_hit or exit_cross:
                pr = pred_by_month.get(dates[i].to_period("M")) if use_ml else None
                ml_says_hold = use_ml and pr is not None and pr > threshold
                if ml_says_hold and trail_hit:
                    m = dates[i].to_period("M")
                    if m != suppr_month:
                        suppr_month, suppr_count = m, 0
                    if cap is not None and suppr_count >= cap:
                        ml_says_hold = False    # cap reached — let the exit fire
                        capped += 1
                if ml_says_hold:
                    if trail_hit:           # suppress exit, reset trail anchor
                        peak = close[i]
                        suppr += 1
                        suppr_count += 1
                else:
                    dr -= TC_FRAC
                    in_pos, prev_eq = False, 0.0

        rets.append(dr); held.append(in_pos); ds.append(dates[i])

    idx = pd.to_datetime(ds)
    return pd.Series(rets, index=idx), pd.Series(held, index=idx), trades, suppr, capped


def run_arm(close, dates, params_for_year, pred_by_month, threshold, years, use_ml, cap=None):
    rets, held, trades, suppr, capped = [], [], 0, 0, 0
    for y in years:
        p = params_for_year(y)
        r, h, t, s, c = simulate_year(close, dates, p, pred_by_month, threshold, y, use_ml, cap)
        rets.append(r); held.append(h); trades += t; suppr += s; capped += c
    return pd.concat(rets), pd.concat(held), trades, suppr, capped


# ── metrics ───────────────────────────────────────────────────────────────────

def stats(rets: pd.Series, held: pd.Series = None, trades: int = 0) -> dict:
    r = rets.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / TRADING
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if len(r) and eq.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING)
    sharpe = (r.mean() * TRADING) / vol if vol > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min() if len(r) else np.nan
    exp = float(held.mean()) if held is not None else np.nan
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=maxdd,
                exposure=exp, trades=trades, n=len(r))


def _f(v, pct=True):
    return "—" if pd.isna(v) else (f"{v*100:+.1f}%" if pct else f"{v:.2f}")


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spy-threshold", type=float, default=None,
                    help="Override SPY ML threshold for both arms (default 0.0)")
    ap.add_argument("--start-year", type=int, default=None,
                    help="First OOS year (default = first adaptive-param year, 2010). "
                         "Earlier years have no cached adaptive params, so the run "
                         "becomes STATIC-ONLY (static+ML vs raw vs B&H, full-cycle).")
    ap.add_argument("--cap", type=int, default=1,
                    help="Max trail-hit suppressions per month for the capped arms "
                         "(default 1). MA-cross suppressions are never capped.")
    args = ap.parse_args()
    if args.spy_threshold is not None:
        ML_THRESHOLD["SPY"] = args.spy_threshold

    adaptive, adaptive_years = load_adaptive_params()
    last_year   = max(adaptive_years)
    start_year  = args.start_year or min(adaptive_years)
    years       = list(range(start_year, last_year + 1))
    static_only = start_year < min(adaptive_years)

    mode = "STATIC-ONLY (full-cycle)" if static_only else "adaptive vs static"
    print(f"\nWalk-forward {mode} (+ML) — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"OOS years {start_year}–{last_year}  |  SPY thr {ML_THRESHOLD['SPY']:+.1f}%  "
          f"QQQ thr {ML_THRESHOLD['QQQ']:+.1f}%")
    print(f"Params source: {WF_PARAMS.name}")
    if static_only:
        print("Note: pre-2010 has no cached adaptive params — adaptive arms skipped.\n")
    else:
        print()

    ml = load_ml()
    prices = load_prices(start=f"{start_year - 1}-01-01")

    all_rows, yearly_rows, equity = [], [], {}
    for sym in ("SPY", "QQQ"):
        close, dates = prices[sym]
        thr = ML_THRESHOLD[sym]
        pbm = ml.get(sym, {})

        capname = f"cap{args.cap}"
        arms = {
            "static+ML":            (lambda y, s=sym: STATIC[s],        True,  None),
            f"static+ML_{capname}": (lambda y, s=sym: STATIC[s],        True,  args.cap),
            "static_raw":           (lambda y, s=sym: STATIC[s],        False, None),
        }
        if not static_only:
            arms["adaptive+ML"]            = (lambda y, s=sym: adaptive[(y, s)], True,  None)
            arms[f"adaptive+ML_{capname}"] = (lambda y, s=sym: adaptive[(y, s)], True,  args.cap)
            arms["adaptive_raw"]           = (lambda y, s=sym: adaptive[(y, s)], False, None)
        arm_rets = {}
        for name, (pf, use_ml, cap) in arms.items():
            r, h, t, suppr, capped = run_arm(close, dates, pf, pbm, thr, years, use_ml, cap)
            arm_rets[name] = r
            st = stats(r, h, t)
            all_rows.append(dict(symbol=sym, arm=name, **st, suppr=suppr, capped=capped))

        # buy & hold over the same span
        span = arm_rets["static+ML"].index
        bh = pd.Series(close, index=pd.to_datetime(dates)).pct_change().reindex(span).fillna(0)
        all_rows.append(dict(symbol=sym, arm="buy_hold", **stats(bh), suppr=0, capped=0))
        equity[sym] = dict(static=arm_rets["static+ML"], bh=bh)
        if not static_only:
            equity[sym]["adaptive"] = arm_rets["adaptive+ML"]

        # year-by-year (the headline series)
        for y in years:
            yr = lambda s: (1 + s[s.index.year == y]).prod() - 1
            row = dict(symbol=sym, year=y, static_ml=yr(arm_rets["static+ML"]),
                       static_ml_cap=yr(arm_rets[f"static+ML_{capname}"]), bh=yr(bh))
            row["adaptive_ml"] = (np.nan if static_only
                                  else yr(arm_rets["adaptive+ML"]))
            row["adaptive_ml_cap"] = (np.nan if static_only
                                      else yr(arm_rets[f"adaptive+ML_{capname}"]))
            yearly_rows.append(row)

    res = pd.DataFrame(all_rows)
    yb = pd.DataFrame(yearly_rows)

    # ── console ──────────────────────────────────────────────────────────────────
    show = res.copy()
    for c in ("cagr", "vol", "maxdd", "exposure"):
        show[c] = (show[c] * 100).round(1)
    show["sharpe"] = show["sharpe"].round(2)
    print(show[["symbol", "arm", "cagr", "vol", "sharpe", "maxdd",
                "exposure", "trades", "suppr", "capped", "n"]].to_string(index=False))

    for sym in ("SPY", "QQQ"):
        sub = yb[yb["symbol"] == sym]
        print(f"\nYear-by-year {sym} (adaptive+ML | adaptive cap | static+ML | static cap | B&H):")
        s = sub.copy()
        for c in ("adaptive_ml", "adaptive_ml_cap", "static_ml", "static_ml_cap", "bh"):
            s[c] = (s[c] * 100).round(1)
        print(s[["year", "adaptive_ml", "adaptive_ml_cap",
                 "static_ml", "static_ml_cap", "bh"]].to_string(index=False))

    # ── save (suffix encodes config so runs don't clobber each other) ───────────
    stamp = datetime.now().strftime("%Y-%m-%d")
    suffix = f"_from{start_year}" if static_only else ""
    if ML_THRESHOLD["SPY"] != 0.0:
        suffix += f"_spythr{ML_THRESHOLD['SPY']:+.1f}"
    base = f"adaptive_vs_static_ml_cap{args.cap}{suffix}_{stamp}"
    res.to_csv(RESULTS_DIR / f"{base}.csv", index=False)
    yb.to_csv(RESULTS_DIR / f"{base}_yearly.csv", index=False)
    (RESULTS_DIR / f"{base}.md").write_text(_markdown(res, yb, years, args.cap))
    print(f"\nSaved → results/{base}.(csv|md)")


def _markdown(res: pd.DataFrame, yb: pd.DataFrame, years: list, cap: int) -> str:
    capname = f"cap{cap}"
    L = [
        "# Adaptive vs Static EMA params (+ ML exit filter) — Walk-Forward",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        f"OOS years **{years[0]}–{years[-1]}**. Both arms: identical annual walk-forward, "
        "identical ML exit filter (threshold held constant); only the EMA params differ. "
        "Per-year adaptive params from `walkforward_ema_results`; static = ema_spy_qqq_scan production. "
        f"`_{capname}` arms limit trail-hit exit suppressions to {cap}/month — once the cap is hit, "
        "the next trail exit fires even on a bullish forecast (MA-cross suppressions never capped). "
        "Each OOS year starts flat (established EMA WF convention). TC 5 bps/side.",
        "",
    ]
    for sym in ("SPY", "QQQ"):
        sub = res[res["symbol"] == sym].set_index("arm")
        L += [f"## {sym}", "",
              "| Arm | CAGR | Sharpe | MaxDD | Exposure | Trades | Suppr | Capped |",
              "|-----|-----:|-------:|------:|---------:|-------:|------:|-------:|"]
        for arm in ["buy_hold", "static+ML", f"static+ML_{capname}",
                    "adaptive+ML", f"adaptive+ML_{capname}",
                    "static_raw", "adaptive_raw"]:
            if arm not in sub.index:
                continue
            r = sub.loc[arm]
            L.append(f"| {arm} | {_f(r['cagr'])} | {_f(r['sharpe'], pct=False)} | "
                     f"{_f(r['maxdd'])} | {_f(r['exposure'])} | {int(r['trades'])} | "
                     f"{int(r['suppr'])} | {int(r['capped'])} |")
        L.append("")
        ys = yb[yb["symbol"] == sym]
        L += [f"| Year | Adaptive+ML | Adaptive {capname} | Static+ML | Static {capname} | B&H |",
              "|-----:|------------:|----------:|----------:|----------:|----:|"]
        for _, r in ys.iterrows():
            L.append(f"| {int(r['year'])} | {_f(r['adaptive_ml'])} | {_f(r['adaptive_ml_cap'])} | "
                     f"{_f(r['static_ml'])} | {_f(r['static_ml_cap'])} | {_f(r['bh'])} |")
        comp = lambda c: (1 + ys[c]).prod() - 1
        L += [f"| **Total** | **{comp('adaptive_ml')*100:+.0f}%** | **{comp('adaptive_ml_cap')*100:+.0f}%** | "
              f"**{comp('static_ml')*100:+.0f}%** | **{comp('static_ml_cap')*100:+.0f}%** | "
              f"**{comp('bh')*100:+.0f}%** |", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
