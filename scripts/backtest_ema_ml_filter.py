# Author: Elliott Middleton, assisted by Claude
# Date: 2026-06-06
# Description: Backtest EMA crossover strategy with MODERATE ML prediction gate.
#   Tests three compound rule variants:
#     exit_filter  - suppress EMA/trail exit if ML predicted return > threshold
#     entry_gate   - only enter on EMA BUY if ML predicted return > threshold
#     combined     - both filters active at the same threshold
#   Sweeps thresholds from -3% to +5% (step 0.5%) and reports CAGR/Sharpe/MaxDD.

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

BASE    = Path(__file__).parent.parent
R_DIR   = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/R"
RESULTS = BASE / "results"
MODELS  = BASE / "data" / "models"

OOS_START = "2010-01-01"
TODAY_STR = date.today().strftime("%Y-%m-%d")

# Static EMA parameters (IS-optimized on 2000-2019, validated on 2020-2026)
PARAMS = {
    "SPY": {"ef": 12, "es": 16, "xf": 8,  "xs": 10, "trail": 0.06},
    "QQQ": {"ef": 12, "es": 24, "xf": 10, "xs": 23, "trail": 0.05},
}

ML_PATTERN = {
    "SPY": "sp500_moderate_results_*.csv",
    "QQQ": "ndx_moderate_results_*.csv",
}

EQ_PATHS = {
    "SPY": MODELS / "ema_spy_equity_curve.csv",
    "QQQ": MODELS / "ema_qqq_equity_curve.csv",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ml_predictions(sym):
    pattern = ML_PATTERN[sym]
    files = sorted(R_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No ML predictions matching {pattern} in {R_DIR}")
    path = files[-1]
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"  ML [{sym}] {path.name}: {len(df)} rows, "
          f"{df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
    return df[["Date", "Predicted_Return", "Reg_Direction", "Probability_Up"]].copy()


def load_prices_full(sym):
    """Full price history (back to 2000) needed for IS window EMA initialization."""
    df = pd.read_csv(EQ_PATHS[sym], parse_dates=["date"])
    return df.dropna(subset=["close", "bh_equity"]).reset_index(drop=True)


# ── EMA helpers ───────────────────────────────────────────────────────────────

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean().values


# ── Core simulation ───────────────────────────────────────────────────────────

def simulate_period(closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
                    pred_dict, params, period_start, period_end,
                    exit_thresh=None, entry_thresh=None):
    """
    Simulate for a specific date range using pre-computed global EMA arrays.
    Position state starts fresh (in_pos=False) at period_start — same convention
    as walkforward_ema_optimization.py.  EMA values are properly initialised
    because they are computed on the full price history before being passed in.

    Returns: (daily_rets, n_trades, n_suppressed, n_blocked, max_suppressed_dd).
    max_suppressed_dd (<=0) is the worst drawdown measured from the peak at the
    moment an exit was first suppressed until the position is finally closed — it
    quantifies the downside risk of holding through a suppressed stop (HIGH-3).
    """
    start_ts = np.datetime64(pd.Timestamp(period_start))
    end_ts   = np.datetime64(pd.Timestamp(period_end))
    idxs = np.where((dates >= start_ts) & (dates <= end_ts))[0]

    if len(idxs) < 2:
        return np.array([]), 0, 0, 0, 0.0

    in_pos = False
    peak   = 0.0
    daily_rets = []
    n_trades = n_sup = n_blk = 0
    supp_active = False    # currently holding past a suppressed exit
    supp_peak   = 0.0      # peak at the moment suppression began
    max_supp_dd = 0.0

    for k in range(1, len(idxs)):
        i      = idxs[k]
        i_prev = idxs[k - 1]

        c_cur  = closes[i]
        c_prev = closes[i_prev]
        ym     = pd.Timestamp(dates[i]).to_period("M")
        pred_ret = pred_dict.get(ym, None)

        if in_pos:
            if c_cur > peak:
                peak = c_cur
            trail_exit = c_cur < peak * (1.0 - params["trail"])
            ema_exit   = (ema_xf[i] < ema_xs[i]) and (ema_xf[i_prev] >= ema_xs[i_prev])

            if trail_exit or ema_exit:
                suppress = (exit_thresh is not None and pred_ret is not None
                            and pred_ret > exit_thresh)
                if suppress:
                    n_sup += 1
                    if not supp_active:
                        supp_active = True
                        supp_peak   = peak   # capture before any trail reset
                    daily_rets.append(c_cur / c_prev - 1.0)
                    if trail_exit:
                        peak = c_cur  # reset trail from here
                else:
                    daily_rets.append(c_cur / c_prev - 1.0)
                    in_pos = False
                    supp_active = False
                    n_trades += 1
            else:
                daily_rets.append(c_cur / c_prev - 1.0)

            # Track worst excursion while holding through a suppressed exit.
            if supp_active and in_pos:
                dd = c_cur / supp_peak - 1.0
                if dd < max_supp_dd:
                    max_supp_dd = dd
        else:
            entry_signal = (ema_ef[i] > ema_es[i]) and (ema_ef[i_prev] <= ema_es[i_prev])
            allow_entry = True
            if entry_thresh is not None and pred_ret is not None:
                allow_entry = pred_ret > entry_thresh

            if entry_signal and allow_entry:
                in_pos = True
                peak = c_cur
                n_trades += 1
            elif entry_signal and not allow_entry:
                n_blk += 1
            daily_rets.append(0.0)

    return np.array(daily_rets), n_trades, n_sup, n_blk, max_supp_dd


# ── Performance metrics ───────────────────────────────────────────────────────

def _cagr(rets, n_years):
    eq = float(np.prod(1.0 + rets))
    return eq ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan


def _max_dd(rets):
    eq = np.cumprod(1.0 + rets)
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def _sharpe(rets):
    std = rets.std()
    return float(rets.mean() / std * np.sqrt(252)) if std > 1e-10 else np.nan


# ── Sweep ─────────────────────────────────────────────────────────────────────

THRESHOLDS = list(np.round(np.arange(-3.0, 5.5, 0.5), 2))

VARIANTS = [
    ("exit_filter", lambda t: {"exit_thresh": t}),
    ("entry_gate",  lambda t: {"entry_thresh": t}),
    ("combined",    lambda t: {"exit_thresh": t, "entry_thresh": t}),
]


def run_all(sym, prices_df_full, pred_df):
    params = PARAMS[sym]

    # Shifted monthly prediction lookup (HIGH-2): a row dated month M holds
    # month-M month-end data, so its forecast is only knowable ~M+1. Apply it to
    # trading days in M+1 (matches the live scanner) to avoid 1-month look-ahead.
    p = pred_df.copy()
    p["ym"] = p["Date"].dt.to_period("M") + 1
    pred_dict = dict(zip(p["ym"], p["Predicted_Return"]))

    # EMAs on full history so the OOS start is properly seeded (MED-5: no warm-up
    # bias), then simulate only over the OOS span.
    closes = prices_df_full["close"].values
    dates  = prices_df_full["date"].values
    ema_ef = _ema(prices_df_full["close"], params["ef"])
    ema_es = _ema(prices_df_full["close"], params["es"])
    ema_xf = _ema(prices_df_full["close"], params["xf"])
    ema_xs = _ema(prices_df_full["close"], params["xs"])

    oos_start = pd.Timestamp(OOS_START)
    oos_end   = prices_df_full["date"].max()
    oos_mask  = (prices_df_full["date"] >= oos_start) & (prices_df_full["date"] <= oos_end)
    n_years   = (int(oos_mask.sum()) - 1) / 252.0

    rows = []

    # BH (OOS slice)
    bh_rets = prices_df_full.loc[oos_mask, "bh_equity"].pct_change().fillna(0.0).values[1:]
    bh = {
        "symbol": sym, "variant": "bh", "threshold": np.nan,
        "cagr": _cagr(bh_rets, n_years),
        "sharpe": _sharpe(bh_rets),
        "max_dd": _max_dd(bh_rets),
        "n_trades": np.nan, "n_suppressed": 0, "n_blocked": 0, "max_supp_dd": np.nan,
    }

    # Baseline EMA (no filter)
    br, bt, _, _, _ = simulate_period(
        closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
        pred_dict, params, oos_start, oos_end
    )
    baseline = {
        "symbol": sym, "variant": "baseline", "threshold": np.nan,
        "cagr": _cagr(br, n_years),
        "sharpe": _sharpe(br),
        "max_dd": _max_dd(br),
        "n_trades": bt, "n_suppressed": 0, "n_blocked": 0, "max_supp_dd": np.nan,
    }
    rows.extend([bh, baseline])

    for thresh in THRESHOLDS:
        for name, kw_fn in VARIANTS:
            rets, nt, ns, nb, msdd = simulate_period(
                closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
                pred_dict, params, oos_start, oos_end, **kw_fn(thresh)
            )
            rows.append({
                "symbol": sym, "variant": name, "threshold": thresh,
                "cagr": _cagr(rets, n_years),
                "sharpe": _sharpe(rets),
                "max_dd": _max_dd(rets),
                "n_trades": nt, "n_suppressed": ns, "n_blocked": nb,
                "max_supp_dd": msdd,
            })

    return rows


# ── Walk-forward threshold validation ────────────────────────────────────────
# Use the same 9-year IS window that was validated as optimal for EMA parameters.

IS_YEARS_WF = 9


def _max_dd_annual(annual_rets):
    """Max drawdown from a list of annual holding-period returns (chain-linked)."""
    eq = np.cumprod(1.0 + np.array(annual_rets, dtype=float))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def run_walkforward_threshold(sym, prices_df_full, pred_df):
    """
    For each OOS year 2010–2026:
      1. Select IS-optimal threshold (max Sharpe) on the prior IS_YEARS_WF years.
      2. Apply that threshold to the OOS year.
    Returns a list of per-year dicts, one row per (year × variant).
    """
    params = PARAMS[sym]

    # Shifted monthly prediction lookup (HIGH-2): row dated M holds month-M
    # month-end data, knowable only ~M+1, so apply it to trading days in M+1.
    p = pred_df.copy()
    p["ym"] = p["Date"].dt.to_period("M") + 1
    pred_dict = dict(zip(p["ym"], p["Predicted_Return"]))

    # Pre-compute EMAs on full history once — all period slices share these arrays.
    closes = prices_df_full["close"].values
    dates  = prices_df_full["date"].values
    ema_ef = _ema(prices_df_full["close"], params["ef"])
    ema_es = _ema(prices_df_full["close"], params["es"])
    ema_xf = _ema(prices_df_full["close"], params["xf"])
    ema_xs = _ema(prices_df_full["close"], params["xs"])

    max_date = prices_df_full["date"].max()
    rows = []
    # Concatenated OOS daily returns per series → true daily-chained MaxDD (MED-6).
    daily_by_variant = {v: [] for v, _ in VARIANTS}
    daily_by_variant["bh"]       = []
    daily_by_variant["baseline"] = []

    for oos_year in range(2010, 2027):
        is_start  = pd.Timestamp(f"{oos_year - IS_YEARS_WF}-01-01")
        is_end    = pd.Timestamp(f"{oos_year - 1}-12-31")
        oos_start = pd.Timestamp(f"{oos_year}-01-01")
        oos_end   = min(pd.Timestamp(f"{oos_year}-12-31"), max_date)

        if oos_start > max_date:
            break

        # BH and baseline for this OOS year
        oos_mask = (prices_df_full["date"] >= oos_start) & (prices_df_full["date"] <= oos_end)
        oos_bh   = prices_df_full.loc[oos_mask, "bh_equity"].pct_change().fillna(0.0).values[1:]
        bh_ret   = float(np.prod(1.0 + oos_bh) - 1.0) if len(oos_bh) else np.nan

        bl_rets, bl_t, _, _, _ = simulate_period(
            closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
            pred_dict, params, oos_start, oos_end
        )
        bl_ret = float(np.prod(1.0 + bl_rets) - 1.0) if len(bl_rets) else np.nan
        daily_by_variant["bh"].append(oos_bh)
        daily_by_variant["baseline"].append(bl_rets)

        for vname, kw_fn in VARIANTS:
            # ── IS: find threshold that maximises Sharpe ──
            best_thresh    = THRESHOLDS[len(THRESHOLDS) // 2]  # fallback: middle value
            best_is_sharpe = -np.inf

            for thresh in THRESHOLDS:
                rets, _, _, _, _ = simulate_period(
                    closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
                    pred_dict, params, is_start, is_end, **kw_fn(thresh)
                )
                if len(rets) < 50:
                    continue
                s = _sharpe(rets)
                if not np.isnan(s) and s > best_is_sharpe:
                    best_is_sharpe = s
                    best_thresh = thresh

            # ── OOS: apply selected threshold ──
            oos_rets, n_t, n_s, n_b, n_msdd = simulate_period(
                closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
                pred_dict, params, oos_start, oos_end, **kw_fn(best_thresh)
            )
            daily_by_variant[vname].append(oos_rets)

            rows.append({
                "year":      oos_year,
                "symbol":    sym,
                "variant":   vname,
                "is_thresh": best_thresh,
                "is_sharpe": best_is_sharpe,
                "oos_ret":   float(np.prod(1.0 + oos_rets) - 1.0) if len(oos_rets) else np.nan,
                "oos_bh_ret":  bh_ret,
                "oos_bl_ret":  bl_ret,
                "oos_sharpe":  _sharpe(oos_rets),
                "oos_max_dd":  _max_dd(oos_rets),
                "oos_trades":  n_t,
                "oos_suppressed": n_s,
                "oos_blocked":    n_b,
                "oos_max_supp_dd": n_msdd,
                "n_oos_days":  len(oos_rets),
            })

    # Concatenate per-variant OOS daily returns for a true daily-chained MaxDD.
    daily_concat = {v: (np.concatenate(arrs) if arrs else np.array([]))
                    for v, arrs in daily_by_variant.items()}
    return rows, daily_concat


def wf_chain_stats(rows, variant, daily_map=None):
    """Aggregate chain-linked stats across OOS years for one variant.

    daily_map (optional): {series_name: concatenated daily-return array}. When
    provided, MaxDD is the true daily-chained drawdown (MED-6); otherwise it
    falls back to the coarse annual-resolution drawdown.
    """
    v = sorted([r for r in rows if r["variant"] == variant], key=lambda r: r["year"])
    if not v:
        return {}
    strat = np.array([r["oos_ret"]    for r in v], dtype=float)
    bh    = np.array([r["oos_bh_ret"] for r in v], dtype=float)
    bl    = np.array([r["oos_bl_ret"] for r in v], dtype=float)
    n_years = sum(r["n_oos_days"] for r in v) / 252.0
    def _c(eq): return float(eq[-1] ** (1.0 / n_years) - 1.0)
    strat_eq = np.cumprod(1.0 + strat)

    if daily_map is not None:
        strat_dd = _max_dd(daily_map[variant]) if len(daily_map.get(variant, [])) else np.nan
        bh_dd    = _max_dd(daily_map["bh"])    if len(daily_map.get("bh", []))   else np.nan
    else:
        strat_dd = _max_dd_annual(strat)
        bh_dd    = _max_dd_annual(bh)

    # Worst drawdown endured while holding through a suppressed exit (HIGH-3).
    supp = [r["oos_max_supp_dd"] for r in v if r.get("oos_max_supp_dd", 0.0) < 0.0]
    worst_supp_dd = float(min(supp)) if supp else 0.0

    return {
        "strat_cagr":   _c(strat_eq),
        "bh_cagr":      _c(np.cumprod(1.0 + bh)),
        "bl_cagr":      _c(np.cumprod(1.0 + bl)),
        "max_dd":       strat_dd,
        "bh_max_dd":    bh_dd,
        "worst_supp_dd": worst_supp_dd,
        "hit_rate":     float(np.mean(strat > 0)),
        "avg_thresh":   float(np.mean([r["is_thresh"] for r in v])),
        "avg_oos_sharpe": float(np.nanmean([r["oos_sharpe"] for r in v])),
        "n_years":      len(v),
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def pct(x, d=2, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    fmt = f"+.{d}f" if signed else f".{d}f"
    return f"{x * 100:{fmt}}%"


def _best(df_sym, variant, maximize="sharpe"):
    v = df_sym[(df_sym["variant"] == variant) & df_sym["threshold"].notna()]
    if v.empty:
        return None
    return v.loc[v[maximize].idxmax()]


def write_markdown(all_rows, out_path):
    df = pd.DataFrame(all_rows)
    lines = [
        "# EMA + ML Compound Filter Backtest",
        f"**Date:** {TODAY_STR}  ",
        f"**Period:** {OOS_START} → present  ",
        "**Method:** EMA crossover (static IS-optimized params 2000–2019) gated by SP500/NDX MODERATE monthly regressor  ",
        "",
        "## Executive Summary",
        "",
        ("Tests whether suppressing EMA exits (or gating entries) when the ML regressor "
         "predicts strong forward returns can recover CAGR without sacrificing drawdown protection. "
         "Best threshold selected by maximizing OOS Sharpe over the 2010–present window "
         "(in-sample on the OOS period — treat as directional evidence only)."),
        "",
        "**EMA parameters:**",
        "- SPY: entry EMA(12/16), exit EMA(8/10), 6% trailing stop",
        "- QQQ: entry EMA(12/24), exit EMA(10/23), 5% trailing stop",
        "",
        "**ML filter:** `Predicted_Return` (%) from SP500/NDX MODERATE walk-forward regressor. "
        "Each row is month-end data; its forecast is shifted +1 month so the month-M "
        "forecast is applied to trading days in M+1 (knowable only after month-M close — "
        "no look-ahead bias).",
        "",
        "**Rule variants:**",
        "1. **Exit filter** — suppress EMA/trail exit if ML pred > threshold",
        "2. **Entry gate** — block EMA entry if ML pred ≤ threshold",
        "3. **Combined** — both filters at same threshold",
        "",
        "---",
        "",
    ]

    for sym in ["SPY", "QQQ"]:
        sub = df[df["symbol"] == sym]
        bh_row = sub[sub["variant"] == "bh"].iloc[0]
        bl_row = sub[sub["variant"] == "baseline"].iloc[0]

        bef = _best(sub, "exit_filter")
        beg = _best(sub, "entry_gate")
        bco = _best(sub, "combined")

        def _f(r, col):
            if r is None: return "n/a"
            v = r[col]
            if col in ("cagr", "max_dd"): return pct(v)
            if col == "sharpe": return f"{v:+.2f}"
            return str(v)

        lines += [
            f"## {sym}",
            "",
            f"| Metric | BH | EMA Baseline | Best Exit Filter | Best Entry Gate | Best Combined |",
            f"|---|---|---|---|---|---|",
            (f"| CAGR | {pct(bh_row['cagr'])} | {pct(bl_row['cagr'])} | "
             f"{_f(bef,'cagr')} | {_f(beg,'cagr')} | {_f(bco,'cagr')} |"),
            (f"| Sharpe | {bh_row['sharpe']:+.2f} | {bl_row['sharpe']:+.2f} | "
             f"{_f(bef,'sharpe')} | {_f(beg,'sharpe')} | {_f(bco,'sharpe')} |"),
            (f"| MaxDD | {pct(bh_row['max_dd'])} | {pct(bl_row['max_dd'])} | "
             f"{_f(bef,'max_dd')} | {_f(beg,'max_dd')} | {_f(bco,'max_dd')} |"),
            (f"| Best threshold | — | — | "
             f"{bef['threshold']:+.1f}% | {beg['threshold']:+.1f}% | {bco['threshold']:+.1f}% |"
             if (bef is not None and beg is not None and bco is not None) else "| Best threshold | — | — | n/a | n/a | n/a |"),
            "",
        ]

        # Per-variant sweep tables
        for vname, vcol, vextra in [
            ("exit_filter", "n_suppressed", "#Suppressed"),
            ("entry_gate",  "n_blocked",    "#Blocked"),
            ("combined",    "n_suppressed", "#Suppressed"),
        ]:
            v_sub = sub[sub["variant"] == vname].sort_values("threshold")
            label = vname.replace("_", " ").title()
            lines += [
                f"### {sym} — {label} Sweep",
                "",
                f"| Threshold | CAGR | Sharpe | MaxDD | Supp MaxDD | #Trades | {vextra} |",
                "|---|---|---|---|---|---|---|",
                (f"| Baseline | {pct(bl_row['cagr'])} | {bl_row['sharpe']:+.2f} | "
                 f"{pct(bl_row['max_dd'])} | n/a | {int(bl_row['n_trades'])} | 0 |"),
            ]
            for _, r in v_sub.iterrows():
                extra_val = int(r[vcol])
                lines.append(
                    f"| {r['threshold']:+.1f}% | {pct(r['cagr'])} | {r['sharpe']:+.2f} | "
                    f"{pct(r['max_dd'])} | {pct(r['max_supp_dd'])} | {int(r['n_trades'])} | {extra_val} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Caveats",
        "",
        "1. **Static EMA params**: equity curves use IS-optimized params (not per-year walk-forward "
        "   params). A rigorous test would apply the filter within each walk-forward fold separately.",
        "",
        "2. **Threshold is in-sample on the OOS window**: scanning 2010–present to find the best "
        "   threshold means the threshold is fit to the same period being evaluated. "
        "   Treat as directional evidence; true OOS requires nested walk-forward threshold selection.",
        "",
        "3. **Trail reset on suppression**: when a trail stop fires and is suppressed, peak resets "
        "   to the current close so the trail doesn't repeatedly fire at the same level.",
        "",
        "4. **Prediction coverage**: ML predictions run 2000-01-01 → 2026-05-01. "
        "   Days beyond May 2026 have no prediction; filter is bypassed (behaves as baseline).",
        "",
        "5. **Entry gate without prediction**: if no prediction is available for a month, "
        "   the entry gate is bypassed (entry allowed). Exit filter is also bypassed.",
    ]

    out_path.write_text("\n".join(lines))
    print(f"Markdown written → {out_path}")


def write_wf_markdown(all_wf_rows, daily_maps, out_path):
    df = pd.DataFrame(all_wf_rows)
    lines = [
        "# EMA + ML Compound Filter — Walk-Forward Threshold Validation",
        f"**Date:** {TODAY_STR}  ",
        f"**IS window:** {IS_YEARS_WF} years (108 months, same as EMA walk-forward optimum)  ",
        f"**OOS years:** 2010–2026 (17 folds)  ",
        "**Method:** For each OOS year, IS-optimal threshold is selected (max Sharpe on prior 9yr), "
        "then applied to the next calendar year OOS.  ",
        "",
        "## Executive Summary",
        "",
        ("Walk-forward threshold selection tests whether the ML-filter improvement survives rigorous "
         "out-of-sample validation. The threshold is chosen each year on the preceding 9 years of data "
         "and applied blind to the next year — no hindsight into the OOS period."),
        "",
        "---",
        "",
    ]

    for sym in ["SPY", "QQQ"]:
        sub_rows = [r for r in all_wf_rows if r["symbol"] == sym]
        lines.append(f"## {sym}")
        lines.append("")

        # Aggregate comparison table
        lines += [
            "| Metric | BH | EMA Baseline | WF Exit Filter | WF Entry Gate | WF Combined |",
            "|---|---|---|---|---|---|",
        ]
        stats = {v: wf_chain_stats(sub_rows, v, daily_maps[sym]) for v, _ in VARIANTS}
        s0 = next(iter(stats.values()))  # BH/baseline same across variants
        lines += [
            (f"| Chain CAGR | {pct(s0['bh_cagr'])} | {pct(s0['bl_cagr'])} | "
             f"{pct(stats['exit_filter']['strat_cagr'])} | "
             f"{pct(stats['entry_gate']['strat_cagr'])} | "
             f"{pct(stats['combined']['strat_cagr'])} |"),
            (f"| Avg OOS Sharpe | — | — | "
             f"{stats['exit_filter']['avg_oos_sharpe']:+.2f} | "
             f"{stats['entry_gate']['avg_oos_sharpe']:+.2f} | "
             f"{stats['combined']['avg_oos_sharpe']:+.2f} |"),
            (f"| Chain MaxDD (daily) | {pct(s0['bh_max_dd'])} | — | "
             f"{pct(stats['exit_filter']['max_dd'])} | "
             f"{pct(stats['entry_gate']['max_dd'])} | "
             f"{pct(stats['combined']['max_dd'])} |"),
            (f"| Worst Supp DD | — | — | "
             f"{pct(stats['exit_filter']['worst_supp_dd'])} | "
             f"{pct(stats['entry_gate']['worst_supp_dd'])} | "
             f"{pct(stats['combined']['worst_supp_dd'])} |"),
            (f"| Hit rate (yr>0) | — | — | "
             f"{pct(stats['exit_filter']['hit_rate'])} | "
             f"{pct(stats['entry_gate']['hit_rate'])} | "
             f"{pct(stats['combined']['hit_rate'])} |"),
            (f"| Avg IS threshold | — | — | "
             f"{stats['exit_filter']['avg_thresh']:+.1f}% | "
             f"{stats['entry_gate']['avg_thresh']:+.1f}% | "
             f"{stats['combined']['avg_thresh']:+.1f}% |"),
            "",
        ]

        # Per-variant year-by-year tables
        for vname, _ in VARIANTS:
            v_rows = sorted([r for r in sub_rows if r["variant"] == vname],
                            key=lambda r: r["year"])
            label = vname.replace("_", " ").title()
            lines += [
                f"### {sym} — {label} (Year-by-Year)",
                "",
                "| Year | IS Thresh | IS Sharpe | OOS Strat | OOS BH | OOS Baseline | OOS Sharpe | OOS MaxDD | Supp MaxDD |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in v_rows:
                lines.append(
                    f"| {r['year']} | {r['is_thresh']:+.1f}% | {r['is_sharpe']:+.2f} | "
                    f"{pct(r['oos_ret'])} | {pct(r['oos_bh_ret'])} | {pct(r['oos_bl_ret'])} | "
                    f"{r['oos_sharpe']:+.2f} | {pct(r['oos_max_dd'])} | {pct(r['oos_max_supp_dd'])} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Interpretation",
        "",
        "- **Chain CAGR**: multiply all 17 annual OOS returns to get the 17-year compounded return.",
        "- **Avg IS threshold**: average threshold selected across 17 IS windows — shows whether",
        "  the model consistently prefers high or low thresholds.",
        "- **Hit rate**: fraction of OOS years where the filtered strategy earned a positive return.",
        "- **IS→OOS Sharpe decay**: IS Sharpe in the year-by-year table measures how well the",
        "  threshold fitted IS data; OOS Sharpe is what it actually delivered.",
        "",
        "## Caveats",
        "",
        "- EMA parameters are static (IS-optimized 2000–2019); only the ML filter threshold is",
        "  walk-forward validated here.",
        "- 2026 is a partial year (through May 2026); its contribution is proportionally smaller.",
        "- Position state starts fresh at the beginning of each OOS year (same convention as",
        "  walkforward_ema_optimization.py). Trades spanning year boundaries are not tracked.",
    ]

    out_path.write_text("\n".join(lines))
    print(f"WF markdown written → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_rows     = []
    all_wf_rows  = []
    all_wf_daily = {}   # {sym: {series_name: daily-return array}} for daily MaxDD

    for sym in ["SPY", "QQQ"]:
        print(f"\n{'='*55}\n{sym}")
        prices_df_full = load_prices_full(sym)
        pred_df        = load_ml_predictions(sym)
        print(f"  Prices (full): {len(prices_df_full)} rows, "
              f"{prices_df_full['date'].iloc[0].date()} → {prices_df_full['date'].iloc[-1].date()}")

        # ── In-sample threshold sweep (existing analysis) ──
        rows = run_all(sym, prices_df_full, pred_df)
        all_rows.extend(rows)

        sub = pd.DataFrame(rows)
        bh  = sub[sub["variant"] == "bh"].iloc[0]
        bl  = sub[sub["variant"] == "baseline"].iloc[0]
        print(f"  BH:       CAGR={pct(bh['cagr'])}, Sharpe={bh['sharpe']:+.2f}, MaxDD={pct(bh['max_dd'])}")
        print(f"  Baseline: CAGR={pct(bl['cagr'])}, Sharpe={bl['sharpe']:+.2f}, MaxDD={pct(bl['max_dd'])}")
        for vname in ("exit_filter", "entry_gate", "combined"):
            best = _best(sub, vname)
            if best is not None:
                print(f"  Best {vname}: threshold={best['threshold']:+.1f}%, "
                      f"CAGR={pct(best['cagr'])}, Sharpe={best['sharpe']:+.2f}, "
                      f"MaxDD={pct(best['max_dd'])}")

        # ── Walk-forward threshold validation ──
        print(f"\n  Running walk-forward threshold validation ({IS_YEARS_WF}yr IS, 17 OOS years)…")
        wf_rows, wf_daily = run_walkforward_threshold(sym, prices_df_full, pred_df)
        all_wf_rows.extend(wf_rows)
        all_wf_daily[sym] = wf_daily

        for vname, _ in VARIANTS:
            st = wf_chain_stats(wf_rows, vname, wf_daily)
            label = vname.replace("_", " ").title()
            print(f"  WF {label}: chain CAGR={pct(st['strat_cagr'])}, "
                  f"avg Sharpe={st['avg_oos_sharpe']:+.2f}, "
                  f"MaxDD={pct(st['max_dd'])}, worst supp DD={pct(st['worst_supp_dd'])}, "
                  f"hit={pct(st['hit_rate'],0)}, avg thresh={st['avg_thresh']:+.1f}%")

    # ── Write outputs ──
    csv_path = RESULTS / f"ema_ml_filter_{TODAY_STR}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nIS sweep CSV   → {csv_path}")

    write_markdown(all_rows, RESULTS / f"ema_ml_filter_{TODAY_STR}.md")

    wf_csv = RESULTS / f"ema_ml_wf_threshold_{TODAY_STR}.csv"
    pd.DataFrame(all_wf_rows).to_csv(wf_csv, index=False)
    print(f"WF results CSV → {wf_csv}")

    write_wf_markdown(all_wf_rows, all_wf_daily, RESULTS / f"ema_ml_wf_threshold_{TODAY_STR}.md")


if __name__ == "__main__":
    main()
