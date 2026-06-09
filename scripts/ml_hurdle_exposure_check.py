# Author: Elliott Middleton, assisted by Claude
# Date: 2026-06-09
# Description: Compare the EMA + MODERATE exit-filter at exit hurdles 0.0% vs 0.5%
#   on the core performance measures — CAGR, Sharpe, MaxDD, and annual hit-rate.
#   Fixed-threshold, full-OOS (2010+) continuous simulation via the validated
#   simulate_period(). Hit-rate is the share of calendar years with a positive
#   strategy return, decomposed from the same continuous daily series.

import numpy as np
import pandas as pd

from backtest_ema_ml_filter import (
    PARAMS, OOS_START, load_prices_full, load_ml_predictions,
    _ema, simulate_period, _cagr, _sharpe, _max_dd,
)

SYMBOLS = ["SPY", "QQQ"]
HURDLES = [0.0, 0.5]          # exit-filter thresholds to compare (%)


def _pred_dict(pred_df):
    """+1-month shift: month-M row forecasts M+1 (matches scanner/backtest)."""
    p = pred_df.copy()
    p["ym"] = p["Date"].dt.to_period("M") + 1
    return dict(zip(p["ym"], p["Predicted_Return"]))


def _hit_rate(rets, idx_dates):
    """Share of calendar years with positive chained return."""
    s = pd.Series(rets, index=pd.DatetimeIndex(idx_dates))
    annual = s.groupby(s.index.year).apply(lambda r: float(np.prod(1.0 + r) - 1.0))
    return float((annual > 0).mean()), int((annual > 0).sum()), len(annual)


def main():
    print(f"\nEMA + MODERATE exit-filter — exit hurdle 0.0% vs +0.5%  (fixed threshold, OOS {OOS_START}+)\n")
    rows = []

    for sym in SYMBOLS:
        params = PARAMS[sym]
        prices = load_prices_full(sym)
        pred   = load_ml_predictions(sym)
        pd_dict = _pred_dict(pred)

        closes = prices["close"].values
        dates  = prices["date"].values
        ema_ef = _ema(prices["close"], params["ef"])
        ema_es = _ema(prices["close"], params["es"])
        ema_xf = _ema(prices["close"], params["xf"])
        ema_xs = _ema(prices["close"], params["xs"])

        oos_start = pd.Timestamp(OOS_START)
        oos_end   = prices["date"].max()
        oos_mask  = (prices["date"] >= oos_start) & (prices["date"] <= oos_end)
        n_years   = (int(oos_mask.sum()) - 1) / 252.0

        # Dates aligned to simulate_period's daily_rets (= OOS idxs[1:]).
        idxs      = np.where((dates >= np.datetime64(oos_start)) &
                             (dates <= np.datetime64(oos_end)))[0]
        sim_dates = dates[idxs[1:]]

        # Buy-hold
        bh = prices.loc[oos_mask].copy()
        bh_rets   = bh["bh_equity"].pct_change().fillna(0.0).values[1:]
        bh_dates  = bh["date"].values[1:]
        hr, hy, ny = _hit_rate(bh_rets, bh_dates)
        rows.append((sym, "buy-hold", _cagr(bh_rets, n_years), _sharpe(bh_rets),
                     _max_dd(bh_rets), hr, hy, ny))

        # Baseline EMA (no filter)
        bl_rets, _, _, _, _ = simulate_period(
            closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
            pd_dict, params, oos_start, oos_end)
        hr, hy, ny = _hit_rate(bl_rets, sim_dates)
        rows.append((sym, "EMA base (no filter)", _cagr(bl_rets, n_years),
                     _sharpe(bl_rets), _max_dd(bl_rets), hr, hy, ny))

        # Exit filter at each hurdle
        for h in HURDLES:
            rets, nt, ns, nb, msdd = simulate_period(
                closes, dates, ema_ef, ema_es, ema_xf, ema_xs,
                pd_dict, params, oos_start, oos_end, exit_thresh=h)
            hr, hy, ny = _hit_rate(rets, sim_dates)
            rows.append((sym, f"exit hurdle {h:+.1f}%", _cagr(rets, n_years),
                         _sharpe(rets), _max_dd(rets), hr, hy, ny))

    hdr = f"{'Sym':4} {'Variant':22} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} {'HitRate':>14}"
    print(hdr); print("-" * len(hdr))
    last_sym = None
    for sym, var, cagr, shp, dd, hr, hy, ny in rows:
        if last_sym and sym != last_sym:
            print()
        print(f"{sym:4} {var:22} {cagr:+7.1%} {shp:+7.2f} {dd:+7.1%} "
              f"{hr:6.0%} ({hy}/{ny})".rjust(0))
        last_sym = sym
    print()


if __name__ == "__main__":
    main()
