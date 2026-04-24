"""Cross-sectional rotation backtest on pooled H=20 OOS predictions.

At each rebalance date, rank SPY/QQQ/PSLV/PHYS by predicted forward return
and allocate weights. Hold REBAL_DAYS trading days, then rebalance.

Strategies tested:
  - top2_eq:    50/50 in top-2 predicted, 0 in bottom-2
  - top1:       100% in top predicted
  - rank_ls:    long-short market-neutral (+50% top-2, -50% bottom-2)
  - rank_lo:    long-only rank-weighted (weights ∝ rank, normalized)

Benchmarks:
  - equal_weight: 25% each asset, rebalanced
  - buy_hold_SPY, buy_hold_QQQ
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "data" / "models"

SYMBOLS = ["SPY", "QQQ", "PSLV", "PHYS"]
REBAL_DAYS = 20
TC_BPS = 1.0


def load_prices(symbols: list[str]) -> pd.DataFrame:
    eq = pd.read_parquet(RAW / "equities_daily.parquet").sort_index()
    eq = eq.reset_index()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"]).dt.tz_localize(None).dt.normalize()
    wide = eq.pivot(index="timestamp", columns="symbol", values="close")[symbols]
    return wide.sort_index()


def load_preds(symbols: list[str]) -> pd.DataFrame:
    p = pd.read_parquet(MODELS / "wf_preds_H20.parquet")
    p["ts"] = pd.to_datetime(p["ts"]).dt.tz_localize(None).dt.normalize()
    return p.pivot_table(index="ts", columns="symbol", values="pred")[symbols].sort_index()


def build_rebalance_schedule(pred_index: pd.DatetimeIndex, price_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Pick every REBAL_DAYS-th date from the intersection."""
    avail = pred_index.intersection(price_index)
    return list(avail[::REBAL_DAYS])


def allocate(pred_row: pd.Series, strategy: str) -> np.ndarray:
    ranks = pred_row.rank(method="first").values  # 1 = lowest
    n = len(ranks)
    w = np.zeros(n)
    if strategy == "top_half_eq":
        k = n // 2
        top = ranks > n - k
        w[top] = 1.0 / k
    elif strategy == "top1":
        w[ranks == n] = 1.0
    elif strategy == "rank_ls":
        k = max(1, n // 2)
        top = ranks > n - k
        bot = ranks <= k
        w[top] = 0.5 / k
        w[bot] = -0.5 / k
    elif strategy == "rank_lo":
        w = ranks / ranks.sum()
    else:
        raise ValueError(strategy)
    return w


def simulate(prices: pd.DataFrame, weights_by_date: dict[pd.Timestamp, np.ndarray],
             tc_bps: float = TC_BPS) -> pd.DataFrame:
    """Simulate daily NAV. Weight at rebalance date applies from next day forward."""
    rets = prices.pct_change().fillna(0)
    w_daily = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    prev = np.zeros(prices.shape[1])
    costs = pd.Series(0.0, index=prices.index)
    rebal_dates = sorted(weights_by_date.keys())

    for i, dt in enumerate(rebal_dates):
        new = weights_by_date[dt]
        costs.loc[dt] = np.abs(new - prev).sum() * (tc_bps / 1e4)
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1]
        mask = (prices.index > dt) & (prices.index <= next_rebal)
        w_daily.loc[mask, :] = new
        prev = new

    port_ret = (w_daily * rets).sum(axis=1) - costs
    nav = (1 + port_ret).cumprod()
    return pd.DataFrame({"ret": port_ret, "nav": nav, "cost": costs})


def metrics(nav: pd.DataFrame, label: str) -> dict:
    r = nav["ret"].dropna()
    n = len(r)
    ann = 252
    cagr = nav["nav"].iloc[-1] ** (ann / n) - 1 if n > 0 else 0
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() * ann) / (r.std() * np.sqrt(ann)) if r.std() > 0 else 0
    dd = nav["nav"] / nav["nav"].cummax() - 1
    max_dd = dd.min()
    downside = r[r < 0]
    sortino = (r.mean() * ann) / (downside.std() * np.sqrt(ann)) if len(downside) > 0 else np.nan
    return {"strategy": label, "CAGR": cagr, "vol": vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": max_dd, "final_nav": nav["nav"].iloc[-1]}


def per_fold_breakdown(rotation_ret: pd.Series, bench_ret: pd.Series):
    folds = pd.read_csv(MODELS / "wf_folds_H20.csv")
    rows = []
    for _, f in folds.iterrows():
        ts_start, ts_end = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        rmask = (rotation_ret.index >= ts_start) & (rotation_ret.index < ts_end)
        bmask = (bench_ret.index >= ts_start) & (bench_ret.index < ts_end)
        rr, br = rotation_ret[rmask], bench_ret[bmask]
        if len(rr) == 0:
            continue
        rows.append({
            "fold": int(f["fold"]), "start": ts_start.date(), "end": ts_end.date(),
            "model_ic": f["rank_ic"],
            "rotation_cumret": (1 + rr).prod() - 1,
            "bench_cumret": (1 + br).prod() - 1,
            "excess": (1 + rr).prod() - (1 + br).prod(),
        })
    return pd.DataFrame(rows)


def run(universe: list[str], tc_bps: float, tag: str):
    prices = load_prices(universe)
    preds = load_preds(universe)
    schedule = build_rebalance_schedule(preds.index, prices.index)
    prices = prices.loc[prices.index >= schedule[0]].copy()
    print(f"\n### {tag} | universe={universe} | tc={tc_bps}bp | "
          f"{prices.index[0].date()}..{prices.index[-1].date()} | {len(schedule)} rebalances")

    strats = ["top_half_eq", "top1", "rank_ls", "rank_lo"]
    navs = {}
    for s in strats:
        wbyd = {dt: allocate(preds.loc[dt], s) for dt in schedule}
        navs[s] = simulate(prices, wbyd, tc_bps=tc_bps)

    n = len(universe)
    navs["equal_weight"] = simulate(prices, {dt: np.full(n, 1.0 / n) for dt in schedule}, tc_bps=tc_bps)
    bh_first = np.zeros(n); bh_first[universe.index("SPY")] = 1.0
    navs["buy_hold_SPY"] = simulate(prices, {schedule[0]: bh_first}, tc_bps=0)
    if "QQQ" in universe:
        bh_q = np.zeros(n); bh_q[universe.index("QQQ")] = 1.0
        navs["buy_hold_QQQ"] = simulate(prices, {schedule[0]: bh_q}, tc_bps=0)

    keys = ["top_half_eq", "top1", "rank_ls", "rank_lo", "equal_weight", "buy_hold_SPY"]
    if "QQQ" in universe: keys.append("buy_hold_QQQ")
    rows = [metrics(navs[k], k) for k in keys]
    results = pd.DataFrame(rows)
    print(results.to_string(index=False,
          formatters={"CAGR": "{:.2%}".format, "vol": "{:.2%}".format,
                      "Sharpe": "{:.2f}".format, "Sortino": "{:.2f}".format,
                      "MaxDD": "{:.2%}".format, "final_nav": "{:.3f}".format}))
    return navs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="SPY,QQQ,PSLV,PHYS")
    ap.add_argument("--tc-bps", type=float, default=1.0)
    ap.add_argument("--per-fold", action="store_true")
    args = ap.parse_args()
    universe = args.universe.split(",")
    navs = run(universe, args.tc_bps, f"tc={args.tc_bps}bp")

    if args.per_fold:
        print("\n=== per-fold decomposition (top_half_eq vs equal_weight) ===")
        pf = per_fold_breakdown(navs["top_half_eq"]["ret"], navs["equal_weight"]["ret"])
        print(pf.to_string(index=False,
              formatters={"model_ic": "{:+.3f}".format,
                          "rotation_cumret": "{:+.2%}".format,
                          "bench_cumret": "{:+.2%}".format,
                          "excess": "{:+.2%}".format}))


if __name__ == "__main__":
    main()
