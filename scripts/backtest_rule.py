"""Backtest conviction-weighted long-only SPY rule on OOS predictions.

Rule:
  Every REBAL_DAYS trading days, target weight = clip(z * Z_SCALE, 0, 1),
  where z = (pred - trailing_mean) / trailing_std over a 252-day window of predictions.
  Remainder earns DGS3MO. 1bp transaction cost on turnover.

Benchmarks:
  - Buy & hold SPY
  - 60/40 SPY/cash (constant)
  - Sign-only long/cash (weight=1 if pred>0 else 0)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "data" / "models"

REBAL_DAYS = 20
Z_WINDOW = 252
Z_SCALE = 0.5
TC_BPS = 1.0  # one-way transaction cost per unit turnover, in basis points


def load_data():
    preds = pd.read_parquet(MODELS / "wf_preds_H20_SPY.parquet")
    preds["ts"] = pd.to_datetime(preds["ts"]).dt.tz_localize(None).dt.normalize()
    preds = preds.sort_values("ts").reset_index(drop=True)

    eq = pd.read_parquet(RAW / "equities_daily.parquet").sort_index()
    eq.index = eq.index.set_names(["symbol", "ts"])
    spy = eq.loc["SPY"].reset_index().rename(columns={"timestamp": "ts"})
    spy["ts"] = pd.to_datetime(spy["ts"]).dt.tz_localize(None).dt.normalize()
    spy = spy[["ts", "close"]].rename(columns={"close": "spy_close"})

    macro = pd.read_parquet(RAW / "macro_daily.parquet").sort_index()
    macro.index = macro.index.tz_localize(None) if macro.index.tz else macro.index
    cash = macro[["t_bill_3mo"]].reset_index().rename(columns={"observation_date": "ts"})
    cash["ts"] = pd.to_datetime(cash["ts"]).dt.tz_localize(None).dt.normalize()
    cash["t_bill_3mo"] = cash["t_bill_3mo"].ffill()

    df = spy.merge(cash, on="ts", how="left").merge(preds[["ts", "pred"]], on="ts", how="left")
    df["t_bill_3mo"] = df["t_bill_3mo"].ffill()
    df["spy_ret"] = df["spy_close"].pct_change().fillna(0)
    df["cash_ret"] = (df["t_bill_3mo"].fillna(0) / 100) / 252
    return df


def build_weights(df: pd.DataFrame) -> pd.Series:
    """Rebalance every REBAL_DAYS when a prediction is available."""
    pred = df["pred"]
    # Trailing z-score of predictions (expanding up to Z_WINDOW, then rolling)
    roll = pred.rolling(Z_WINDOW, min_periods=60)
    z = (pred - roll.mean()) / roll.std()
    target = (z * Z_SCALE).clip(0, 1)

    w = pd.Series(np.nan, index=df.index)
    # First rebalance = first index where we have a prediction AND enough history
    first = int(target.first_valid_index() or 0)
    w.iloc[first] = target.iloc[first]
    last_rebal = first
    for i in range(first + 1, len(df)):
        if i - last_rebal >= REBAL_DAYS and not np.isnan(target.iloc[i]):
            w.iloc[i] = target.iloc[i]
            last_rebal = i
    return w.ffill().fillna(0)


def simulate(df: pd.DataFrame, w: pd.Series, tc_bps: float = TC_BPS) -> pd.DataFrame:
    port_ret = w.shift(1).fillna(0) * df["spy_ret"] + (1 - w.shift(1).fillna(0)) * df["cash_ret"]
    turnover = w.diff().abs().fillna(w.iloc[0])
    cost = turnover * (tc_bps / 1e4)
    port_ret_net = port_ret - cost
    out = pd.DataFrame({"ts": df["ts"], "weight": w, "gross_ret": port_ret,
                        "cost": cost, "ret": port_ret_net})
    out["nav"] = (1 + out["ret"]).cumprod()
    return out


def metrics(nav: pd.DataFrame, label: str) -> dict:
    r = nav["ret"].dropna()
    n = len(r)
    ann_factor = 252
    cagr = nav["nav"].iloc[-1] ** (ann_factor / n) - 1
    vol = r.std() * np.sqrt(ann_factor)
    sharpe = (r.mean() * ann_factor) / (r.std() * np.sqrt(ann_factor)) if r.std() > 0 else 0
    dd = nav["nav"] / nav["nav"].cummax() - 1
    max_dd = dd.min()
    downside = r[r < 0]
    sortino = (r.mean() * ann_factor) / (downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 else np.nan
    turnover = nav["cost"].sum() / (TC_BPS / 1e4) / (n / ann_factor)  # annualized turnover units
    return {"strategy": label, "CAGR": cagr, "vol": vol, "Sharpe": sharpe,
            "Sortino": sortino, "MaxDD": max_dd, "turnover_pa": turnover,
            "final_nav": nav["nav"].iloc[-1]}


def main():
    df = load_data().reset_index(drop=True)
    # Restrict to the OOS window (first prediction onward)
    start = df["pred"].first_valid_index()
    df = df.loc[start:].reset_index(drop=True)
    print(f"backtest window: {df['ts'].iloc[0].date()} .. {df['ts'].iloc[-1].date()}  ({len(df)} days)")

    # Strategy
    w_rule = build_weights(df)
    nav_rule = simulate(df, w_rule)

    # Benchmarks
    w_bh = pd.Series(1.0, index=df.index)
    nav_bh = simulate(df, w_bh, tc_bps=0)

    w_6040 = pd.Series(0.6, index=df.index)
    nav_6040 = simulate(df, w_6040, tc_bps=0)

    # Sign-only: weight=1 if pred>0, else 0, rebalanced every REBAL_DAYS
    sign = (df["pred"] > 0).astype(float)
    w_sign = pd.Series(np.nan, index=df.index)
    first = int(df["pred"].first_valid_index() or 0)
    w_sign.iloc[first] = sign.iloc[first]
    last_rebal = first
    for i in range(first + 1, len(df)):
        if i - last_rebal >= REBAL_DAYS and not np.isnan(sign.iloc[i]):
            w_sign.iloc[i] = sign.iloc[i]
            last_rebal = i
    w_sign = w_sign.ffill().fillna(0)
    nav_sign = simulate(df, w_sign)

    results = pd.DataFrame([
        metrics(nav_rule, "conviction_long_only"),
        metrics(nav_bh, "buy_hold_SPY"),
        metrics(nav_6040, "60/40_static"),
        metrics(nav_sign, "sign_only_long_cash"),
    ])
    print("\n=== strategy comparison ===")
    print(results.to_string(index=False,
          formatters={"CAGR": "{:.2%}".format, "vol": "{:.2%}".format,
                      "Sharpe": "{:.2f}".format, "Sortino": "{:.2f}".format,
                      "MaxDD": "{:.2%}".format, "turnover_pa": "{:.2f}".format,
                      "final_nav": "{:.3f}".format}))

    # Cash-decision hit rate: when rule went "mostly cash" (w < 0.2),
    # was the next 20d SPY return negative?
    nav_rule["w"] = w_rule.values
    nav_rule["spy_fwd_20"] = df["spy_close"].shift(-20) / df["spy_close"] - 1
    cash_mask = nav_rule["w"] < 0.2
    if cash_mask.sum() > 5:
        hit = (nav_rule.loc[cash_mask, "spy_fwd_20"] < 0).mean()
        print(f"\ncash-decision hit rate (w<0.2 AND spy_fwd_20<0): {hit:.1%}  over {int(cash_mask.sum())} obs")

    # Save curves for later inspection
    out = df[["ts", "spy_close", "pred"]].copy()
    out["w_rule"] = w_rule.values
    out["nav_rule"] = nav_rule["nav"].values
    out["nav_bh"] = nav_bh["nav"].values
    out["nav_6040"] = nav_6040["nav"].values
    out["nav_sign"] = nav_sign["nav"].values
    path = MODELS / "backtest_rule_H20_SPY.parquet"
    out.to_parquet(path, index=False)
    print(f"\ncurves saved -> {path}")


if __name__ == "__main__":
    main()
