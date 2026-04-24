"""Build XGBoost feature matrix with N-day forward return target.

Inputs:  data/raw/equities_daily.parquet, rates_daily.parquet
Output:  data/processed/features_daily_H{N}.parquet

One row per (symbol, date). Columns = features + target_fwd_ret.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def rsi(x: pd.Series, n: int = 14) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = -d.clip(upper=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def per_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    c, v, h, l = g["close"], g["volume"], g["high"], g["low"]
    logret = np.log(c / c.shift(1))
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out = pd.DataFrame({
        "ret_1": logret,
        "ret_5": logret.rolling(5).sum(),
        "ret_10": logret.rolling(10).sum(),
        "ret_20": logret.rolling(20).sum(),
        "vol_10": logret.rolling(10).std(),
        "vol_20": logret.rolling(20).std(),
        "rsi_14": rsi(c, 14),
        "macd": macd,
        "macd_sig": macd.ewm(span=9, adjust=False).mean(),
        "atr_14": tr.rolling(14).mean() / c,
        "vol_z_20": (v - v.rolling(20).mean()) / v.rolling(20).std(),
        "skew_20": logret.rolling(20).skew(),
        "kurt_20": logret.rolling(20).kurt(),
        "close_to_hi_20": c / c.rolling(20).max() - 1,
        "close_to_lo_20": c / c.rolling(20).min() - 1,
    }, index=g.index)
    return out


def build(horizon: int) -> pd.DataFrame:
    eq = pd.read_parquet(RAW / "equities_daily.parquet").sort_index()
    eq.index = eq.index.set_names(["symbol", "ts"])
    eq_flat = eq.reset_index()
    eq_flat["ts"] = pd.to_datetime(eq_flat["ts"]).dt.tz_localize(None).dt.normalize()

    feats = []
    for sym, g in eq_flat.groupby("symbol"):
        g = g.sort_values("ts").set_index("ts")
        f = per_symbol_features(g)
        f["symbol"] = sym
        f["close"] = g["close"]
        f["target_fwd_ret"] = np.log(g["close"].shift(-horizon) / g["close"])
        feats.append(f.reset_index())
    panel = pd.concat(feats, ignore_index=True)

    # Cross-asset features (wide close matrix)
    wide = panel.pivot(index="ts", columns="symbol", values="close")
    cross = pd.DataFrame(index=wide.index)
    cross["spy_qqq_sprd"] = np.log(wide["QQQ"] / wide["SPY"])
    cross["pslv_phys_sprd"] = np.log(wide["PSLV"] / wide["PHYS"])
    cross["eq_metals_ratio"] = np.log(wide["SPY"] / wide["PHYS"])
    spy_ret = np.log(wide["SPY"] / wide["SPY"].shift(1))
    phys_ret = np.log(wide["PHYS"] / wide["PHYS"].shift(1))
    cross["eq_metals_corr_20"] = spy_ret.rolling(20).corr(phys_ret)
    # Rotation momentum on the SPY/QQQ spread
    cross["spy_qqq_sprd_chg_5"] = cross["spy_qqq_sprd"].diff(5)
    cross["spy_qqq_sprd_chg_20"] = cross["spy_qqq_sprd"].diff(20)
    cross["pslv_phys_sprd_chg_20"] = cross["pslv_phys_sprd"].diff(20)

    # Macro (rates + VIX + credit + USD)
    m = pd.read_parquet(RAW / "macro_daily.parquet").sort_index()
    m.index = m.index.tz_localize(None) if m.index.tz else m.index
    m["term_10y_3m"] = m["t_bond_10y"] - m["t_bill_3mo"]
    m["d_t_bond_10y"] = m["t_bond_10y"].diff()
    m["d_t_bill_3mo"] = m["t_bill_3mo"].diff()
    # VIX regime
    m["vix_chg_5"] = m["vix"].diff(5)
    m["vix_term"] = m["vix_3m"] - m["vix"]           # positive = contango (calm), negative = stress
    m["vix_term_chg_5"] = m["vix_term"].diff(5)
    # Credit stress
    m["hy_spread_chg_20"] = m["hy_spread"].diff(20)
    # Dollar momentum — matters for PSLV/PHYS
    m["usd_ret_20"] = np.log(m["usd_index"] / m["usd_index"].shift(20))

    # Breadth (sector ETFs + small-cap)
    br = pd.read_parquet(RAW / "breadth_daily.parquet").sort_index()
    br = br.reset_index()
    br["timestamp"] = pd.to_datetime(br["timestamp"]).dt.tz_localize(None).dt.normalize()
    br_wide = br.pivot(index="timestamp", columns="symbol", values="close")
    # SPY daily for denominators
    spy_daily = wide["SPY"].reindex(br_wide.index).ffill()
    breadth = pd.DataFrame(index=br_wide.index)
    breadth["iwm_spy_ratio"] = np.log(br_wide["IWM"] / spy_daily)
    breadth["iwm_spy_chg_20"] = breadth["iwm_spy_ratio"].diff(20)
    cyc = ["XLK", "XLY", "XLF", "XLI"]
    defn = ["XLP", "XLU", "XLV"]
    cyc_idx = br_wide[cyc].mean(axis=1)
    def_idx = br_wide[defn].mean(axis=1)
    breadth["cyc_def_ratio"] = np.log(cyc_idx / def_idx)
    breadth["cyc_def_chg_20"] = breadth["cyc_def_ratio"].diff(20)
    # Breadth: fraction of sectors above their 50d MA
    sectors = cyc + defn
    ma50 = br_wide[sectors].rolling(50).mean()
    breadth["sec_above_50ma"] = (br_wide[sectors] > ma50).sum(axis=1) / len(sectors)
    # Dispersion: cross-sectional std of 20d sector log returns
    sec_ret_20 = np.log(br_wide[sectors] / br_wide[sectors].shift(20))
    breadth["sec_dispersion_20"] = sec_ret_20.std(axis=1)

    panel = panel.merge(cross, left_on="ts", right_index=True, how="left")
    panel = panel.merge(breadth, left_on="ts", right_index=True, how="left")
    panel = panel.merge(m, left_on="ts", right_index=True, how="left")
    panel = panel.drop(columns=["close"])
    panel = panel.dropna(subset=["target_fwd_ret"]).sort_values(["symbol", "ts"])

    out = PROC / f"features_daily_H{horizon}.parquet"
    panel.to_parquet(out, index=False)
    print(f"features: {len(panel):,} rows, {panel.shape[1]} cols, target=target_fwd_ret (H={horizon}d) -> {out}")
    print(f"symbols: {panel['symbol'].value_counts().to_dict()}")
    print(f"date range: {panel['ts'].min().date()} .. {panel['ts'].max().date()}")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5, help="forward return horizon in trading days")
    args = ap.parse_args()
    build(args.horizon)
