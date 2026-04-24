"""Coppock 'turns up from negative' trigger on daily SPY — 20-day forward return log.

Trigger rule (local trough below zero):
    Coppock[t-1] < 0  AND  Coppock[t-1] <= Coppock[t-2]  AND  Coppock[t] > Coppock[t-1]

For each trigger, record entry close, close 20 trading days later, $ change, and % return.
Writes per-trigger CSV and prints summary stats. Not a strategy backtest — no PnL, no costs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "data" / "models"

ROC_FAST = 11
ROC_SLOW = 14
WMA_LEN = 10
FWD_DAYS = 20


def load_symbol(symbol: str) -> pd.DataFrame:
    eq = pd.read_parquet(RAW / "equities_daily.parquet").sort_index()
    eq.index = eq.index.set_names(["symbol", "ts"])
    df = eq.loc[symbol].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


def coppock_curve(close: pd.Series) -> pd.Series:
    roc_slow = close.pct_change(ROC_SLOW) * 100
    roc_fast = close.pct_change(ROC_FAST) * 100
    s = roc_slow + roc_fast
    weights = np.arange(1, WMA_LEN + 1, dtype=float)
    wsum = weights.sum()
    return s.rolling(WMA_LEN).apply(lambda x: np.dot(x, weights) / wsum, raw=True)


def find_triggers(close: pd.Series, coppock: pd.Series, symbol: str) -> pd.DataFrame:
    c1 = coppock.shift(1)
    c2 = coppock.shift(2)
    mask = (c1 < 0) & (c1 <= c2) & (coppock > c1)

    exit_close = close.shift(-FWD_DAYS)
    exit_date = close.index.to_series().shift(-FWD_DAYS)

    df = pd.DataFrame({
        "symbol": symbol,
        "trigger_date": close.index,
        "coppock_value": coppock.values,
        "entry_close": close.values,
        "exit_date": exit_date.values,
        "exit_close": exit_close.values,
    })
    df = df[mask.values].dropna(subset=["exit_close"]).reset_index(drop=True)
    df["price_change"] = df["exit_close"] - df["entry_close"]
    df["pct_return"] = df["exit_close"] / df["entry_close"] - 1.0
    return df


def summarize(trades: pd.DataFrame, symbol: str) -> None:
    n = len(trades)
    print(f"\n=== Coppock trough triggers on {symbol} (daily) ===")
    print(f"fwd window: {FWD_DAYS} trading days   |   triggers: {n}")
    if n == 0:
        print("no triggers in sample — nothing to summarize.")
        return

    r = trades["pct_return"]
    hit = (r > 0).mean()
    mean, median, std = r.mean(), r.median(), r.std()
    imin, imax = r.idxmin(), r.idxmax()
    sharpe_approx = (mean / std) * np.sqrt(252 / FWD_DAYS) if std > 0 else np.nan

    print(f"hit rate (> 0):     {hit:.1%}")
    print(f"mean return:        {mean:+.2%}")
    print(f"median return:      {median:+.2%}")
    print(f"stdev return:       {std:.2%}")
    print(f"min:  {r[imin]:+.2%}  on {trades.loc[imin, 'trigger_date'].date()}")
    print(f"max:  {r[imax]:+.2%}  on {trades.loc[imax, 'trigger_date'].date()}")
    print(f"approx annualized Sharpe (non-overlap adj): {sharpe_approx:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    bars = load_symbol(symbol)
    close = bars["close"]
    coppock = coppock_curve(close)

    trades = find_triggers(close, coppock, symbol)

    MODELS.mkdir(parents=True, exist_ok=True)
    out_path = MODELS / f"coppock_triggers_{symbol}.csv"
    trades.to_csv(out_path, index=False)

    print(f"data: {close.index[0].date()} .. {close.index[-1].date()}  ({len(close)} bars)")
    print(f"per-trigger log -> {out_path}")
    if not trades.empty:
        print("\n--- triggers ---")
        show = trades[["trigger_date", "coppock_value", "entry_close",
                       "exit_date", "exit_close", "price_change", "pct_return"]].copy()
        show["trigger_date"] = show["trigger_date"].dt.date
        show["exit_date"] = pd.to_datetime(show["exit_date"]).dt.date
        print(show.to_string(index=False,
            formatters={"coppock_value": "{:+.3f}".format,
                        "entry_close":   "{:.2f}".format,
                        "exit_close":    "{:.2f}".format,
                        "price_change":  "{:+.2f}".format,
                        "pct_return":    "{:+.2%}".format}))
    summarize(trades, symbol)


if __name__ == "__main__":
    main()
