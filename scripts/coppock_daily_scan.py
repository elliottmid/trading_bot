"""Daily Coppock trough-trigger scan on SPY.

Run once per weekday after market close. Fetches fresh daily bars from Alpaca,
computes the Coppock curve, and decides whether to emit a verdict.

Stdout contract (for the launchd wrapper):
  - Empty stdout  -> no new bar since last run; stay silent.
  - One line      -> new bar found; line is the notification body.
  - Non-zero exit -> error; stderr holds the detail.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

SYMBOL = "SPY"
ROC_FAST, ROC_SLOW, WMA_LEN = 11, 14, 10
STATE_FILE = ROOT / "data" / "models" / "coppock_last_scan.txt"


def fetch_spy_daily() -> pd.Series:
    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=180)
    req = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError("Alpaca returned no SPY daily bars")
    close = bars.loc[SYMBOL]["close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.sort_index()


def coppock_curve(close: pd.Series) -> pd.Series:
    s = close.pct_change(ROC_SLOW) * 100 + close.pct_change(ROC_FAST) * 100
    w = np.arange(1, WMA_LEN + 1, dtype=float)
    return s.rolling(WMA_LEN).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def verdict_line(close: pd.Series, coppock: pd.Series) -> str:
    c = coppock.iloc[-1]
    c1 = coppock.iloc[-2]
    c2 = coppock.iloc[-3]
    last_date = close.index[-1].date()
    last_close = close.iloc[-1]

    fired = (c1 < 0) and (c1 <= c2) and (c > c1)
    if fired:
        tag = "TRIGGER YES"
        reason = ""
    else:
        tag = "TRIGGER NO"
        if c1 >= 0:
            reason = " (c1>=0)"
        elif c1 > c2:
            reason = " (not a trough)"
        else:
            reason = " (coppock ticked down)"

    return (f"SPY {last_date} close={last_close:.2f} "
            f"coppock={c:+.2f} (yday {c1:+.2f}) {tag}{reason}")


def main() -> int:
    close = fetch_spy_daily()
    if len(close) < ROC_SLOW + WMA_LEN + 2:
        print("ERROR: not enough SPY history for Coppock", file=sys.stderr)
        return 1

    last_bar = str(close.index[-1].date())
    prev = STATE_FILE.read_text().strip() if STATE_FILE.exists() else ""
    if last_bar == prev:
        return 0  # silent: no new bar

    coppock = coppock_curve(close)
    line = verdict_line(close, coppock)
    print(line)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(last_bar + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
