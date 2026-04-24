"""Fetch 60+ months of daily and hourly bars from Alpaca for SPY/QQQ/PSLV/PHYS."""
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv(Path(__file__).parent.parent / ".env")

SYMBOLS = ["SPY", "QQQ", "PSLV", "PHYS"]
MONTHS_BACK = 63
RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

client = StockHistoricalDataClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
)

end = datetime.utcnow() - timedelta(minutes=20)  # free-tier 15m delay buffer
start = end - timedelta(days=MONTHS_BACK * 31)


def fetch(timeframe: TimeFrame, label: str) -> pd.DataFrame:
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=timeframe,
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(req).df
    if bars.empty:
        raise RuntimeError(f"No {label} bars returned")
    out = RAW / f"equities_{label}.parquet"
    bars.to_parquet(out)
    print(f"{label}: {len(bars):,} rows across {bars.index.get_level_values(0).nunique()} symbols -> {out}")
    return bars


if __name__ == "__main__":
    fetch(TimeFrame.Day, "daily")
    fetch(TimeFrame.Hour, "hourly")
