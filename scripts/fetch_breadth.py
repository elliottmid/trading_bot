"""Fetch breadth ETFs (IWM + 6 sector SPDRs) daily for breadth features."""
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv(Path(__file__).parent.parent / ".env")

SYMBOLS = ["IWM", "XLK", "XLY", "XLF", "XLI", "XLP", "XLU", "XLV"]
MONTHS_BACK = 63
RAW = Path(__file__).parent.parent / "data" / "raw"

client = StockHistoricalDataClient(
    api_key=os.environ["ALPACA_API_KEY"],
    secret_key=os.environ["ALPACA_SECRET_KEY"],
)

end = datetime.utcnow() - timedelta(minutes=20)
start = end - timedelta(days=MONTHS_BACK * 31)

req = StockBarsRequest(symbol_or_symbols=SYMBOLS, timeframe=TimeFrame.Day,
                      start=start, end=end, feed="iex")
bars = client.get_stock_bars(req).df
out = RAW / "breadth_daily.parquet"
bars.to_parquet(out)
print(f"breadth: {len(bars):,} rows across {bars.index.get_level_values(0).nunique()} symbols -> {out}")
