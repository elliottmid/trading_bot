#!/usr/bin/env python3
"""
Fetch daily OHLCV bars for EMA backtest universe from Alpaca.
Stores as parquet for efficient reloading.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ema_etf_universe import PRIMARY_UNIVERSE, UNIVERSE

# Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Load credentials from .env
from dotenv import load_dotenv
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_etf_data(universe, start_date="2020-12-14", end_date=None):
    """
    Fetch daily bars for all ETFs in universe, batching to avoid API limits.

    Args:
        universe: dict of {ticker: description}
        start_date: ISO date string or "YYYY-MM-DD"
        end_date: ISO date string or None (defaults to today)

    Returns:
        DataFrame with MultiIndex (symbol, date), columns [open, high, low, close, volume]
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, url_override=ALPACA_BASE_URL)

    symbols = list(universe.keys())
    logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date}")

    # Batch symbols to avoid API limits (fetch 5 at a time)
    batch_size = 5
    all_dfs = []

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        logger.info(f"  Batch {i//batch_size + 1}: {', '.join(batch_symbols)}")

        request = StockBarsRequest(
            symbol_or_symbols=batch_symbols,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )

        try:
            bars = client.get_stock_bars(request)
            df = bars.df

            # Rename columns to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]

            # Reset index to have symbol and timestamp as columns
            df = df.reset_index()

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            all_dfs.append(df)
            logger.info(f"    ✓ Fetched {len(df)} bars for batch")

        except Exception as e:
            logger.warning(f"    ✗ Failed to fetch batch {i//batch_size + 1}: {e}")
            # Continue to next batch instead of failing completely
            continue

    if not all_dfs:
        logger.error("Failed to fetch data from any batch")
        raise ValueError("No data fetched from Alpaca")

    result_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"✓ Fetched {len(result_df)} bars across {result_df['symbol'].nunique()} symbols")
    return result_df


def save_ema_etf_data(df, filename="ema_etfs_daily.parquet"):
    """Save DataFrame to parquet file."""
    output_path = DATA_DIR / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Saved {len(df)} rows to {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch EMA ETF data from Alpaca")
    parser.add_argument("--universe", choices=["primary", "full"], default="primary",
                        help="Which universe to fetch (primary=13 ETFs, full=30+)")
    parser.add_argument("--start-date", default="2020-12-14",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None,
                        help="End date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    universe = PRIMARY_UNIVERSE if args.universe == "primary" else UNIVERSE

    logger.info(f"Using {args.universe} universe ({len(universe)} symbols)")

    df = fetch_etf_data(universe, start_date=args.start_date, end_date=args.end_date)

    filename = f"ema_etfs_{args.universe}_daily.parquet"
    save_ema_etf_data(df, filename=filename)

    # Quick stats
    logger.info("\n--- Data Summary ---")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Symbols: {df['symbol'].nunique()}")
    logger.info(f"Total bars: {len(df)}")
    logger.info(f"Avg bars per symbol: {len(df) / df['symbol'].nunique():.0f}")


if __name__ == "__main__":
    main()
