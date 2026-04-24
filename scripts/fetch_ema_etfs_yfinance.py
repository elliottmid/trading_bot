#!/usr/bin/env python3
"""
Fetch daily OHLCV bars for EMA backtest universe from Yahoo Finance.
Stores as parquet for efficient reloading.
No API keys needed.
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import yfinance as yf

from ema_etf_universe import PRIMARY_UNIVERSE, UNIVERSE

# Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_etf_data_yfinance(universe, start_date="2020-12-14", end_date=None):
    """
    Fetch daily bars for all ETFs in universe from Yahoo Finance.

    Args:
        universe: dict of {ticker: description}
        start_date: ISO date string or "YYYY-MM-DD"
        end_date: ISO date string or None (defaults to today)

    Returns:
        DataFrame with symbol and timestamp columns
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    symbols = list(universe.keys())
    logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date} via Yahoo Finance")

    all_dfs = []

    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"  [{i}/{len(symbols)}] {symbol}...")
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
            )

            if df.empty:
                logger.warning(f"    ✗ No data returned for {symbol}")
                continue

            # Reset index to have date as column
            df = df.reset_index()

            # Handle column names (yfinance returns them as is)
            # Convert all column names to lowercase strings
            df.columns = [str(col).lower() if not isinstance(col, tuple) else col[0].lower() for col in df.columns]

            # Rename date column to timestamp
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})

            # Use adjusted close if available, otherwise close
            if 'adj close' in df.columns:
                df['close'] = df['adj close']
            elif 'close' not in df.columns and 'adj close' in df.columns:
                df['close'] = df['adj close']

            # Add symbol column
            df['symbol'] = symbol

            # Keep only relevant columns (select those that exist)
            cols_to_keep = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[available_cols]

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            all_dfs.append(df)
            logger.info(f"    ✓ Fetched {len(df)} bars")

        except Exception as e:
            logger.warning(f"    ✗ Failed to fetch {symbol}: {e}")
            continue

    if not all_dfs:
        logger.error("Failed to fetch data for any symbols")
        raise ValueError("No data fetched from Yahoo Finance")

    result_df = pd.concat(all_dfs, ignore_index=True)
    result_df = result_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

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

    parser = argparse.ArgumentParser(description="Fetch EMA ETF data from Yahoo Finance")
    parser.add_argument("--universe", choices=["primary", "full"], default="primary",
                        help="Which universe to fetch (primary=13 ETFs, full=30+)")
    parser.add_argument("--start-date", default="2020-12-14",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None,
                        help="End date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    universe = PRIMARY_UNIVERSE if args.universe == "primary" else UNIVERSE

    logger.info(f"Using {args.universe} universe ({len(universe)} symbols)")

    df = fetch_etf_data_yfinance(universe, start_date=args.start_date, end_date=args.end_date)

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
