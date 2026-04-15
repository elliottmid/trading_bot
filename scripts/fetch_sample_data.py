#!/usr/bin/env python3
"""
fetch_sample_data.py — Download OHLCV data and save to data/raw/{SYMBOL}.csv

Usage:
    python scripts/fetch_sample_data.py SPY 2025-01-01 2026-04-12
    python scripts/fetch_sample_data.py QQQ              # uses defaults
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.schwab_fetcher import SchwabFetcher
from src.logger import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV price history for a symbol and save to CSV."
    )
    parser.add_argument("symbol", nargs="?", default="SPY", help="Ticker symbol.")
    parser.add_argument(
        "start_date",
        nargs="?",
        default=None,
        help="Start date YYYY-MM-DD (informational only; API uses 'days' param).",
    )
    parser.add_argument(
        "end_date",
        nargs="?",
        default=None,
        help="End date YYYY-MM-DD (informational only).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Number of calendar days of history to fetch (default: 252).",
    )
    parser.add_argument("--env", default=None, help="Path to .env file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()

    config = Config.from_env(dotenv_path=args.env)
    config.ensure_dirs()

    log.info("Fetching %d days of data for %s.", args.days, symbol)

    try:
        fetcher = SchwabFetcher(config=config)
    except Exception as exc:
        print(
            "ERROR: Could not initialise Schwab client: %s\n"
            "Run 'python scripts/auth_setup.py' first." % exc
        )
        sys.exit(1)

    try:
        df = fetcher.fetch_price_history(symbol=symbol, days=args.days)
    except Exception as exc:
        print("ERROR: Data fetch failed: %s" % exc)
        sys.exit(1)

    if df.empty:
        print("No data returned for %s." % symbol)
        sys.exit(1)

    # Save to CSV
    out_path = Path("data/raw/%s.csv" % symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print("Saved %d rows to %s" % (len(df), out_path))

    # Print summary stats
    print("\n--- Summary for %s ---" % symbol)
    print("  Rows:          %d" % len(df))
    print("  Date range:    %s  to  %s" % (df.index[0].date(), df.index[-1].date()))
    print("  Close (last):  $%.2f" % df["close"].iloc[-1])
    print("  Close min/max: $%.2f / $%.2f" % (df["close"].min(), df["close"].max()))
    print("  Volume avg:    {:,.0f}".format(df["volume"].mean()))
    print(
        "  Daily return:  mean=%.3f%%  std=%.3f%%"
        % (
            df["close"].pct_change().mean() * 100,
            df["close"].pct_change().std() * 100,
        )
    )


if __name__ == "__main__":
    main()
