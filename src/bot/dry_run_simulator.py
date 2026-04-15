"""
dry_run_simulator.py — Paper-trading simulator with the same interface as SchwabExecutor.

Logs all simulated orders to a CSV file instead of submitting real orders.
Drop-in replacement for SchwabExecutor when DRY_RUN=true.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Config
from ..logger import get_logger

log = get_logger(__name__)

_CSV_HEADER = [
    "timestamp",
    "action",
    "symbol",
    "qty",
    "price",
    "estimated_value",
]


class DryRunSimulator:
    """Simulates order execution by writing to a CSV log.

    Implements the same ``buy`` / ``sell`` interface as ``SchwabExecutor``
    so it can be swapped in transparently.

    Args:
        config: Application configuration (reads dry_run_trades_path).
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._csv_path = Path(config.dry_run_trades_path)
        self._ensure_csv()

    # ------------------------------------------------------------------
    # Order interface (mirrors SchwabExecutor)
    # ------------------------------------------------------------------

    def buy(self, symbol: str, qty: int, price: float) -> Optional[str]:
        """Simulate a limit buy order (mirrors SchwabExecutor.buy).

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            price: Anchoring price (limit = price * 1.002 in live execution).

        Returns:
            A synthetic order ID string (timestamp-based).
        """
        order_id = self._record("BUY", symbol, qty, price)
        log.info(
            "[DRY RUN] BUY %s x%d @ $%.2f | order_id=%s.",
            symbol,
            qty,
            price,
            order_id,
        )
        return order_id

    def sell(self, symbol: str, qty: int, price: float) -> Optional[str]:
        """Simulate a limit sell order (mirrors SchwabExecutor.sell).

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            price: Anchoring price (limit = price * 0.998 in live execution).

        Returns:
            A synthetic order ID string (timestamp-based).
        """
        order_id = self._record("SELL", symbol, qty, price)
        log.info(
            "[DRY RUN] SELL %s x%d @ $%.2f | order_id=%s.",
            symbol,
            qty,
            price,
            order_id,
        )
        return order_id

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_trade_history(self):
        """Return all simulated trades as a list of dicts.

        Returns:
            List of trade record dicts read from the CSV log.
        """
        import csv as _csv

        if not self._csv_path.exists():
            return []
        with self._csv_path.open("r", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            return list(reader)

    def print_summary(self) -> None:
        """Print a summary of all dry-run trades to stdout."""
        trades = self.get_trade_history()
        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        print("=" * 50)
        print("DRY RUN SUMMARY".center(50))
        print("=" * 50)
        print("  Total orders: %d (buys=%d, sells=%d)" % (len(trades), len(buys), len(sells)))
        print("  Log file: %s" % self._csv_path)
        print("=" * 50)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record(self, action: str, symbol: str, qty: int, price: float) -> str:
        """Append a trade record to the CSV and return a synthetic order ID."""
        timestamp = datetime.utcnow()
        order_id = "DRY-%s" % timestamp.strftime("%Y%m%d%H%M%S%f")
        estimated_value = qty * price

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            writer.writerow(
                {
                    "timestamp": timestamp.isoformat(),
                    "action": action,
                    "symbol": symbol,
                    "qty": qty,
                    "price": price,
                    "estimated_value": "%.2f" % estimated_value,
                }
            )
        return order_id

    def _ensure_csv(self) -> None:
        """Create the CSV with headers if it does not already exist."""
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
                writer.writeheader()
            log.info("Dry-run trade log initialised at %s.", self._csv_path)
