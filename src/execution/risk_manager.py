"""
risk_manager.py — Pre-trade risk checks and position-sizing.

Enforces the PDT (Pattern Day Trader) rule, daily loss limits, and
provides a confidence-scaled position-size calculator.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd

from ..config import Config
from ..data.schwab_fetcher import SchwabFetcher
from ..logger import get_logger

log = get_logger(__name__)

_TRADES_CSV_HEADER = ["timestamp", "symbol", "side", "qty", "price"]


class RiskManager:
    """Enforces pre-trade risk rules before order submission.

    Args:
        config: Application configuration.
        fetcher: SchwabFetcher instance used for market-hours checks.
    """

    def __init__(self, config: Config, fetcher: SchwabFetcher) -> None:
        self._config = config
        self._fetcher = fetcher
        self._trades_path = Path(config.trades_log_path)
        self._day_start_equity: float = 0.0
        self._ensure_trades_csv()

    # ------------------------------------------------------------------
    # Public gate checks
    # ------------------------------------------------------------------

    def can_trade(self) -> Tuple[bool, str]:
        """Run all pre-trade checks in order.

        Returns:
            Tuple of (allowed: bool, reason: str).  If *allowed* is False,
            *reason* explains why trading is blocked.
        """
        if not self.is_market_open():
            return False, "Market is closed."

        if not self.check_pdt_rule():
            return False, "PDT rule: 4 or more round-trips in the last 5 business days."

        return True, "OK"

    def check_pdt_rule(self) -> bool:
        """Check the Pattern Day Trader (PDT) rule.

        Counts round-trip trades (one buy + one sell = one round-trip) in the
        last 5 business days.  Accounts with fewer than $25k margin equity
        are limited to 3 round-trips per rolling 5-day window.

        Returns:
            True if trading is allowed (fewer than 4 round-trips), else False.
        """
        trades = self._load_recent_trades(business_days=5)
        # Count sell-side entries as round-trip completions
        round_trips = sum(1 for t in trades if t.get("side", "").upper() == "SELL")
        log.debug("PDT check: %d round-trips in last 5 biz days.", round_trips)
        return round_trips < 4

    def check_daily_loss(self, current_equity: float) -> bool:
        """Check whether the daily loss limit has been breached.

        Args:
            current_equity: Current account equity in USD.

        Returns:
            True if within limit (trading allowed), False if limit breached.
        """
        if self._day_start_equity <= 0:
            # Initialise on first call
            self._day_start_equity = current_equity
            return True

        loss_pct = (
            (self._day_start_equity - current_equity) / self._day_start_equity * 100
        )
        limit = self._config.max_daily_loss_pct
        log.debug(
            "Daily loss check: %.2f%% lost vs %.2f%% limit.", loss_pct, limit
        )
        if loss_pct >= limit:
            log.warning(
                "Daily loss limit reached: %.2f%% >= %.2f%%. Halting trading.",
                loss_pct,
                limit,
            )
            return False
        return True

    def is_market_open(self) -> bool:
        """Check whether the US equity market is currently open.

        Returns:
            True if the market is open, False otherwise.
        """
        try:
            return self._fetcher.is_market_open()
        except Exception as exc:
            log.error("Market hours check failed: %s. Assuming closed.", exc)
            return False

    def set_day_start_equity(self, equity: float) -> None:
        """Set the equity reference point for daily loss tracking.

        Call this once at the start of each trading day.

        Args:
            equity: Account equity at the start of the day.
        """
        self._day_start_equity = equity
        log.info("Day-start equity set to $%.2f.", equity)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def get_position_size(
        self, account_equity: float, confidence: float, price: float
    ) -> int:
        """Calculate the number of shares to buy.

        Uses a confidence-scaled fraction of account equity:
            dollar_amount = equity * max_position_size_pct * confidence
            shares = dollar_amount / price

        Args:
            account_equity: Total account equity in USD.
            confidence: Model confidence score (0.0–1.0).
            price: Current share price.

        Returns:
            Number of whole shares to purchase (minimum 1).
        """
        if price <= 0:
            return 0
        dollar_amount = (
            account_equity * self._config.max_position_size_pct * confidence
        )
        shares = int(dollar_amount / price)
        shares = max(1, shares)
        log.debug(
            "Position size: equity=$%.0f conf=%.2f price=$%.2f -> %d shares.",
            account_equity,
            confidence,
            price,
            shares,
        )
        return shares

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a trade record to the CSV trade log.

        Args:
            symbol: Ticker symbol.
            side: ``"BUY"`` or ``"SELL"``.
            qty: Number of shares.
            price: Execution price per share.
            timestamp: Execution timestamp (defaults to ``datetime.utcnow()``).
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self._trades_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trades_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_TRADES_CSV_HEADER)
            writer.writerow(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "side": side.upper(),
                    "qty": qty,
                    "price": price,
                }
            )
        log.debug("Trade logged: %s %s x%d @ $%.2f.", side, symbol, qty, price)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_trades_csv(self) -> None:
        """Create the trades CSV with headers if it does not already exist."""
        self._trades_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._trades_path.exists():
            with self._trades_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_TRADES_CSV_HEADER)
                writer.writeheader()

    def _load_recent_trades(self, business_days: int = 5) -> list:
        """Load trades from the CSV that fall within the last N business days.

        Args:
            business_days: Look-back window in business days.

        Returns:
            List of trade dicts.
        """
        if not self._trades_path.exists():
            return []

        cutoff = datetime.utcnow() - timedelta(days=business_days * 2)
        biz_day_count = 0
        # Use a business-day count approximation
        today = datetime.utcnow().date()
        biz_dates = set()
        d = today
        while biz_day_count < business_days:
            if d.weekday() < 5:  # Mon–Fri
                biz_dates.add(d)
                biz_day_count += 1
            d -= timedelta(days=1)

        try:
            df = pd.read_csv(self._trades_path, parse_dates=["timestamp"])
            df = df[df["timestamp"].dt.date.isin(biz_dates)]
            return df.to_dict("records")
        except Exception as exc:
            log.warning("Could not read trades CSV: %s", exc)
            return []
