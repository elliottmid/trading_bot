"""
test_risk_manager.py — Unit tests for RiskManager.

Tests:
- PDT rule allows 3 round-trips, blocks at 4.
- Daily loss limit allows trading within limit, blocks when exceeded.
- Market hours check delegates to fetcher correctly.
- Position sizing scales with confidence.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.execution.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> Config:
    """Return a Config with paths pointing at tmp_path."""
    config = Config()
    config.trades_log_path = str(tmp_path / "trades.csv")
    config.max_daily_loss_pct = 2.0
    config.max_position_size_pct = 0.05
    config.stop_loss_pct = 2.0
    config.take_profit_pct = 5.0
    return config


def _make_fetcher(market_open: bool = True) -> MagicMock:
    """Return a mock SchwabFetcher."""
    fetcher = MagicMock()
    fetcher.is_market_open.return_value = market_open
    return fetcher


def _write_sells(trades_path: Path, n: int) -> None:
    """Write *n* SELL rows to the trades CSV with today's timestamp."""
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    with trades_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["timestamp", "symbol", "side", "qty", "price"]
        )
        writer.writeheader()
        today = datetime.utcnow()
        for i in range(n):
            writer.writerow(
                {
                    "timestamp": today.isoformat(),
                    "symbol": "SPY",
                    "side": "SELL",
                    "qty": 10,
                    "price": 500.0,
                }
            )


# ---------------------------------------------------------------------------
# PDT tests
# ---------------------------------------------------------------------------

class TestPdtRule:
    def test_three_sells_allowed(self, tmp_path):
        """3 round-trips in 5 days should be allowed."""
        config = _make_config(tmp_path)
        _write_sells(Path(config.trades_log_path), 3)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        assert rm.check_pdt_rule() is True

    def test_four_sells_blocked(self, tmp_path):
        """4 round-trips in 5 days should be blocked."""
        config = _make_config(tmp_path)
        _write_sells(Path(config.trades_log_path), 4)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        assert rm.check_pdt_rule() is False

    def test_zero_sells_allowed(self, tmp_path):
        """No trades should pass PDT check."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        assert rm.check_pdt_rule() is True


# ---------------------------------------------------------------------------
# Daily loss tests
# ---------------------------------------------------------------------------

class TestDailyLoss:
    def test_within_limit_allowed(self, tmp_path):
        """A 1% loss on a 2% limit should be allowed."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        rm.set_day_start_equity(100_000)
        # 1% loss
        assert rm.check_daily_loss(99_000) is True

    def test_at_limit_blocked(self, tmp_path):
        """Exactly 2% loss should block trading."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        rm.set_day_start_equity(100_000)
        # Exactly 2% loss — limit is >=
        assert rm.check_daily_loss(98_000) is False

    def test_exceeded_limit_blocked(self, tmp_path):
        """A 3% loss on a 2% limit should be blocked."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        rm.set_day_start_equity(100_000)
        assert rm.check_daily_loss(97_000) is False

    def test_no_start_equity_allows(self, tmp_path):
        """If day-start equity was never set, first call should return True."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        # _day_start_equity is 0.0 initially
        assert rm.check_daily_loss(50_000) is True


# ---------------------------------------------------------------------------
# Market hours tests
# ---------------------------------------------------------------------------

class TestMarketHours:
    def test_open_returns_true(self, tmp_path):
        """is_market_open should return True when fetcher says open."""
        config = _make_config(tmp_path)
        fetcher = _make_fetcher(market_open=True)
        rm = RiskManager(config=config, fetcher=fetcher)
        assert rm.is_market_open() is True

    def test_closed_returns_false(self, tmp_path):
        """is_market_open should return False when fetcher says closed."""
        config = _make_config(tmp_path)
        fetcher = _make_fetcher(market_open=False)
        rm = RiskManager(config=config, fetcher=fetcher)
        assert rm.is_market_open() is False

    def test_fetcher_exception_returns_false(self, tmp_path):
        """If fetcher raises, is_market_open should return False safely."""
        config = _make_config(tmp_path)
        fetcher = _make_fetcher()
        fetcher.is_market_open.side_effect = RuntimeError("API down")
        rm = RiskManager(config=config, fetcher=fetcher)
        assert rm.is_market_open() is False


# ---------------------------------------------------------------------------
# Position sizing tests
# ---------------------------------------------------------------------------

class TestPositionSizing:
    def test_basic_sizing(self, tmp_path):
        """Position size should scale with equity and price."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        # 100k equity, 5% max, 100% confidence, $100 price
        # expected: 100000 * 0.05 * 1.0 / 100 = 50 shares
        qty = rm.get_position_size(100_000, 1.0, 100.0)
        assert qty == 50

    def test_confidence_scaling(self, tmp_path):
        """Lower confidence should reduce position size."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        qty_full = rm.get_position_size(100_000, 1.0, 100.0)
        qty_half = rm.get_position_size(100_000, 0.5, 100.0)
        assert qty_half < qty_full

    def test_minimum_one_share(self, tmp_path):
        """Position size should always be at least 1 share."""
        config = _make_config(tmp_path)
        rm = RiskManager(config=config, fetcher=_make_fetcher())
        # Very low confidence, high price
        qty = rm.get_position_size(1_000, 0.01, 10_000.0)
        assert qty >= 1
