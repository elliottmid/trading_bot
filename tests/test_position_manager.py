"""
test_position_manager.py — Unit tests for PositionManager.

Tests:
- add_position stores the position and persists to JSON.
- update_price updates current price and unrealised P&L.
- close_position returns correct realised P&L and removes the record.
- has_position returns the correct boolean.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.execution.position_manager import PositionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pm(tmp_path: Path) -> PositionManager:
    """Construct a PositionManager backed by a temp directory."""
    return PositionManager(path=str(tmp_path / "positions.json"))


def _add_spy(pm: PositionManager, qty: int = 10, entry_price: float = 500.0) -> None:
    """Add a standard SPY position to *pm*."""
    pm.add_position(
        symbol="SPY",
        qty=qty,
        entry_price=entry_price,
        entry_time=datetime(2026, 1, 15, 10, 30),
        stop_loss=entry_price * 0.98,
        take_profit=entry_price * 1.05,
    )


# ---------------------------------------------------------------------------
# add_position
# ---------------------------------------------------------------------------

class TestAddPosition:
    def test_position_is_stored(self, tmp_path):
        """After add_position, has_position should return True."""
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        assert pm.has_position("SPY") is True

    def test_position_persists_to_json(self, tmp_path):
        """Position data should be written to the backing JSON file."""
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        json_path = tmp_path / "positions.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "SPY" in data
        assert data["SPY"]["qty"] == 10
        assert data["SPY"]["entry_price"] == 500.0

    def test_position_fields_correct(self, tmp_path):
        """Position dict should contain all expected fields."""
        pm = _make_pm(tmp_path)
        _add_spy(pm, qty=5, entry_price=420.0)
        pos = pm.get_position("SPY")
        assert pos is not None
        assert pos["symbol"] == "SPY"
        assert pos["qty"] == 5
        assert pos["entry_price"] == 420.0
        assert pos["current_price"] == 420.0  # initialised to entry_price


# ---------------------------------------------------------------------------
# update_price + get_unrealized_pnl
# ---------------------------------------------------------------------------

class TestUpdatePrice:
    def test_unrealized_pnl_profit(self, tmp_path):
        """Rising price should produce positive unrealised P&L."""
        pm = _make_pm(tmp_path)
        _add_spy(pm, qty=10, entry_price=500.0)
        pm.update_price("SPY", 520.0)
        pnl = pm.get_unrealized_pnl("SPY")
        assert pnl == pytest.approx(200.0)  # (520-500) * 10

    def test_unrealized_pnl_loss(self, tmp_path):
        """Falling price should produce negative unrealised P&L."""
        pm = _make_pm(tmp_path)
        _add_spy(pm, qty=10, entry_price=500.0)
        pm.update_price("SPY", 480.0)
        pnl = pm.get_unrealized_pnl("SPY")
        assert pnl == pytest.approx(-200.0)  # (480-500) * 10

    def test_update_nonexistent_symbol_is_noop(self, tmp_path):
        """Updating price for unknown symbol should not raise."""
        pm = _make_pm(tmp_path)
        pm.update_price("AAPL", 200.0)  # should not raise

    def test_pnl_for_nonexistent_symbol_is_zero(self, tmp_path):
        """get_unrealized_pnl for unknown symbol should return 0.0."""
        pm = _make_pm(tmp_path)
        assert pm.get_unrealized_pnl("AAPL") == 0.0


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def test_realized_pnl_is_correct(self, tmp_path):
        """close_position should return (exit - entry) * qty."""
        pm = _make_pm(tmp_path)
        _add_spy(pm, qty=10, entry_price=500.0)
        pm.update_price("SPY", 525.0)
        pnl = pm.close_position("SPY")
        assert pnl == pytest.approx(250.0)  # (525-500) * 10

    def test_position_removed_after_close(self, tmp_path):
        """has_position should return False after close_position."""
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        pm.close_position("SPY")
        assert pm.has_position("SPY") is False

    def test_close_nonexistent_returns_zero(self, tmp_path):
        """Closing a symbol that was never opened should return 0.0."""
        pm = _make_pm(tmp_path)
        pnl = pm.close_position("AAPL")
        assert pnl == 0.0

    def test_json_updated_after_close(self, tmp_path):
        """JSON file should not contain the symbol after close."""
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        pm.close_position("SPY")
        data = json.loads((tmp_path / "positions.json").read_text())
        assert "SPY" not in data


# ---------------------------------------------------------------------------
# has_position / get_symbols / get_all_positions
# ---------------------------------------------------------------------------

class TestHasPosition:
    def test_returns_true_when_exists(self, tmp_path):
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        assert pm.has_position("SPY") is True

    def test_returns_false_when_absent(self, tmp_path):
        pm = _make_pm(tmp_path)
        assert pm.has_position("SPY") is False

    def test_get_symbols_returns_correct_list(self, tmp_path):
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        assert "SPY" in pm.get_symbols()

    def test_get_all_positions_length(self, tmp_path):
        pm = _make_pm(tmp_path)
        _add_spy(pm)
        assert len(pm.get_all_positions()) == 1


# ---------------------------------------------------------------------------
# Persistence across instances
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_positions_survive_restart(self, tmp_path):
        """Positions written by one instance should be readable by another."""
        pm1 = _make_pm(tmp_path)
        _add_spy(pm1, qty=7, entry_price=450.0)

        pm2 = _make_pm(tmp_path)  # New instance, same file
        assert pm2.has_position("SPY") is True
        assert pm2.get_position("SPY")["qty"] == 7
