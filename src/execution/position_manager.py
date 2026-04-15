"""
position_manager.py — Local position ledger backed by a JSON file.

Tracks open positions in memory and persists them to disk after every
mutation so state survives restarts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logger import get_logger

log = get_logger(__name__)


class PositionManager:
    """Persistent, file-backed position ledger.

    Positions are stored as a JSON file at *path*.  Every mutating method
    saves the file immediately so state is not lost on crash or restart.

    Args:
        path: Filesystem path for the JSON position store.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_position(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        entry_time: datetime,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Record a new open position.

        Args:
            symbol: Ticker symbol (e.g. ``"SPY"``).
            qty: Number of shares purchased.
            entry_price: Price paid per share.
            entry_time: Timestamp of the entry.
            stop_loss: Absolute price at which to stop-loss exit.
            take_profit: Absolute price at which to take-profit exit.
        """
        if symbol in self._positions:
            log.warning(
                "Position for %s already exists; overwriting.", symbol
            )

        self._positions[symbol] = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "entry_time": entry_time.isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "current_price": entry_price,
        }
        self._save()
        log.info(
            "Position added: %s x%d @ $%.2f (SL=%.2f, TP=%.2f).",
            symbol,
            qty,
            entry_price,
            stop_loss,
            take_profit,
        )

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update the latest market price for an open position.

        Args:
            symbol: Ticker symbol to update.
            current_price: Latest market price per share.
        """
        if symbol not in self._positions:
            log.debug(
                "update_price called for %s but no open position found.", symbol
            )
            return
        self._positions[symbol]["current_price"] = current_price
        self._save()

    def get_unrealized_pnl(self, symbol: str) -> float:
        """Calculate unrealised P&L for an open position.

        Args:
            symbol: Ticker symbol.

        Returns:
            Unrealised P&L in dollars.  Returns 0.0 if position not found.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        return (pos["current_price"] - pos["entry_price"]) * pos["qty"]

    def close_position(self, symbol: str) -> float:
        """Remove a position from the ledger and return realised P&L.

        Args:
            symbol: Ticker symbol of the position to close.

        Returns:
            Realised P&L in dollars (current_price - entry_price) * qty.
            Returns 0.0 if the position did not exist.
        """
        pos = self._positions.pop(symbol, None)
        if pos is None:
            log.warning(
                "close_position called for %s but no position found.", symbol
            )
            return 0.0

        pnl = (pos["current_price"] - pos["entry_price"]) * pos["qty"]
        self._save()
        log.info(
            "Position closed: %s x%d | Entry=%.2f Exit=%.2f | P&L=$%.2f.",
            symbol,
            pos["qty"],
            pos["entry_price"],
            pos["current_price"],
            pnl,
        )
        return pnl

    def has_position(self, symbol: str) -> bool:
        """Check whether there is an open position for *symbol*.

        Args:
            symbol: Ticker symbol to query.

        Returns:
            True if an open position exists, False otherwise.
        """
        return symbol in self._positions

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Return a list of all open positions.

        Returns:
            List of position dicts.  Dicts contain: symbol, qty,
            entry_price, entry_time, stop_loss, take_profit, current_price.
        """
        return list(self._positions.values())

    def get_symbols(self) -> List[str]:
        """Return the list of symbols with open positions.

        Returns:
            List of ticker symbol strings.
        """
        return list(self._positions.keys())

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return the position dict for *symbol*, or None.

        Args:
            symbol: Ticker symbol to look up.

        Returns:
            Position dict or None.
        """
        return self._positions.get(symbol)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist the current positions to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(self._positions, fh, indent=2)

    def _load(self) -> None:
        """Load positions from the JSON file (if it exists)."""
        if not self._path.exists():
            log.debug("Position store not found at %s; starting fresh.", self._path)
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                self._positions = json.load(fh)
            log.info(
                "Loaded %d open position(s) from %s.",
                len(self._positions),
                self._path,
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.error(
                "Failed to load positions from %s: %s. Starting fresh.",
                self._path,
                exc,
            )
            self._positions = {}
