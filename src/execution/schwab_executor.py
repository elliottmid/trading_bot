"""
schwab_executor.py — Live order execution via the Schwab API.

Submits DAY limit orders to Schwab using schwab-py and logs each execution.
Limit prices include a small slip buffer so orders are likely to fill
while still protecting against adverse market-order slippage.
All order calls are gated by the RiskManager.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import schwab
from schwab.orders.equities import equity_buy_limit, equity_sell_limit
from schwab.orders.common import Duration, Session

from ..config import Config
from ..data.schwab_fetcher import SchwabFetcher
from ..execution.risk_manager import RiskManager
from ..logger import get_logger

log = get_logger(__name__)

# Fraction added to buy price / subtracted from sell price so that
# limit orders fill quickly without paying full market-order slippage.
_SLIP_BUFFER = 0.002  # 0.20 %


def _limit_buy_price(last: float) -> float:
    """Return a limit price slightly above *last* to improve fill probability."""
    return round(last * (1 + _SLIP_BUFFER), 2)


def _limit_sell_price(last: float) -> float:
    """Return a limit price slightly below *last* to improve fill probability."""
    return round(last * (1 - _SLIP_BUFFER), 2)


class SchwabExecutor:
    """Submit and track live equity limit orders via Schwab.

    All orders are DAY limits placed in the NORMAL (RTH) session.
    Using limit orders rather than market orders prevents runaway
    slippage in fast or illiquid markets.

    Args:
        config: Application configuration.
        fetcher: Initialised SchwabFetcher (shares the authenticated client).
        risk_manager: RiskManager for trade logging.
    """

    def __init__(
        self,
        config: Config,
        fetcher: SchwabFetcher,
        risk_manager: RiskManager,
    ) -> None:
        self._config = config
        self._fetcher = fetcher
        self._risk = risk_manager

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def buy(self, symbol: str, qty: int, price: float) -> Optional[str]:
        """Submit a DAY limit buy order at *price* + slip buffer.

        Args:
            symbol: Ticker symbol to buy.
            qty: Number of shares.
            price: Last trade price used to anchor the limit.

        Returns:
            Order ID string if submission succeeded, None on failure.
        """
        if not self._config.live_trading:
            log.warning(
                "LIVE_TRADING=false — skipping real buy for %s x%d.", symbol, qty
            )
            return None

        limit_price = _limit_buy_price(price)
        account_hash = self._fetcher.get_account_hash()
        order = (
            equity_buy_limit(symbol, qty, limit_price)
            .set_duration(Duration.DAY)
            .set_session(Session.NORMAL)
            .build()
        )
        try:
            resp = self._fetcher._client.place_order(account_hash, order)
            resp.raise_for_status()
            order_id = resp.headers.get("Location", "unknown").split("/")[-1]
            log.info(
                "BUY limit order placed: %s x%d limit=$%.2f (last=$%.2f) | order_id=%s.",
                symbol,
                qty,
                limit_price,
                price,
                order_id,
            )
            self._risk.log_trade(
                symbol=symbol,
                side="BUY",
                qty=qty,
                price=limit_price,
                timestamp=datetime.utcnow(),
            )
            return order_id
        except Exception as exc:
            log.error("BUY order failed for %s: %s", symbol, exc)
            return None

    def sell(self, symbol: str, qty: int, price: float) -> Optional[str]:
        """Submit a DAY limit sell order at *price* - slip buffer.

        Args:
            symbol: Ticker symbol to sell.
            qty: Number of shares to sell.
            price: Last trade price used to anchor the limit.

        Returns:
            Order ID string if submission succeeded, None on failure.
        """
        if not self._config.live_trading:
            log.warning(
                "LIVE_TRADING=false — skipping real sell for %s x%d.", symbol, qty
            )
            return None

        limit_price = _limit_sell_price(price)
        account_hash = self._fetcher.get_account_hash()
        order = (
            equity_sell_limit(symbol, qty, limit_price)
            .set_duration(Duration.DAY)
            .set_session(Session.NORMAL)
            .build()
        )
        try:
            resp = self._fetcher._client.place_order(account_hash, order)
            resp.raise_for_status()
            order_id = resp.headers.get("Location", "unknown").split("/")[-1]
            log.info(
                "SELL limit order placed: %s x%d limit=$%.2f (last=$%.2f) | order_id=%s.",
                symbol,
                qty,
                limit_price,
                price,
                order_id,
            )
            self._risk.log_trade(
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=limit_price,
                timestamp=datetime.utcnow(),
            )
            return order_id
        except Exception as exc:
            log.error("SELL order failed for %s: %s", symbol, exc)
            return None
