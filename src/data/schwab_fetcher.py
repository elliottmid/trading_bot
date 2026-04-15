"""
schwab_fetcher.py — Wrapper around the schwab-py client for market data.

Provides a high-level interface for fetching price history, quotes, account
balances, positions, and market status.  All network calls include
exponential-backoff retry logic for HTTP 429 (rate-limit) responses.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import schwab

from ..config import Config
from ..logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def _retry_on_rate_limit(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    backoff: float = 2.0,
):
    """Decorator that retries a method on HTTP 429 / transient errors.

    Args:
        max_attempts: Maximum number of total attempts.
        base_delay: Initial wait in seconds before the first retry.
        backoff: Multiplicative factor applied to the delay after each retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    # schwab-py raises httpx.HTTPStatusError for 4xx/5xx
                    is_rate_limit = (
                        "429" in str(exc)
                        or "rate" in str(exc).lower()
                        or "too many" in str(exc).lower()
                    )
                    if not is_rate_limit or attempt == max_attempts:
                        raise
                    log.warning(
                        "Rate-limited (attempt %d/%d). Retrying in %.1fs.",
                        attempt,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# SchwabFetcher
# ---------------------------------------------------------------------------

class SchwabFetcher:
    """High-level Schwab API client.

    Args:
        config: Application configuration instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client: Optional[schwab.client.Client] = None
        self._account_hash: Optional[str] = None
        self._client = self._build_client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> schwab.client.Client:
        """Initialise the schwab-py client from the saved token file."""
        try:
            client = schwab.auth.client_from_token_file(
                token_path=self._config.schwab_token_path,
                api_key=self._config.schwab_api_key,
                app_secret=self._config.schwab_api_secret,
            )
            log.info("Schwab client initialised from token file.")
            return client
        except FileNotFoundError:
            log.error(
                "Token file not found at %s. "
                "Run scripts/auth_setup.py to authenticate.",
                self._config.schwab_token_path,
            )
            raise
        except Exception as exc:
            log.error("Failed to build Schwab client: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_retry_on_rate_limit()
    def fetch_price_history(
        self, symbol: str, days: int = 252
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for *symbol*.

        Args:
            symbol: Ticker symbol (e.g. ``"SPY"``).
            days: Number of calendar days of history to request.

        Returns:
            DataFrame with columns ``[open, high, low, close, volume]``
            indexed by ``datetime`` (timezone-aware).
        """
        log.debug("Fetching %d days of price history for %s.", days, symbol)
        resp = self._client.get_price_history(
            symbol=symbol,
            period_type=schwab.client.Client.PriceHistory.PeriodType.YEAR,
            period=schwab.client.Client.PriceHistory.Period.ONE_YEAR,
            frequency_type=schwab.client.Client.PriceHistory.FrequencyType.DAILY,
            frequency=schwab.client.Client.PriceHistory.Frequency.DAILY,
            need_extended_hours_data=False,
        )
        resp.raise_for_status()
        data = resp.json()

        candles = data.get("candles", [])
        if not candles:
            log.warning("No candles returned for %s.", symbol)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        df = df[["open", "high", "low", "close", "volume"]]
        log.info(
            "Fetched %d bars for %s (from %s to %s).",
            len(df),
            symbol,
            df.index[0].date(),
            df.index[-1].date(),
        )
        return df

    @_retry_on_rate_limit()
    def fetch_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch real-time quotes for a list of symbols.

        Args:
            symbols: List of ticker symbols.

        Returns:
            Dict mapping each symbol to its raw quote payload.
        """
        log.debug("Fetching quotes for: %s", symbols)
        resp = self._client.get_quotes(symbols)
        resp.raise_for_status()
        data = resp.json()
        return data

    @_retry_on_rate_limit()
    def fetch_account_balance(self) -> float:
        """Return total account equity (liquidation value) in USD.

        Returns:
            Account equity as a float.
        """
        account_hash = self.get_account_hash()
        resp = self._client.get_account(
            account_hash=account_hash,
            fields=[schwab.client.Client.Account.Fields.POSITIONS],
        )
        resp.raise_for_status()
        data = resp.json()
        equity = (
            data.get("securitiesAccount", {})
            .get("currentBalances", {})
            .get("liquidationValue", 0.0)
        )
        log.debug("Account equity: $%.2f", equity)
        return float(equity)

    @_retry_on_rate_limit()
    def fetch_positions(self) -> List[Dict[str, Any]]:
        """Return a list of current open positions from the brokerage.

        Returns:
            List of position dicts as returned by the Schwab API.
        """
        account_hash = self.get_account_hash()
        resp = self._client.get_account(
            account_hash=account_hash,
            fields=[schwab.client.Client.Account.Fields.POSITIONS],
        )
        resp.raise_for_status()
        data = resp.json()
        positions = (
            data.get("securitiesAccount", {}).get("positions", [])
        )
        log.debug("Fetched %d open positions.", len(positions))
        return positions

    @_retry_on_rate_limit()
    def is_market_open(self) -> bool:
        """Check whether the US equity market is currently open.

        Returns:
            True if the market is open, False otherwise.
        """
        resp = self._client.get_markets(
            markets=[schwab.client.Client.Markets.EQUITY]
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            is_open = data["equity"]["EQ"]["isOpen"]
            return bool(is_open)
        except (KeyError, TypeError):
            log.warning(
                "Could not parse market-hours response; assuming closed."
            )
            return False

    @_retry_on_rate_limit()
    def get_account_hash(self) -> str:
        """Fetch and cache the hashed account number.

        Returns:
            The encrypted account hash string required by most Schwab endpoints.
        """
        if self._account_hash:
            return self._account_hash

        resp = self._client.get_account_numbers()
        resp.raise_for_status()
        accounts = resp.json()
        if not accounts:
            raise RuntimeError("No accounts found on this Schwab login.")
        self._account_hash = accounts[0]["hashValue"]
        log.info("Account hash cached (truncated): ...%s", self._account_hash[-6:])
        return self._account_hash
