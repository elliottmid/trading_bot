"""
test_schwab_fetcher.py — Unit tests for SchwabFetcher.

All Schwab API calls are mocked; no network access required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.schwab_fetcher import SchwabFetcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> Config:
    config = Config()
    config.schwab_api_key = "test_key"
    config.schwab_api_secret = "test_secret"
    config.schwab_token_path = str(tmp_path / "token.json")
    return config


def _make_mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx/requests Response object."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _make_fetcher(tmp_path: Path) -> tuple:
    """Build a SchwabFetcher with a mocked client.

    Returns:
        (fetcher, mock_client)
    """
    config = _make_config(tmp_path)
    with patch("schwab.auth.client_from_token_file") as mock_auth:
        mock_client = MagicMock()
        mock_auth.return_value = mock_client
        # Create token file so the fetcher doesn't raise FileNotFoundError
        Path(config.schwab_token_path).write_text("{}")
        fetcher = SchwabFetcher(config=config)
        fetcher._client = mock_client
        return fetcher, mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchPriceHistory:
    def test_returns_dataframe_with_ohlcv(self, tmp_path):
        """fetch_price_history should return a DataFrame with OHLCV columns."""
        fetcher, mock_client = _make_fetcher(tmp_path)

        candles = [
            {
                "datetime": 1704067200000,  # 2024-01-01 00:00:00 UTC in ms
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1_000_000,
            },
            {
                "datetime": 1704153600000,
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 1_100_000,
            },
        ]
        mock_client.get_price_history.return_value = _make_mock_response(
            {"candles": candles, "symbol": "SPY"}
        )

        df = fetcher.fetch_price_history("SPY", days=2)
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2

    def test_empty_candles_returns_empty_dataframe(self, tmp_path):
        """An empty candles list should return an empty DataFrame."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        mock_client.get_price_history.return_value = _make_mock_response(
            {"candles": [], "symbol": "SPY"}
        )
        df = fetcher.fetch_price_history("SPY", days=30)
        assert df.empty


class TestFetchQuotes:
    def test_returns_dict(self, tmp_path):
        """fetch_quotes should return the raw JSON dict from the API."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        payload = {"SPY": {"quote": {"lastPrice": 520.5}}}
        mock_client.get_quotes.return_value = _make_mock_response(payload)
        result = fetcher.fetch_quotes(["SPY"])
        assert "SPY" in result
        assert result["SPY"]["quote"]["lastPrice"] == 520.5


class TestIsMarketOpen:
    def test_open_true(self, tmp_path):
        """is_market_open returns True when API says open."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        mock_client.get_markets.return_value = _make_mock_response(
            {"equity": {"EQ": {"isOpen": True}}}
        )
        assert fetcher.is_market_open() is True

    def test_open_false(self, tmp_path):
        """is_market_open returns False when API says closed."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        mock_client.get_markets.return_value = _make_mock_response(
            {"equity": {"EQ": {"isOpen": False}}}
        )
        assert fetcher.is_market_open() is False

    def test_malformed_response_returns_false(self, tmp_path):
        """Malformed API response should safely return False."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        mock_client.get_markets.return_value = _make_mock_response({})
        assert fetcher.is_market_open() is False


class TestGetAccountHash:
    def test_caches_hash_on_second_call(self, tmp_path):
        """get_account_hash should cache and avoid a second API call."""
        fetcher, mock_client = _make_fetcher(tmp_path)
        mock_client.get_account_numbers.return_value = _make_mock_response(
            [{"hashValue": "abc123xyz"}]
        )
        hash1 = fetcher.get_account_hash()
        hash2 = fetcher.get_account_hash()
        assert hash1 == hash2 == "abc123xyz"
        # Should only have been called once
        mock_client.get_account_numbers.assert_called_once()
