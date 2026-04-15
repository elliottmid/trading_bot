"""
fred_fetcher.py — Lightweight FRED (Federal Reserve Economic Data) fetcher.

Uses the public FRED REST API (no API key required for basic series).
Provides macro-economic series that can serve as additional features
for the trading model (e.g., VIX, yield spreads, CPI).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from ..logger import get_logger

log = get_logger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Common FRED series IDs
SERIES_VIX = "VIXCLS"          # CBOE Volatility Index
SERIES_FED_FUNDS = "FEDFUNDS"  # Effective Federal Funds Rate
SERIES_T10Y2Y = "T10Y2Y"       # 10-Year minus 2-Year Treasury Spread
SERIES_CPI = "CPIAUCSL"        # Consumer Price Index


class FredFetcher:
    """Fetch economic time-series data from the FRED public API.

    Args:
        api_key: Optional FRED API key.  If None the public (anonymous)
            endpoint is used, which has lower rate limits.
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """Download a single FRED data series.

        Args:
            series_id: FRED series identifier, e.g. ``"VIXCLS"``.
            start_date: First observation date (inclusive). Defaults to
                365 days before today.
            end_date: Last observation date (inclusive). Defaults to today.

        Returns:
            A pandas Series with a DatetimeIndex and float values.
            Missing observations are forward-filled.
        """
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        params: dict = {
            "series_id": series_id,
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
            "file_type": "json",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        log.debug("Fetching FRED series %s from %s to %s.", series_id, start_date, end_date)
        try:
            resp = requests.get(_FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.error("FRED request failed for series %s: %s", series_id, exc)
            raise

        observations = resp.json().get("observations", [])
        records = []
        for obs in observations:
            try:
                val = float(obs["value"])
            except (ValueError, KeyError):
                val = float("nan")
            records.append({"date": obs["date"], "value": val})

        if not records:
            log.warning("No observations returned for FRED series %s.", series_id)
            return pd.Series(dtype=float, name=series_id)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        series = df.set_index("date")["value"].rename(series_id)
        series = series.ffill()
        log.info(
            "Fetched %d observations for FRED series %s.", len(series), series_id
        )
        return series

    def fetch_vix(self, **kwargs) -> pd.Series:
        """Convenience method to fetch the CBOE VIX series."""
        return self.fetch_series(SERIES_VIX, **kwargs)

    def fetch_yield_spread(self, **kwargs) -> pd.Series:
        """Convenience method to fetch the 10Y-2Y Treasury spread."""
        return self.fetch_series(SERIES_T10Y2Y, **kwargs)
