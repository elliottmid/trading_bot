"""
preprocessor.py — Data cleaning and normalisation utilities.

Used between the raw OHLCV fetcher and the feature engineering step to
ensure the DataFrame is in a consistent, model-ready state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..logger import get_logger

log = get_logger(__name__)


class Preprocessor:
    """Clean and normalise raw OHLCV DataFrames.

    Typical usage::

        pp = Preprocessor()
        clean_df = pp.clean(raw_df)
    """

    # Required OHLCV columns
    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline.

        Steps:
        1. Validate required columns are present.
        2. Drop duplicate index entries, keeping the last.
        3. Sort by datetime index ascending.
        4. Remove rows with non-positive close price.
        5. Forward-fill NaN values (max 3 consecutive), then drop remaining.

        Args:
            df: Raw OHLCV DataFrame with DatetimeIndex.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        self._validate_columns(df)

        # Drop duplicate timestamps
        n_before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < n_before:
            log.debug("Dropped %d duplicate rows.", n_before - len(df))

        df = df.sort_index()

        # Remove bad close prices
        bad_close = df["close"] <= 0
        if bad_close.any():
            log.warning("Dropping %d rows with non-positive close.", bad_close.sum())
            df = df[~bad_close]

        # Forward-fill short gaps then drop remaining NaN
        df = df.ffill(limit=3)
        n_nan = df.isnull().any(axis=1).sum()
        if n_nan > 0:
            log.debug("Dropping %d rows with residual NaN after ffill.", n_nan)
            df = df.dropna()

        log.info("Preprocessor: %d rows after cleaning.", len(df))
        return df

    def align_to_trading_days(
        self,
        df: pd.DataFrame,
        reference: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Reindex *df* to a reference DatetimeIndex and forward-fill.

        Useful when merging multiple DataFrames that may have slightly
        different trading-day calendars (e.g. ETFs vs FRED data).

        Args:
            df: DataFrame to reindex.
            reference: Target DatetimeIndex.  If None, *df*'s own index is
                used (effectively a no-op).

        Returns:
            Reindexed and forward-filled DataFrame.
        """
        if reference is None:
            return df
        return df.reindex(reference).ffill()

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``log_return`` column (log of daily close ratio).

        Args:
            df: DataFrame with a ``close`` column.

        Returns:
            DataFrame with added ``log_return`` column.
        """
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.OHLCV_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                "DataFrame is missing required columns: %s" % missing
            )
