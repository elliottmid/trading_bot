"""
feature_eng.py — Technical indicator feature engineering.

Uses pandas-ta to compute a standard set of swing-trading indicators and
appends them to an OHLCV DataFrame.  Also provides a target-variable
constructor for binary classification (price up > threshold in N days).
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from ..logger import get_logger

log = get_logger(__name__)

# Feature column names produced by compute_features()
FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_pct",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "sma_20",
    "sma_50",
    "sma_200",
    "volume_sma_20",
    # Derived price-relative features
    "close_vs_sma20",
    "close_vs_sma50",
    "close_vs_bb_mid",
    "volume_ratio",
]


class FeatureEngineer:
    """Compute technical indicators and build the model feature matrix.

    All indicator parameters follow common swing-trading defaults and match
    the values described in the project spec.
    """

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator columns to *df*.

        Args:
            df: OHLCV DataFrame with columns ``[open, high, low, close,
                volume]`` and a DatetimeIndex.

        Returns:
            A new DataFrame containing all original columns plus the
            indicator columns listed in ``FEATURE_COLS``.  Rows that contain
            NaN in any indicator column are dropped (typically the first
            ~200 rows needed to compute SMA-200).
        """
        df = df.copy()

        # --- RSI ---
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # --- MACD ---
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df["macd"] = macd.iloc[:, 0]        # MACD line
            df["macd_signal"] = macd.iloc[:, 2]  # Signal line
            df["macd_hist"] = macd.iloc[:, 1]    # Histogram

        # --- Bollinger Bands ---
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df["bb_upper"] = bbands.iloc[:, 0]
            df["bb_mid"] = bbands.iloc[:, 1]
            df["bb_lower"] = bbands.iloc[:, 2]
            df["bb_pct"] = bbands.iloc[:, 3]  # %B

        # --- ATR ---
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # --- Stochastic ---
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
        if stoch is not None:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        # --- Simple moving averages ---
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["sma_200"] = ta.sma(df["close"], length=200)

        # --- Volume SMA ---
        df["volume_sma_20"] = ta.sma(df["volume"], length=20)

        # --- Derived relative features ---
        df["close_vs_sma20"] = df["close"] / df["sma_20"] - 1
        df["close_vs_sma50"] = df["close"] / df["sma_50"] - 1
        df["close_vs_bb_mid"] = df["close"] / df["bb_mid"] - 1
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # Drop rows with any NaN in the feature columns
        feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
        n_before = len(df)
        df = df.dropna(subset=feature_cols_present)
        n_dropped = n_before - len(df)
        if n_dropped:
            log.debug(
                "Dropped %d rows with NaN in features (indicator warm-up).",
                n_dropped,
            )

        log.info(
            "FeatureEngineer: %d rows, %d feature columns.",
            len(df),
            len(feature_cols_present),
        )
        return df

    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.025,
    ) -> pd.Series:
        """Build a binary classification target.

        A label of **1** means the closing price rises by more than
        *threshold* (default 2.5%) within the next *horizon* trading days.
        Label **0** covers all other outcomes (flat or down).

        Args:
            df: OHLCV DataFrame with a ``close`` column.
            horizon: Look-ahead window in trading days.
            threshold: Minimum fractional price increase to label as 1.

        Returns:
            Binary pandas Series (0 or 1) aligned to *df*'s index.
            The last *horizon* rows will be NaN and should be dropped
            before training.
        """
        future_close = df["close"].shift(-horizon)
        pct_change = (future_close - df["close"]) / df["close"]
        target = (pct_change > threshold).astype(int)
        target.name = "target"
        log.info(
            "Target: horizon=%d days, threshold=%.1f%%, positive rate=%.1f%%.",
            horizon,
            threshold * 100,
            target.dropna().mean() * 100,
        )
        return target

    def get_feature_cols(self) -> list:
        """Return the list of feature column names produced by compute_features."""
        return list(FEATURE_COLS)
