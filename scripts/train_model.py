#!/usr/bin/env python3
"""
train_model.py — Train and save the swing-trading XGBoost model.

Loads historical OHLCV CSV files from data/raw/, runs feature engineering,
trains SwingTradingModel, prints metrics, and saves the model to
data/models/swing_v1.pkl.

Usage:
    python scripts/train_model.py [--symbols SPY QQQ IWM] [--horizon 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.preprocessor import Preprocessor
from src.logger import get_logger
from src.models.feature_eng import FeatureEngineer, FEATURE_COLS
from src.models.model_utils import print_metrics_table
from src.models.swing_trading_v1 import SwingTradingModel

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the swing-trading model.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to load from data/raw/ (defaults to config.symbols).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon in trading days (defaults to config value).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.025,
        help="Minimum return threshold for a positive label (default 0.025 = 2.5%%).",
    )
    parser.add_argument("--env", default=None, help="Path to .env file.")
    return parser.parse_args()


def load_symbol_data(symbol: str, raw_dir: Path) -> pd.DataFrame | None:
    """Load OHLCV CSV for *symbol* from data/raw/.

    Args:
        symbol: Ticker symbol string.
        raw_dir: Path to the raw data directory.

    Returns:
        DataFrame or None if the file is missing.
    """
    csv_path = raw_dir / ("%s.csv" % symbol)
    if not csv_path.exists():
        log.warning(
            "No data file for %s at %s. "
            "Run 'python scripts/fetch_sample_data.py %s' first.",
            symbol,
            csv_path,
            symbol,
        )
        return None

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    log.info("Loaded %d rows for %s.", len(df), symbol)
    return df


def main() -> None:
    args = parse_args()
    config = Config.from_env(dotenv_path=args.env)
    config.ensure_dirs()

    symbols = args.symbols or config.symbols
    horizon = args.horizon or config.prediction_horizon
    raw_dir = Path("data/raw")

    preprocessor = Preprocessor()
    feature_eng = FeatureEngineer()

    all_X = []
    all_y = []

    for symbol in symbols:
        raw_df = load_symbol_data(symbol, raw_dir)
        if raw_df is None:
            continue

        clean_df = preprocessor.clean(raw_df)
        featured_df = feature_eng.compute_features(clean_df)
        target = feature_eng.create_target(
            featured_df, horizon=horizon, threshold=args.threshold
        )
        featured_df["target"] = target
        featured_df = featured_df.dropna(subset=["target"])

        feature_cols = [c for c in FEATURE_COLS if c in featured_df.columns]
        all_X.append(featured_df[feature_cols])
        all_y.append(featured_df["target"])

    if not all_X:
        print(
            "ERROR: No training data found. "
            "Fetch data first with scripts/fetch_sample_data.py."
        )
        sys.exit(1)

    X = pd.concat(all_X).sort_index()
    y = pd.concat(all_y).sort_index()

    log.info(
        "Training dataset: %d rows, %d features. Positive rate: %.1f%%.",
        len(X),
        len(X.columns),
        y.mean() * 100,
    )

    model = SwingTradingModel(
        lookback_days=config.lookback_days,
        prediction_horizon=horizon,
    )
    metrics = model.train(X, y)

    print_metrics_table(metrics, title="Train / Validation Metrics")

    model.save(config.model_path)
    print("\nModel saved to: %s" % config.model_path)


if __name__ == "__main__":
    main()
