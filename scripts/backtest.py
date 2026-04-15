#!/usr/bin/env python3
"""
backtest.py — Walk-forward backtest script.

Loads raw OHLCV data, splits it into monthly folds, trains on the prior
252 days, evaluates on the next 20 days, and aggregates Sharpe ratio,
win rate, and total return into a summary table.

Usage:
    python scripts/backtest.py [--symbols SPY QQQ] [--horizon 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.preprocessor import Preprocessor
from src.logger import get_logger
from src.models.feature_eng import FeatureEngineer, FEATURE_COLS
from src.backtest.backtest_engine import BacktestEngine, BacktestResult
from src.backtest.metrics import sharpe_ratio, win_rate, profit_factor
from src.backtest.reporter import BacktestReporter

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest runner.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to backtest (default: config.symbols).",
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="Prediction horizon in days."
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=252,
        help="Training window in trading days (default: 252).",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=20,
        help="Test window per fold in trading days (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/backtest",
        help="Directory for CSV exports.",
    )
    parser.add_argument("--env", default=None, help="Path to .env file.")
    return parser.parse_args()


def load_symbol_csv(symbol: str) -> pd.DataFrame | None:
    """Load OHLCV CSV from data/raw/.

    Args:
        symbol: Ticker symbol.

    Returns:
        DataFrame or None if missing.
    """
    csv_path = Path("data/raw/%s.csv" % symbol)
    if not csv_path.exists():
        log.warning(
            "No CSV for %s at %s. Run fetch_sample_data.py first.", symbol, csv_path
        )
        return None
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def print_results_table(rows: List[dict]) -> None:
    """Print a formatted results table.

    Args:
        rows: List of per-symbol result dicts.
    """
    header = "%-8s  %8s  %8s  %8s  %10s  %10s" % (
        "Symbol",
        "Trades",
        "WinRate",
        "Sharpe",
        "TotalRet%",
        "MaxDD%",
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("WALK-FORWARD BACKTEST RESULTS".center(len(header)))
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(
            "%-8s  %8d  %8.1f%%  %8.3f  %10.2f%%  %10.2f%%"
            % (
                row["symbol"],
                row["n_trades"],
                row["win_rate"] * 100,
                row["sharpe"],
                row["total_return_pct"],
                row["max_drawdown_pct"],
            )
        )
    print(sep + "\n")


def main() -> None:
    args = parse_args()
    config = Config.from_env(dotenv_path=args.env)
    config.ensure_dirs()

    symbols = args.symbols or config.symbols
    horizon = args.horizon or config.prediction_horizon

    engine = BacktestEngine(
        train_window=args.train_window,
        test_window=args.test_window,
        prediction_horizon=horizon,
        min_confidence=config.min_confidence,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        position_size_pct=config.max_position_size_pct,
    )

    reporter = BacktestReporter(output_dir=args.output_dir)
    preprocessor = Preprocessor()

    summary_rows = []

    for symbol in symbols:
        df = load_symbol_csv(symbol)
        if df is None:
            continue

        clean_df = preprocessor.clean(df)
        log.info("Running walk-forward backtest for %s...", symbol)

        # TODO: plug in BacktestEngine.run() with walk-forward splits
        # The engine handles training + test loop internally
        result: BacktestResult = engine.run(clean_df, symbol=symbol)

        reporter.print_summary(result, title="Results: %s" % symbol)
        reporter.export_trades(result, filename="%s_trades.csv" % symbol)
        reporter.export_equity_curve(result, filename="%s_equity.csv" % symbol)

        summary_rows.append(
            {
                "symbol": symbol,
                "n_trades": result.n_trades,
                "win_rate": result.win_rate,
                "sharpe": result.sharpe_ratio,
                "total_return_pct": result.total_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
            }
        )

    if summary_rows:
        print_results_table(summary_rows)
    else:
        print(
            "No backtest results generated. "
            "Ensure data/raw/{SYMBOL}.csv files exist."
        )


if __name__ == "__main__":
    main()
