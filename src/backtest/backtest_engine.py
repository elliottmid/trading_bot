"""
backtest_engine.py — Walk-forward backtesting engine.

Drives rolling-window training and simulation runs, collecting trade-by-trade
P&L so that aggregate performance metrics can be computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logger import get_logger
from ..models.feature_eng import FeatureEngineer, FEATURE_COLS
from ..models.model_utils import walk_forward_splits
from ..models.swing_trading_v1 import SwingTradingModel

log = get_logger(__name__)


@dataclass
class Trade:
    """Represents a single simulated round-trip trade."""

    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    qty: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Aggregated result from a backtest run."""

    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    n_trades: int = 0


class BacktestEngine:
    """Walk-forward backtesting engine for the swing-trading model.

    Args:
        train_window: Number of trading days in each training window.
        test_window: Number of trading days in each out-of-sample window.
        step: Number of days to advance between folds.
        prediction_horizon: Hold period in trading days.
        min_confidence: Minimum model confidence to enter a trade.
        stop_loss_pct: Stop-loss threshold as a percentage (e.g. 2.0 = 2%).
        take_profit_pct: Take-profit threshold as a percentage.
        initial_equity: Starting portfolio equity in USD.
        position_size_pct: Fraction of equity to allocate per trade.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 20,
        step: int = 20,
        prediction_horizon: int = 5,
        min_confidence: float = 0.65,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 5.0,
        initial_equity: float = 100_000.0,
        position_size_pct: float = 0.05,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.prediction_horizon = prediction_horizon
        self.min_confidence = min_confidence
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        self._fe = FeatureEngineer()

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """Execute a full walk-forward backtest.

        Args:
            df: Raw OHLCV DataFrame for a single symbol.
            symbol: Ticker symbol label (used for logging).

        Returns:
            BacktestResult containing trades and aggregate statistics.
        """
        log.info("Starting backtest for %s.", symbol)

        # Compute features once on the full dataset
        featured = self._fe.compute_features(df)
        target = self._fe.create_target(
            featured, horizon=self.prediction_horizon
        )
        featured["target"] = target
        featured = featured.dropna(subset=["target"])

        feature_cols = [c for c in FEATURE_COLS if c in featured.columns]

        all_trades: List[Trade] = []
        equity = self.initial_equity
        equity_records: List[Tuple[datetime, float]] = [
            (featured.index[0], equity)
        ]

        n_folds = 0
        for train_df, test_df in walk_forward_splits(
            featured,
            train_window=self.train_window,
            test_window=self.test_window,
            step=self.step,
        ):
            n_folds += 1
            X_train = train_df[feature_cols]
            y_train = train_df["target"]

            model = SwingTradingModel(
                prediction_horizon=self.prediction_horizon
            )
            try:
                model.train(X_train, y_train)
            except Exception as exc:
                log.warning("Fold %d training failed: %s", n_folds, exc)
                continue

            # Simulate trades on the test window
            for i in range(len(test_df) - self.prediction_horizon):
                row_features = test_df[feature_cols].iloc[: i + 1]
                signal, confidence = model.predict(row_features)

                if signal != 1 or confidence < self.min_confidence:
                    continue

                entry_date = test_df.index[i]
                entry_price = float(test_df["close"].iloc[i])
                qty = max(
                    1,
                    int(equity * self.position_size_pct / entry_price),
                )

                # Determine exit
                exit_idx = i + self.prediction_horizon
                exit_date = test_df.index[exit_idx]
                exit_price = float(test_df["close"].iloc[exit_idx])
                exit_reason = "horizon"

                # Check stop / take-profit intra-window
                for j in range(i + 1, exit_idx + 1):
                    price = float(test_df["close"].iloc[j])
                    pct = (price - entry_price) / entry_price * 100
                    if pct <= -self.stop_loss_pct:
                        exit_price = price
                        exit_date = test_df.index[j]
                        exit_reason = "stop_loss"
                        break
                    if pct >= self.take_profit_pct:
                        exit_price = price
                        exit_date = test_df.index[j]
                        exit_reason = "take_profit"
                        break

                pnl = (exit_price - entry_price) * qty
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                equity += pnl
                equity_records.append((exit_date, equity))

                all_trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=exit_date,
                        exit_price=exit_price,
                        qty=qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                    )
                )

        log.info(
            "Backtest complete: %d folds, %d trades.", n_folds, len(all_trades)
        )
        return self._build_result(all_trades, equity_records)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        trades: List[Trade],
        equity_records: List[Tuple[datetime, float]],
    ) -> BacktestResult:
        """Compute aggregate statistics from raw trade list."""
        if not trades:
            log.warning("No trades generated — check confidence threshold.")
            return BacktestResult(trades=[], n_trades=0)

        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls) if pnls else 0.0

        equity_series = pd.Series(
            {dt: eq for dt, eq in equity_records}
        ).sort_index()
        total_return = (equity_series.iloc[-1] / self.initial_equity - 1) * 100
        max_dd = self._max_drawdown(equity_series)

        # Annualised Sharpe (daily returns, risk-free = 0)
        daily_returns = pd.Series(pnls) / 100
        sharpe = 0.0
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            total_return_pct=float(total_return),
            sharpe_ratio=float(sharpe),
            win_rate=float(win_rate),
            max_drawdown_pct=float(max_dd),
            n_trades=len(trades),
        )

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        """Compute maximum drawdown percentage from an equity curve."""
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max * 100
        return float(drawdown.min())
