"""
swing_trader_bot.py — Main trading loop for the swing-trading bot.

Orchestrates data fetching, signal generation, risk checks, and order
execution.  Designed to run continuously as a daemon process.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from ..config import Config
from ..data.preprocessor import Preprocessor
from ..data.schwab_fetcher import SchwabFetcher
from ..execution.position_manager import PositionManager
from ..execution.risk_manager import RiskManager
from ..execution.schwab_executor import SchwabExecutor
from ..logger import get_logger
from ..models.feature_eng import FeatureEngineer, FEATURE_COLS
from ..models.swing_trading_v1 import SwingTradingModel
from ..signals.signal_aggregator import SignalAggregator

log = get_logger(__name__)


class SwingTraderBot:
    """Main trading bot that runs a continuous signal-and-execute loop.

    Args:
        config: Application configuration.
        model: Pre-trained SwingTradingModel.
        fetcher: Authenticated SchwabFetcher.
        executor: SchwabExecutor (or DryRunSimulator with the same interface).
    """

    def __init__(
        self,
        config: Config,
        model: SwingTradingModel,
        fetcher: SchwabFetcher,
        executor: SchwabExecutor,
    ) -> None:
        self._config = config
        self._model = model
        self._fetcher = fetcher
        self._executor = executor

        self._preprocessor = Preprocessor()
        self._feature_eng = FeatureEngineer()
        self._risk_mgr = RiskManager(config=config, fetcher=fetcher)
        self._position_mgr = PositionManager(path=config.positions_path)
        self._aggregator = SignalAggregator(config=config)

        self._running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main trading loop.

        Runs indefinitely until a KeyboardInterrupt is received or
        ``self._running`` is set to False.  Each iteration:

        1. Checks market hours.
        2. Refreshes account equity for daily loss tracking.
        3. Updates prices on open positions and checks exit conditions.
        4. For each symbol without an open position, generates a signal and
           potentially enters a trade.
        5. Sleeps for ``config.loop_interval_seconds``.
        """
        self._running = True
        log.info(
            "SwingTraderBot started. Symbols: %s | DryRun: %s | Interval: %ds.",
            self._config.symbols,
            self._config.dry_run,
            self._config.loop_interval_seconds,
        )

        try:
            while self._running:
                try:
                    self._loop_iteration()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error(
                        "Unhandled error in loop iteration: %s. "
                        "Sleeping %ds before retry.",
                        exc,
                        self._config.loop_interval_seconds,
                        exc_info=True,
                    )
                time.sleep(self._config.loop_interval_seconds)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received — shutting down cleanly.")
        finally:
            self._running = False
            log.info("SwingTraderBot stopped.")

    def stop(self) -> None:
        """Signal the main loop to stop after the current iteration."""
        log.info("Stop requested.")
        self._running = False

    # ------------------------------------------------------------------
    # Loop iteration
    # ------------------------------------------------------------------

    def _loop_iteration(self) -> None:
        """Execute one full iteration of the trading loop."""
        now = datetime.utcnow()
        log.info("--- Loop iteration at %s UTC ---", now.strftime("%Y-%m-%d %H:%M:%S"))

        # 1. Market hours gate
        if not self._risk_mgr.is_market_open():
            log.info("Market is closed — skipping iteration.")
            return

        # 2. Refresh account equity
        try:
            equity = self._fetcher.fetch_account_balance()
            self._risk_mgr.set_day_start_equity(equity)
        except Exception as exc:
            log.error("Could not fetch account balance: %s", exc)
            equity = 0.0

        # 3. Daily loss gate
        if not self._risk_mgr.check_daily_loss(equity):
            log.warning("Daily loss limit breached — no new trades today.")
            self._update_open_positions(exit_only=True)
            return

        # 4. Check exit conditions on open positions
        self._update_open_positions(exit_only=False)

        # 5. PDT gate
        allowed, reason = self._risk_mgr.can_trade()
        if not allowed:
            log.info("Trading blocked: %s", reason)
            return

        # 6. Signal generation and entry
        for symbol in self._config.symbols:
            if self._position_mgr.has_position(symbol):
                log.debug("%s: already have an open position; skipping entry.", symbol)
                continue
            self._evaluate_entry(symbol, equity)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _evaluate_entry(self, symbol: str, equity: float) -> None:
        """Generate a signal for *symbol* and enter if conditions are met.

        Args:
            symbol: Ticker symbol to evaluate.
            equity: Current account equity for position sizing.
        """
        try:
            raw_df = self._fetcher.fetch_price_history(
                symbol, days=self._config.lookback_days
            )
        except Exception as exc:
            log.error("Could not fetch price history for %s: %s", symbol, exc)
            return

        try:
            clean_df = self._preprocessor.clean(raw_df)
            featured_df = self._feature_eng.compute_features(clean_df)
        except Exception as exc:
            log.error("Feature engineering failed for %s: %s", symbol, exc)
            return

        feature_cols = [c for c in FEATURE_COLS if c in featured_df.columns]
        if len(featured_df) < 5:
            log.warning("%s: insufficient data after feature engineering.", symbol)
            return

        try:
            signal, confidence = self._model.predict(featured_df[feature_cols])
        except Exception as exc:
            log.error("Model prediction failed for %s: %s", symbol, exc)
            return

        # Pass the latest feature row so the aggregator can apply
        # RSI/MACD technical filters before committing to a BUY signal.
        latest_features = featured_df.iloc[-1]
        final_signal, final_confidence = self._aggregator.aggregate(
            model_signal=signal,
            model_confidence=confidence,
            features=latest_features,
        )

        log.info(
            "%s: signal=%d confidence=%.3f.", symbol, final_signal, final_confidence
        )

        if final_signal != 1:
            return

        # Get current price from latest bar
        current_price = float(featured_df["close"].iloc[-1])
        qty = self._risk_mgr.get_position_size(
            account_equity=equity,
            confidence=final_confidence,
            price=current_price,
        )
        if qty < 1:
            log.warning("%s: calculated qty < 1; skipping entry.", symbol)
            return

        stop_loss_price = current_price * (
            1 - self._config.stop_loss_pct / 100
        )
        take_profit_price = current_price * (
            1 + self._config.take_profit_pct / 100
        )

        order_id = self._executor.buy(symbol=symbol, qty=qty, price=current_price)
        if order_id is not None or self._config.dry_run:
            self._position_mgr.add_position(
                symbol=symbol,
                qty=qty,
                entry_price=current_price,
                entry_time=datetime.utcnow(),
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
            )
            log.info(
                "Entered %s x%d @ $%.2f (SL=%.2f TP=%.2f).",
                symbol,
                qty,
                current_price,
                stop_loss_price,
                take_profit_price,
            )

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _update_open_positions(self, exit_only: bool = False) -> None:
        """Refresh prices on open positions and check exit conditions.

        Args:
            exit_only: If True, only check exits; do not log entry signals.
        """
        symbols = self._position_mgr.get_symbols()
        if not symbols:
            return

        try:
            quotes = self._fetcher.fetch_quotes(symbols)
        except Exception as exc:
            log.error("Could not fetch quotes for open positions: %s", exc)
            return

        for symbol in list(symbols):
            pos = self._position_mgr.get_position(symbol)
            if pos is None:
                continue

            quote = quotes.get(symbol, {})
            current_price = float(
                quote.get("quote", {}).get("lastPrice", pos["current_price"])
            )
            self._position_mgr.update_price(symbol, current_price)

            unrealized_pnl = self._position_mgr.get_unrealized_pnl(symbol)
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100

            log.debug(
                "%s: price=%.2f P&L=$%.2f (%.2f%%).",
                symbol,
                current_price,
                unrealized_pnl,
                pnl_pct,
            )

            should_exit, reason = self._check_exit_conditions(pos, current_price)
            if should_exit:
                log.info("Exiting %s: %s.", symbol, reason)
                self._executor.sell(
                    symbol=symbol, qty=pos["qty"], price=current_price
                )
                realized_pnl = self._position_mgr.close_position(symbol)
                self._risk_mgr.log_trade(
                    symbol=symbol,
                    side="SELL",
                    qty=pos["qty"],
                    price=current_price,
                )
                log.info(
                    "Closed %s | Realized P&L: $%.2f (%s).",
                    symbol,
                    realized_pnl,
                    reason,
                )

    @staticmethod
    def _check_exit_conditions(
        pos: dict, current_price: float
    ) -> tuple:
        """Evaluate stop-loss and take-profit conditions.

        Args:
            pos: Position dict from PositionManager.
            current_price: Latest market price.

        Returns:
            Tuple of (should_exit: bool, reason: str).
        """
        if current_price <= pos["stop_loss"]:
            return True, "stop_loss"
        if current_price >= pos["take_profit"]:
            return True, "take_profit"
        return False, ""
