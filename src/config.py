"""
config.py — Central configuration loader.

Reads all settings from environment variables (populated via .env) and
exposes them as typed attributes on a Config dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv


@dataclass
class Config:
    """Typed configuration for the swing-trading bot.

    All attributes have sensible defaults so the bot can run in dry-run mode
    without a fully populated .env file.
    """

    # --- Schwab API ---
    schwab_api_key: str = ""
    schwab_api_secret: str = ""
    schwab_callback_url: str = "https://127.0.0.1:8182"
    schwab_token_path: str = "/Users/elliottmiddleton/trading-bot/token.json"

    # --- Trading universe ---
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])
    live_trading: bool = False
    dry_run: bool = True
    loop_interval_seconds: int = 60

    # --- Model parameters ---
    lookback_days: int = 252
    prediction_horizon: int = 5
    min_confidence: float = 0.65

    # --- Risk management ---
    max_position_size_pct: float = 0.05
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 5.0
    max_daily_loss_pct: float = 2.0

    # --- External signals ---
    finviz_api_key: str = ""
    tradingview_webhook_key: str = ""
    signal_agreement_weight: float = 0.2

    # --- Paths ---
    model_path: str = "./data/models/swing_v1.pkl"
    positions_path: str = "./data/processed/positions.json"
    trades_log_path: str = "./data/processed/trades.csv"
    dry_run_trades_path: str = "./data/processed/dry_run_trades.csv"

    # --- Logging ---
    log_level: str = "INFO"
    log_path: str = "./logs/bot.log"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to a .env file.  If None, python-dotenv
                searches from the current working directory upward.

        Returns:
            A fully populated Config instance.
        """
        load_dotenv(dotenv_path=dotenv_path, override=False)

        symbols_raw = os.getenv("SYMBOLS", "SPY,QQQ,IWM")
        symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

        return cls(
            # Schwab
            schwab_api_key=os.getenv("SCHWAB_API_KEY", ""),
            schwab_api_secret=os.getenv("SCHWAB_API_SECRET", ""),
            schwab_callback_url=os.getenv(
                "SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182"
            ),
            schwab_token_path=os.getenv(
                "SCHWAB_TOKEN_PATH",
                "/Users/elliottmiddleton/trading-bot/token.json",
            ),
            # Trading
            symbols=symbols,
            live_trading=os.getenv("LIVE_TRADING", "false").lower() == "true",
            dry_run=os.getenv("DRY_RUN", "true").lower() == "true",
            loop_interval_seconds=int(
                os.getenv("LOOP_INTERVAL_SECONDS", "60")
            ),
            # Model
            lookback_days=int(os.getenv("LOOKBACK_DAYS", "252")),
            prediction_horizon=int(os.getenv("PREDICTION_HORIZON", "5")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.65")),
            # Risk
            max_position_size_pct=float(
                os.getenv("MAX_POSITION_SIZE_PCT", "0.05")
            ),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "2.0")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "5.0")),
            max_daily_loss_pct=float(
                os.getenv("MAX_DAILY_LOSS_PCT", "2.0")
            ),
            # External signals
            finviz_api_key=os.getenv("FINVIZ_API_KEY", ""),
            tradingview_webhook_key=os.getenv("TRADINGVIEW_WEBHOOK_KEY", ""),
            signal_agreement_weight=float(
                os.getenv("SIGNAL_AGREEMENT_WEIGHT", "0.2")
            ),
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_path=os.getenv("LOG_PATH", "./logs/bot.log"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError if any required field is missing for live trading."""
        if self.live_trading and not self.dry_run:
            missing = []
            if not self.schwab_api_key:
                missing.append("SCHWAB_API_KEY")
            if not self.schwab_api_secret:
                missing.append("SCHWAB_API_SECRET")
            if missing:
                raise ValueError(
                    "Live trading enabled but missing env vars: %s"
                    % ", ".join(missing)
                )

    def ensure_dirs(self) -> None:
        """Create any missing directories referenced by path configs."""
        for path_str in (
            self.log_path,
            self.model_path,
            self.positions_path,
            self.trades_log_path,
            self.dry_run_trades_path,
        ):
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
