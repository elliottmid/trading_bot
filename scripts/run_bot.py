#!/usr/bin/env python3
"""
run_bot.py — Entry point for the swing-trading bot.

Usage:
    python scripts/run_bot.py [--env /path/to/.env]

Loads configuration, verifies the auth token, loads the trained model,
then starts the SwingTraderBot (or DryRunSimulator based on DRY_RUN env var).
Handles SIGINT for clean shutdown.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.schwab_fetcher import SchwabFetcher
from src.execution.auth import SchwabAuth
from src.execution.schwab_executor import SchwabExecutor
from src.bot.dry_run_simulator import DryRunSimulator
from src.bot.swing_trader_bot import SwingTraderBot
from src.logger import get_logger
from src.models.swing_trading_v1 import SwingTradingModel

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Swing-trading bot entry point.")
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: searches upward from cwd).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # 1. Load configuration
    config = Config.from_env(dotenv_path=args.env)
    config.ensure_dirs()
    try:
        config.validate()
    except ValueError as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    log.info(
        "Configuration loaded. DryRun=%s LiveTrading=%s Symbols=%s.",
        config.dry_run,
        config.live_trading,
        config.symbols,
    )

    # 2. Authenticate and check token age
    auth = SchwabAuth(config=config)
    try:
        client = auth.get_client()
    except (FileNotFoundError, RuntimeError) as exc:
        log.error(
            "Authentication failed: %s\n"
            "Run 'python scripts/auth_setup.py' to generate a token.",
            exc,
        )
        sys.exit(1)

    auth.warn_if_expiring_soon()

    # 3. Build fetcher
    fetcher = SchwabFetcher(config=config)

    # 4. Load trained model
    model_path = config.model_path
    if not Path(model_path).exists():
        log.error(
            "Model file not found at '%s'. "
            "Run 'python scripts/train_model.py' to train and save a model first.",
            model_path,
        )
        sys.exit(1)

    try:
        model = SwingTradingModel.load(model_path)
    except Exception as exc:
        log.error("Failed to load model from '%s': %s", model_path, exc)
        sys.exit(1)

    # 5. Build executor (dry-run or live)
    if config.dry_run:
        log.info("DRY RUN mode — no real orders will be placed.")
        executor = DryRunSimulator(config=config)
    else:
        from src.execution.risk_manager import RiskManager
        risk_mgr = RiskManager(config=config, fetcher=fetcher)
        executor = SchwabExecutor(
            config=config, fetcher=fetcher, risk_manager=risk_mgr
        )

    # 6. Instantiate and start bot
    bot = SwingTraderBot(
        config=config,
        model=model,
        fetcher=fetcher,
        executor=executor,
    )

    # Handle SIGINT / SIGTERM gracefully
    def _shutdown(signum, frame):
        log.info("Signal %d received — requesting bot shutdown.", signum)
        bot.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Starting bot...")
    bot.run()

    if config.dry_run and hasattr(executor, "print_summary"):
        executor.print_summary()

    log.info("Bot exited cleanly.")


if __name__ == "__main__":
    main()
