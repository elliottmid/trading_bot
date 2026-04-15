"""
reporter.py — Backtest reporting utilities.

Formats BacktestResult data into console tables and optional CSV exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .backtest_engine import BacktestResult
from ..logger import get_logger

log = get_logger(__name__)


class BacktestReporter:
    """Render backtest results to stdout and optionally to CSV files.

    Args:
        output_dir: Directory for CSV exports.  If None, no files are written.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else None

    def print_summary(self, result: BacktestResult, title: str = "Backtest Summary") -> None:
        """Print a formatted summary table to stdout.

        Args:
            result: BacktestResult instance from BacktestEngine.run().
            title: Heading for the summary block.
        """
        sep = "=" * 50
        print(sep)
        print(title.center(50))
        print(sep)
        print("  %-30s %d" % ("Total trades:", result.n_trades))
        print("  %-30s %.2f%%" % ("Total return:", result.total_return_pct))
        print("  %-30s %.3f" % ("Sharpe ratio:", result.sharpe_ratio))
        print("  %-30s %.1f%%" % ("Win rate:", result.win_rate * 100))
        print("  %-30s %.2f%%" % ("Max drawdown:", result.max_drawdown_pct))
        print(sep)

        if result.trades:
            pnls = [t.pnl_pct for t in result.trades]
            print("  %-30s %.2f%%" % ("Avg trade return:", sum(pnls) / len(pnls)))
            print(
                "  %-30s %.2f%% / %.2f%%"
                % (
                    "Best / Worst trade:",
                    max(pnls),
                    min(pnls),
                )
            )

            exits = {}
            for t in result.trades:
                exits[t.exit_reason] = exits.get(t.exit_reason, 0) + 1
            print("  Exit reasons:", exits)
        print(sep)

    def export_trades(self, result: BacktestResult, filename: str = "trades.csv") -> None:
        """Write the trade log to a CSV file.

        Args:
            result: BacktestResult containing trade list.
            filename: Output filename (relative to output_dir).
        """
        if not self._output_dir:
            log.warning("No output_dir set; skipping trade export.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        rows = [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
            }
            for t in result.trades
        ]
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Trade log exported to %s.", path)

    def export_equity_curve(
        self, result: BacktestResult, filename: str = "equity_curve.csv"
    ) -> None:
        """Write the equity curve to a CSV file.

        Args:
            result: BacktestResult with an equity_curve Series.
            filename: Output filename.
        """
        if not self._output_dir:
            log.warning("No output_dir set; skipping equity curve export.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        result.equity_curve.to_csv(path, header=["equity"])
        log.info("Equity curve exported to %s.", path)
