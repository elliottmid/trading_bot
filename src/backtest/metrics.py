"""
metrics.py — Standalone performance metric calculations for backtests.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Periodic (e.g. daily) return series.
        risk_free: Risk-free rate per period (default 0).
        periods: Number of periods per year for annualisation (252 for daily).

    Returns:
        Annualised Sharpe ratio, or 0.0 if std is zero.
    """
    excess = returns - risk_free
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * (periods ** 0.5))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation in denominator).

    Args:
        returns: Periodic return series.
        risk_free: Risk-free rate per period.
        periods: Periods per year.

    Returns:
        Annualised Sortino ratio, or 0.0 if downside std is zero.
    """
    excess = returns - risk_free
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * (periods ** 0.5))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a percentage.

    Args:
        equity_curve: Series of portfolio equity values.

    Returns:
        Maximum drawdown as a negative percentage (e.g. -15.3).
    """
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max * 100
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate.

    Args:
        equity_curve: Series of portfolio equity values.
        periods_per_year: Trading periods in a year (252 for daily).

    Returns:
        CAGR as a percentage (e.g. 12.5 means 12.5% per year).
    """
    n = len(equity_curve)
    if n < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n / periods_per_year
    return float((total_return ** (1 / years) - 1) * 100)


def win_rate(pnl_series: List[float]) -> float:
    """Fraction of trades with positive P&L.

    Args:
        pnl_series: List of per-trade P&L values.

    Returns:
        Win rate between 0.0 and 1.0.
    """
    if not pnl_series:
        return 0.0
    wins = sum(1 for p in pnl_series if p > 0)
    return wins / len(pnl_series)


def profit_factor(pnl_series: List[float]) -> float:
    """Gross profit divided by gross loss.

    Args:
        pnl_series: List of per-trade P&L values.

    Returns:
        Profit factor (> 1 is net profitable), or inf if no losses.
    """
    gross_profit = sum(p for p in pnl_series if p > 0)
    gross_loss = abs(sum(p for p in pnl_series if p < 0))
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)
