#!/usr/bin/env python3
"""
EMA12/EMA26 crossover backtest engine.
Per-symbol and universe-level performance analysis.
"""

import sys
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def load_etf_data(filename="ema_etfs_primary_daily.parquet"):
    """Load daily OHLCV data for all ETFs."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    logger.info(f"✓ Loaded {len(df)} bars for {df['symbol'].nunique()} symbols")
    return df


def calculate_emas(symbol_df, ema_fast=12, ema_slow=26):
    """
    Calculate EMA12 and EMA26 for a single symbol.

    Args:
        symbol_df: DataFrame with close prices, sorted by date
        ema_fast: Fast EMA period (default 12)
        ema_slow: Slow EMA period (default 26)

    Returns:
        DataFrame with ema_fast and ema_slow columns added
    """
    symbol_df = symbol_df.copy().sort_values('timestamp')
    symbol_df['ema_fast'] = symbol_df['close'].ewm(span=ema_fast, adjust=False).mean()
    symbol_df['ema_slow'] = symbol_df['close'].ewm(span=ema_slow, adjust=False).mean()
    return symbol_df


def generate_signals(symbol_df):
    """
    Generate EMA crossover signals.

    Signal logic:
      - BUY: ema_fast crosses above ema_slow (1.0)
      - SELL: ema_fast crosses below ema_slow (-1.0)
      - NEUTRAL: not in position or between signals (0.0)

    Returns:
        DataFrame with signal column (-1, 0, 1)
    """
    symbol_df = symbol_df.copy()

    # Determine crossover state: 1 if fast > slow, 0 otherwise
    symbol_df['fast_above_slow'] = (symbol_df['ema_fast'] > symbol_df['ema_slow']).astype(int)

    # Previous day's state
    symbol_df['fast_above_slow_prev'] = symbol_df['fast_above_slow'].shift(1)

    # Buy signal: transition from 0 to 1 (fast crosses above slow)
    # Sell signal: transition from 1 to 0 (fast crosses below slow)
    symbol_df['signal'] = 0
    symbol_df.loc[
        (symbol_df['fast_above_slow_prev'] == 0) & (symbol_df['fast_above_slow'] == 1),
        'signal'
    ] = 1  # BUY

    symbol_df.loc[
        (symbol_df['fast_above_slow_prev'] == 1) & (symbol_df['fast_above_slow'] == 0),
        'signal'
    ] = -1  # SELL

    return symbol_df[['timestamp', 'symbol', 'close', 'ema_fast', 'ema_slow', 'fast_above_slow', 'signal']]


def backtest_symbol(symbol_df, tc_bps=0):
    """
    Backtest a single symbol with EMA crossover rule.

    Args:
        symbol_df: DataFrame with close, ema_fast, ema_slow, signal columns
        tc_bps: transaction cost in basis points (default 0)

    Returns:
        dict with trade-level and performance metrics
    """
    symbol = symbol_df['symbol'].iloc[0]
    df = symbol_df.copy()

    # Track positions and trades
    in_position = False
    entry_price = None
    entry_idx = None
    trades = []

    for idx, row in df.iterrows():
        signal = row['signal']
        close = row['close']
        date = row['timestamp']

        if signal == 1 and not in_position:  # BUY signal
            entry_price = close
            entry_idx = idx
            in_position = True

        elif signal == -1 and in_position:  # SELL signal
            exit_price = close
            pnl_pct = (exit_price - entry_price) / entry_price - (2 * tc_bps / 10000)
            hold_days = idx - entry_idx
            trades.append({
                'symbol': symbol,
                'entry_date': df.loc[entry_idx, 'timestamp'],
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'hold_days': hold_days,
            })
            in_position = False
            entry_price = None

    # Build equity curve
    df['position'] = 0.0
    in_pos = False
    entry_px = None
    for idx, row in df.iterrows():
        if row['signal'] == 1:
            in_pos = True
            entry_px = row['close']
        elif row['signal'] == -1:
            in_pos = False

        if in_pos:
            df.loc[idx, 'position'] = float((row['close'] - entry_px) / entry_px - (tc_bps / 10000))

    # Benchmark: buy and hold
    buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

    # Performance metrics
    trades_df = pd.DataFrame(trades)
    if len(trades) > 0:
        strategy_return = np.prod(1 + trades_df['pnl_pct']) - 1
        win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades)
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] < 0).any() else 0
    else:
        strategy_return = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0

    # Estimate annual return and Sharpe (simplified)
    years = (df['timestamp'].max() - df['timestamp'].min()).days / 365.25
    cagr = ((1 + strategy_return) ** (1 / years) - 1) if years > 0 else 0

    # Daily returns for Sharpe
    df['daily_return'] = df['position'].diff()
    daily_returns = df['daily_return'].dropna()
    daily_vol = daily_returns.std()
    sharpe = (daily_returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0

    metrics = {
        'symbol': symbol,
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'years': years,
        'trades': len(trades),
        'strategy_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': strategy_return - buy_hold_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }

    return metrics, trades_df


def backtest_universe(df, tc_bps=0, ema_fast=12, ema_slow=26):
    """
    Backtest all symbols in the universe.

    Args:
        df: DataFrame with OHLCV data for all symbols
        tc_bps: transaction cost in basis points
        ema_fast: fast EMA period (default 12)
        ema_slow: slow EMA period (default 26)

    Returns:
        (metrics_df, all_trades_df)
    """
    all_metrics = []
    all_trades = []

    for symbol in df['symbol'].unique():
        logger.info(f"Backtesting {symbol}...")
        symbol_df = df[df['symbol'] == symbol].copy()

        # Calculate EMAs and signals
        symbol_df = calculate_emas(symbol_df, ema_fast=ema_fast, ema_slow=ema_slow)
        symbol_df = generate_signals(symbol_df)

        # Backtest
        metrics, trades = backtest_symbol(symbol_df, tc_bps=tc_bps)
        all_metrics.append(metrics)
        if len(trades) > 0:
            all_trades.append(trades)

    metrics_df = pd.DataFrame(all_metrics)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return metrics_df, trades_df


def print_summary(metrics_df, trades_df):
    """Print backtest summary to console."""
    print("\n" + "="*100)
    print("EMA12/EMA26 CROSSOVER BACKTEST SUMMARY")
    print("="*100)

    print(f"\nUniverse: {len(metrics_df)} symbols")
    print(f"Date range: {metrics_df['start_date'].min()} to {metrics_df['end_date'].max()}")
    print(f"Total trades: {metrics_df['trades'].sum():.0f}")

    print("\n" + "-"*100)
    print("PER-SYMBOL PERFORMANCE")
    print("-"*100)

    display_cols = ['symbol', 'trades', 'cagr', 'sharpe', 'win_rate', 'strategy_return', 'buy_hold_return', 'excess_return']
    print(metrics_df[display_cols].to_string(index=False))

    print("\n" + "-"*100)
    print("SUMMARY STATISTICS")
    print("-"*100)
    print(f"Avg CAGR:        {metrics_df['cagr'].mean():.2%}")
    print(f"Avg Sharpe:      {metrics_df['sharpe'].mean():.2f}")
    print(f"Avg win rate:    {metrics_df['win_rate'].mean():.2%}")
    print(f"Profitable symbols: {(metrics_df['strategy_return'] > 0).sum()} / {len(metrics_df)}")

    print("\n" + "-"*100)
    print("TOP 5 BY CAGR")
    print("-"*100)
    top_cagr = metrics_df.nlargest(5, 'cagr')[['symbol', 'cagr', 'sharpe', 'trades', 'win_rate']]
    print(top_cagr.to_string(index=False))

    print("\n" + "-"*100)
    print("BOTTOM 5 BY CAGR")
    print("-"*100)
    bottom_cagr = metrics_df.nsmallest(5, 'cagr')[['symbol', 'cagr', 'sharpe', 'trades', 'win_rate']]
    print(bottom_cagr.to_string(index=False))

    if len(trades_df) > 0:
        print("\n" + "-"*100)
        print("SAMPLE TRADES (first 10)")
        print("-"*100)
        print(trades_df.head(10)[['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl_pct', 'hold_days']].to_string(index=False))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest EMA crossover on ETF universe")
    parser.add_argument("--data", default="ema_etfs_primary_daily.parquet",
                        help="Parquet file to load")
    parser.add_argument("--tc-bps", type=float, default=0,
                        help="Transaction cost in basis points (default 0)")
    parser.add_argument("--ema-fast", type=int, default=12,
                        help="Fast EMA period (default 12)")
    parser.add_argument("--ema-slow", type=int, default=26,
                        help="Slow EMA period (default 26)")
    parser.add_argument("--output", default="data/models/ema_backtest_results.csv",
                        help="Output CSV for metrics")
    args = parser.parse_args()

    # Load data
    df = load_etf_data(args.data)

    # Backtest
    logger.info(f"Running backtest: EMA({args.ema_fast},{args.ema_slow}) with tc={args.tc_bps}bps...")
    metrics_df, trades_df = backtest_universe(df, tc_bps=args.tc_bps, ema_fast=args.ema_fast, ema_slow=args.ema_slow)

    # Print summary
    print_summary(metrics_df, trades_df)

    # Save results
    output_dir = Path(__file__).parent.parent / args.output.split('/')[0] / args.output.split('/')[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    ema_label = f"ema{args.ema_fast}_{args.ema_slow}"
    metrics_path = output_dir / f"{ema_label}_metrics_{args.tc_bps:.0f}bps.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Saved metrics to {metrics_path}")

    if len(trades_df) > 0:
        trades_path = output_dir / f"{ema_label}_trades_{args.tc_bps:.0f}bps.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"✓ Saved trades to {trades_path}")


if __name__ == "__main__":
    main()
