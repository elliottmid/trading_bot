#!/bin/bash
# Complete EMA backtest pipeline: fetch → backtest → cost analysis

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"

echo "====================================="
echo "EMA12/EMA26 CROSSOVER BACKTEST PIPELINE"
echo "====================================="

cd "$SCRIPTS_DIR"

# Step 1: Fetch data
echo ""
echo "[1/3] Fetching daily data for primary universe (13 ETFs)..."
python3 fetch_ema_etfs.py --universe primary --start-date 2020-12-14

# Step 2: Backtest with no costs
echo ""
echo "[2/3] Backtesting with 0bp transaction costs..."
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 0

# Step 3: Backtest with 5bp costs
echo ""
echo "[3/3] Backtesting with 5bp transaction costs..."
python3 backtest_ema_crossover.py --data ema_etfs_primary_daily.parquet --tc-bps 5

echo ""
echo "====================================="
echo "✓ Pipeline complete!"
echo "====================================="
echo ""
echo "Results saved to:"
echo "  • data/raw/ema_etfs_primary_daily.parquet (raw data)"
echo "  • data/models/ema_metrics_0bps.csv (no costs)"
echo "  • data/models/ema_metrics_5bps.csv (5bp costs)"
echo "  • data/models/ema_trades_*.csv (trade logs)"
