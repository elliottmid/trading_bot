"""Walk-forward backtest of an XGBoost regressor on N-day forward returns.

Rolling-origin evaluation: train on a window of TRAIN_MONTHS, test the next
TEST_MONTHS, roll forward by TEST_MONTHS. Reports out-of-sample metrics and
saves per-fold predictions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "models"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"ts", "symbol", "target_fwd_ret"}


def ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC — signal quality metric standard for return prediction."""
    return pd.Series(y_true).rank().corr(pd.Series(y_pred).rank())


def run(horizon: int, train_m: int, test_m: int, embargo_d: int, symbol: str | None = None):
    df = pd.read_parquet(PROC / f"features_daily_H{horizon}.parquet").sort_values("ts").reset_index(drop=True)
    tag = "all"
    if symbol:
        df = df[df["symbol"] == symbol].reset_index(drop=True)
        tag = symbol
        features = [c for c in df.columns if c not in EXCLUDE]
    else:
        df["symbol_code"] = df["symbol"].astype("category").cat.codes
        features = [c for c in df.columns if c not in EXCLUDE] + ["symbol_code"]

    t0, tN = df["ts"].min(), df["ts"].max()
    train_td = pd.Timedelta(days=int(train_m * 30.44))
    test_td = pd.Timedelta(days=int(test_m * 30.44))
    embargo = pd.Timedelta(days=embargo_d + horizon)  # prevent target leakage across split

    folds, preds = [], []
    anchor = t0 + train_td
    fold = 0
    while anchor + test_td <= tN:
        train = df[df["ts"] < anchor - embargo]
        test = df[(df["ts"] >= anchor) & (df["ts"] < anchor + test_td)]
        min_train = 200 if symbol else 500
        min_test = 20 if symbol else 50
        if len(train) < min_train or len(test) < min_test:
            anchor += test_td
            continue

        X_tr, y_tr = train[features], train["target_fwd_ret"]
        X_te, y_te = test[features], test["target_fwd_ret"]

        model = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=42,
        )
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_te)

        mae = mean_absolute_error(y_te, yhat)
        rank_ic = ic(y_te.values, yhat)
        # Directional hit-rate, long-only sign of prediction
        hit = (np.sign(yhat) == np.sign(y_te.values)).mean()
        folds.append({
            "fold": fold, "train_end": (anchor - embargo).date(), "test_start": anchor.date(),
            "test_end": (anchor + test_td).date(), "n_train": len(train), "n_test": len(test),
            "mae": mae, "rank_ic": rank_ic, "hit_rate": hit,
        })
        out = test[["ts", "symbol", "target_fwd_ret"]].copy()
        out["pred"] = yhat
        out["fold"] = fold
        preds.append(out)

        fold += 1
        anchor += test_td

    folds_df = pd.DataFrame(folds)
    preds_df = pd.concat(preds, ignore_index=True)
    folds_df.to_csv(OUT / f"wf_folds_H{horizon}_{tag}.csv", index=False)
    preds_df.to_parquet(OUT / f"wf_preds_H{horizon}_{tag}.parquet", index=False)

    # Feature importance from final fit (on all data up to last train_end)
    final_train = df[df["ts"] < anchor - embargo]
    if len(final_train) > 500:
        fm = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          tree_method="hist", n_jobs=-1, random_state=42)
        fm.fit(final_train[features], final_train["target_fwd_ret"])
        imp = pd.Series(fm.feature_importances_, index=features).sort_values(ascending=False)
        print("\n=== feature importance (final fit, top 15) ===")
        print(imp.head(15).to_string())

    print("\n=== walk-forward folds ===")
    print(folds_df.to_string(index=False))
    print("\n=== OOS aggregate ===")
    print(f"mean MAE:     {folds_df['mae'].mean():.5f}")
    print(f"mean rank-IC: {folds_df['rank_ic'].mean():+.4f}  (>0.03 is tradeable, >0.05 strong)")
    print(f"mean hit:     {folds_df['hit_rate'].mean():.3f}")
    print(f"\nfolds saved -> {OUT / f'wf_folds_H{horizon}.csv'}")
    print(f"preds saved -> {OUT / f'wf_preds_H{horizon}.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--embargo-days", type=int, default=5)
    ap.add_argument("--symbol", type=str, default=None, help="Filter to one symbol (e.g., SPY)")
    args = ap.parse_args()
    run(args.horizon, args.train_months, args.test_months, args.embargo_days, args.symbol)
