"""
model_utils.py — Shared model utility functions.

Helpers for walk-forward splitting, cross-validation scoring, and
feature-selection utilities used across training scripts.
"""

from __future__ import annotations

from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..logger import get_logger

log = get_logger(__name__)


def walk_forward_splits(
    df: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 20,
    step: int = 20,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Generate rolling (walk-forward) train/test index pairs.

    Each split uses a fixed-size training window immediately preceding a
    fixed-size test window, stepped forward by *step* rows at a time.

    Args:
        df: The full feature + target DataFrame.
        train_window: Number of rows in each training window.
        test_window: Number of rows in each test window.
        step: Number of rows to advance the window per split.

    Yields:
        Tuples of (train_df, test_df) sliced from *df*.
    """
    n = len(df)
    start = train_window
    while start + test_window <= n:
        train = df.iloc[start - train_window : start]
        test = df.iloc[start : start + test_window]
        yield train, test
        start += step


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute a standard suite of classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for the positive class (optional).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc (if proba
        provided), positive_rate.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_true)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def select_features(
    df: pd.DataFrame,
    target_col: str = "target",
    exclude_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y.

    Args:
        df: Combined features + target DataFrame.
        target_col: Name of the target column.
        exclude_cols: Additional columns to drop from X (e.g. OHLCV originals).

    Returns:
        Tuple of (X, y).
    """
    drop = [target_col] + (exclude_cols or [])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[target_col]
    return X, y


def print_metrics_table(metrics: dict, title: str = "Metrics") -> None:
    """Pretty-print a metrics dict to stdout.

    Args:
        metrics: Dict of metric_name -> float value.
        title: Header string for the table.
    """
    header = "=" * 40
    print(header)
    print(title.center(40))
    print(header)
    for key, val in metrics.items():
        if key == "feature_importances":
            continue
        if isinstance(val, float):
            print("  %-20s %.4f" % (key, val))
        else:
            print("  %-20s %s" % (key, val))
    print(header)
