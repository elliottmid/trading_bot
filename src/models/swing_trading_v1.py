"""
swing_trading_v1.py — XGBoost swing-trading classifier (version 1).

Trains a gradient-boosted tree classifier on technical indicator features
to predict whether a stock will rise more than a threshold amount within
a defined holding horizon (default: 5 trading days).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ..logger import get_logger

log = get_logger(__name__)


class SwingTradingModel:
    """XGBoost binary classifier for swing-trading signals.

    The model predicts whether a stock's close price will increase by at
    least a target threshold within a forward-looking window.

    Args:
        lookback_days: Number of historical days used for training (informational).
        prediction_horizon: Number of days the model looks ahead.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        prediction_horizon: int = 5,
    ) -> None:
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon
        self._model: XGBClassifier = self._build_model()
        self._feature_names: list = []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model() -> XGBClassifier:
        """Instantiate the XGBClassifier with project-standard hyperparameters."""
        return XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            use_label_encoder=False,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the XGBoost classifier and return performance metrics.

        The data is split 80/20 (chronologically) into train and validation
        sets to avoid look-ahead bias.

        Args:
            X: Feature DataFrame (rows = observations, cols = features).
            y: Binary target Series aligned to X.

        Returns:
            Dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
            ``feature_importances`` (dict), ``n_train``, ``n_val``.
        """
        self._feature_names = list(X.columns)

        # Chronological split — no shuffling
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        log.info(
            "Training on %d samples, validating on %d samples.",
            len(X_train),
            len(X_val),
        )

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._is_trained = True

        y_pred = self._model.predict(X_val)
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "feature_importances": self._get_importances(),
        }

        log.info(
            "Train complete — acc=%.3f, prec=%.3f, rec=%.3f, f1=%.3f.",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        self._log_top_features(metrics["feature_importances"])
        return metrics

    def predict(self, X: pd.DataFrame) -> Tuple[int, float]:
        """Generate a trading signal for the most recent feature row.

        Args:
            X: Feature DataFrame.  Only the **last** row is used for inference.

        Returns:
            Tuple of ``(signal, confidence)`` where *signal* is 0 (no trade)
            or 1 (buy), and *confidence* is the model's predicted probability
            for the positive class.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained.  Call train() or load() first."
            )
        row = X.iloc[[-1]]
        proba = self._model.predict_proba(row)[0]
        confidence = float(proba[1])
        signal = int(confidence >= 0.5)
        log.debug("Predict: signal=%d, confidence=%.4f.", signal, confidence)
        return signal, confidence

    def save(self, path: str) -> None:
        """Persist the model to disk using joblib.

        Args:
            path: File path for the serialised model (e.g.
                ``"data/models/swing_v1.pkl"``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "lookback_days": self.lookback_days,
                "prediction_horizon": self.prediction_horizon,
            },
            path,
        )
        log.info("Model saved to %s.", path)

    @classmethod
    def load(cls, path: str) -> "SwingTradingModel":
        """Load a previously saved model from disk.

        Args:
            path: Path to the joblib-serialised model file.

        Returns:
            A trained SwingTradingModel instance.
        """
        payload = joblib.load(path)
        instance = cls(
            lookback_days=payload["lookback_days"],
            prediction_horizon=payload["prediction_horizon"],
        )
        instance._model = payload["model"]
        instance._feature_names = payload["feature_names"]
        instance._is_trained = True
        log.info("Model loaded from %s.", path)
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_importances(self) -> Dict[str, float]:
        """Return feature importances as a dict sorted descending."""
        importances = self._model.feature_importances_
        return dict(
            sorted(
                zip(self._feature_names, importances.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    def _log_top_features(self, importances: Dict[str, float], top_n: int = 10) -> None:
        """Log the top-N most important features."""
        top = list(importances.items())[:top_n]
        log.info("Top %d features by importance:", top_n)
        for rank, (feat, score) in enumerate(top, start=1):
            log.info("  %2d. %-25s %.4f", rank, feat, score)
