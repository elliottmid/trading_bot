"""
signal_aggregator.py — Combine model output with optional external signals,
then gate through technical indicator filters.

Pipeline:
  1. Blend model confidence with optional external confidence score.
  2. Apply technical filters (RSI overbought, MACD direction) that can
     block or reduce confidence for a BUY signal.
  3. Return final (signal, confidence) pair.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from ..config import Config
from ..logger import get_logger

log = get_logger(__name__)

# RSI thresholds for filtering BUY signals.
_RSI_OVERBOUGHT = 70.0   # Block BUY if RSI >= this (momentum already exhausted)
_RSI_ELEVATED   = 60.0   # Reduce confidence if RSI >= this (caution zone)

# Confidence penalty applied when RSI is elevated but not overbought.
_RSI_PENALTY = 0.10


class SignalAggregator:
    """Blend internal model signals with optional external signals,
    then validate against technical indicator filters.

    Blend step (weighted average):
        blended = (1 - w) * model_confidence + w * external_confidence

    Technical filter step (applied to BUY signals only):
        - RSI >= 70  → signal blocked (overbought; poor swing entry)
        - RSI >= 60  → confidence reduced by _RSI_PENALTY
        - MACD < 0   → confidence reduced by _RSI_PENALTY (bearish momentum)

    Filters are skipped when *features* is not provided (e.g. in backtests
    that lack real-time indicator data).

    Args:
        config: Application configuration.
    """

    def __init__(self, config: Config) -> None:
        self._weight = config.signal_agreement_weight
        self._min_confidence = config.min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        model_signal: int,
        model_confidence: float,
        external_confidence: Optional[float] = None,
        features: Optional[pd.Series] = None,
    ) -> Tuple[int, float]:
        """Compute the final trade signal.

        Args:
            model_signal: Raw binary signal from the ML model (0 or 1).
            model_confidence: Model's predicted probability for the positive
                class (0.0–1.0).
            external_confidence: Optional agreement score from an external
                source (0.0–1.0, where 1.0 = fully bullish).
            features: Optional pandas Series of the latest technical
                indicators (expects keys ``RSI_14``, ``MACD``).  When
                provided, acts as a secondary filter on BUY signals.

        Returns:
            Tuple of ``(final_signal, final_confidence)``.  *final_signal*
            is 1 only if *final_confidence* >= min_confidence AND
            *model_signal* is 1 AND technical filters pass.
        """
        # Step 1 — blend model + external confidence
        blended = self._blend(model_confidence, external_confidence)

        # Step 2 — apply technical filters to BUY signals
        if model_signal == 1 and features is not None:
            blocked, blended = self._apply_technical_filters(blended, features)
            if blocked:
                log.debug("Signal blocked by technical filter for BUY.")
                return 0, blended

        # Step 3 — threshold gate
        final_signal = 1 if (model_signal == 1 and blended >= self._min_confidence) else 0
        log.debug(
            "Final signal: %d (confidence=%.3f, threshold=%.3f).",
            final_signal,
            blended,
            self._min_confidence,
        )
        return final_signal, blended

    def is_above_threshold(self, confidence: float) -> bool:
        """Return True if *confidence* meets the minimum trading threshold."""
        return confidence >= self._min_confidence

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _blend(
        self,
        model_confidence: float,
        external_confidence: Optional[float],
    ) -> float:
        """Weighted blend of model and external confidence scores."""
        if external_confidence is not None:
            blended = (
                (1 - self._weight) * model_confidence
                + self._weight * external_confidence
            )
            log.debug(
                "Signal blend: model=%.3f ext=%.3f weight=%.2f -> blended=%.3f.",
                model_confidence,
                external_confidence,
                self._weight,
                blended,
            )
            return blended
        return model_confidence

    def _apply_technical_filters(
        self,
        confidence: float,
        features: pd.Series,
    ) -> Tuple[bool, float]:
        """Apply RSI and MACD filters to a BUY signal.

        Args:
            confidence: Blended confidence score before filtering.
            features: Latest indicator row (expects ``RSI_14``, ``MACD``).

        Returns:
            Tuple of ``(blocked, adjusted_confidence)``.
            *blocked* is True when the signal should be discarded outright.
        """
        rsi  = features.get("RSI_14")
        macd = features.get("MACD")

        # --- RSI filter ---
        if rsi is not None:
            if rsi >= _RSI_OVERBOUGHT:
                log.info(
                    "BUY signal blocked: RSI=%.1f >= overbought threshold %.1f.",
                    rsi,
                    _RSI_OVERBOUGHT,
                )
                return True, confidence  # blocked

            if rsi >= _RSI_ELEVATED:
                confidence -= _RSI_PENALTY
                log.debug(
                    "BUY confidence reduced by %.2f: RSI=%.1f in elevated zone.",
                    _RSI_PENALTY,
                    rsi,
                )

        # --- MACD filter ---
        if macd is not None and macd < 0:
            confidence -= _RSI_PENALTY
            log.debug(
                "BUY confidence reduced by %.2f: MACD=%.4f is negative (bearish).",
                _RSI_PENALTY,
                macd,
            )

        return False, max(confidence, 0.0)  # clamp to non-negative
