# src/calibration.py
"""Probability calibration for risk predictions."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class Calibrator:
    """Calibrate raw probabilities using isotonic or Platt scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._model = None
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit calibrator on validation predictions."""
        valid = ~(np.isnan(probs) | np.isnan(labels))
        p, y = probs[valid], labels[valid]
        if len(p) < 10:
            logger.warning("Too few samples for calibration")
            return self

        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(p, y)
        else:  # platt
            self._model = LogisticRegression()
            self._model.fit(p.reshape(-1, 1), y)

        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if not self._fitted:
            return probs
        if self.method == "isotonic":
            return self._model.predict(probs)
        else:
            return self._model.predict_proba(probs.reshape(-1, 1))[:, 1]


class CalibrationManager:
    """Manage calibrators for multiple models."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._calibrators: Dict[str, Calibrator] = {}

    def fit(self, key: str, probs: np.ndarray, labels: np.ndarray):
        cal = Calibrator(self.method)
        cal.fit(probs, labels)
        self._calibrators[key] = cal

    def transform(self, key: str, probs: np.ndarray) -> np.ndarray:
        if key in self._calibrators:
            return self._calibrators[key].transform(probs)
        return probs

    def get(self, key: str) -> Optional[Calibrator]:
        return self._calibrators.get(key)
