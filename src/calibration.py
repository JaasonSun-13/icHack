"""Isotonic regression for probability calibration."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, raw_probs, labels):
        if len(raw_probs) < 10:
            return
        self.iso.fit(raw_probs, labels)
        self.fitted = True

    def transform(self, raw_probs):
        if not self.fitted:
            return raw_probs
        return self.iso.predict(raw_probs)
