"""Forecast shock motion from its recent history.

This is the ML that matters operationally: given the last few shock-foot
positions (or sparse wall-pressure readings), predict where the shock goes next.
That is the early-warning signal for separation, unstart, and buffet. The
trajectory comes from track.track_fields, so this model trains on the toolkit's
own extracted output.

v0.1 uses gradient boosting on lagged features, an autoregressive setup that
needs no deep-learning stack. LSTM/TCN variants are a later, optional upgrade.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

__all__ = ["windowed_xy", "ShockForecaster", "persistence_forecast"]


def windowed_xy(series, window=8, horizon=1):
    """Turn a 1D series into (X, y) for h-step-ahead autoregression."""
    series = np.asarray(series, dtype=float)
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


class ShockForecaster:
    """Predict shock position h steps ahead from a window of past positions."""

    def __init__(self, window=8, horizon=1, seed=0):
        self.window = window
        self.horizon = horizon
        self.model = GradientBoostingRegressor(random_state=seed)

    def fit(self, series):
        X, y = windowed_xy(series, self.window, self.horizon)
        self.model.fit(X, y)
        return self

    def predict_next(self, recent):
        """One prediction from the most recent `window` samples."""
        recent = np.asarray(recent, dtype=float)[-self.window:]
        return float(self.model.predict(recent.reshape(1, -1))[0])

    def forecast(self, series):
        """h-step-ahead prediction at every valid point of a series."""
        X, _ = windowed_xy(series, self.window, self.horizon)
        return self.model.predict(X)


def persistence_forecast(series, window=8, horizon=1):
    """Naive baseline: predict that the shock stays where it last was.

    A forecaster only earns its keep if it beats this. mlcard reports the skill
    score 1 - MSE_model / MSE_persistence.
    """
    series = np.asarray(series, dtype=float)
    X, _ = windowed_xy(series, window, horizon)
    return X[:, -1]   # last observed value in each window
