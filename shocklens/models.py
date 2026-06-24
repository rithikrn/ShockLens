"""Sparse-sensor ML baseline.

Deliberately small: a random forest that maps wall-pressure sensor readings to an
SBLI target (separation length, shock-foot location). The point of v0.1 is to
show the pipeline works end to end and that sparse sensors carry the signal, not
to win an accuracy contest. Temporal models (LSTM/TCN) and neural operators come
once the extraction is mature.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

__all__ = ["SparseSensorModel", "build_xy"]


def build_xy(dataset, target="L_sep"):
    """Stack sensor vectors (X) and a chosen target (y) from a case list."""
    X = np.stack([c["sensors"] for c in dataset]).astype(np.float64)
    y = np.array([c[target] for c in dataset], dtype=np.float64)
    return X, y


class SparseSensorModel:
    """Random-forest regressor over wall-pressure sensors."""

    def __init__(self, target="L_sep", n_estimators=200, seed=0):
        self.target = target
        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           random_state=seed)

    def fit(self, dataset):
        X, y = build_xy(dataset, self.target)
        self.model.fit(X, y)
        return self

    def predict(self, dataset):
        X, _ = build_xy(dataset, self.target)
        return self.model.predict(X)

    def predict_with_uncertainty(self, dataset):
        """Return (mean, std) using the spread across the forest's trees.

        The per-tree disagreement is a cheap, honest error bar: wide where the
        sensors are uninformative, tight where they pin the target down.
        """
        X, _ = build_xy(dataset, self.target)
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def sensor_importance(self):
        """Per-sensor importance, useful for sensor-placement studies."""
        return self.model.feature_importances_

    def top_sensors(self, k):
        """Indices of the k most informative sensors (placement guidance)."""
        return list(np.argsort(self.sensor_importance())[::-1][:k])

    def save(self, path):
        import joblib
        joblib.dump({"target": self.target, "model": self.model}, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(target=blob["target"])
        obj.model = blob["model"]
        return obj
