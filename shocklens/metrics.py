"""SBLI scorecard.

Extraction metrics check the detectors against ground truth (shock-angle error,
separation-point error). Prediction metrics score the ML baseline (MAE and R2).
These are the SBLI-specific numbers that matter, not a global field RMSE.
"""

from __future__ import annotations

import numpy as np

__all__ = ["shock_angle_error", "point_error", "regression_scores", "scorecard"]


def shock_angle_error(detected_deg, true_deg):
    return abs(detected_deg - true_deg)


def point_error(detected, true):
    return abs(detected - true)


def regression_scores(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    mae = float(np.mean(np.abs(pred - true)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or 1e-12
    return {"mae": mae, "r2": 1 - ss_res / ss_tot}


def scorecard(extraction=None, prediction=None):
    """Assemble a flat dict of whichever metrics were supplied."""
    out = {}
    if extraction:
        out.update({f"extract_{k}": v for k, v in extraction.items()})
    if prediction:
        out.update({f"predict_{k}": v for k, v in prediction.items()})
    return out
