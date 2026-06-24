"""Shock detection from a density field.

Numerical schlieren is |grad rho|; the shock is the bright ridge. Detection is
pluggable: ``oblique_line`` fits a weighted line to that ridge, ``oblique_ransac``
fits it robustly and ignores competing gradients (the first step toward turbulent
fields). New methods (3D shock surfaces, multi-shock labelling) register with
``@register_detector`` and are reached through ``get_detector`` or the ``detect``
dispatcher, so adding a regime never touches the call sites. Array ops come from
the input's namespace (see shocklens.backend), so the same code runs on GPU.
"""

from __future__ import annotations

import numpy as np

from . import backend

__all__ = ["schlieren", "shock_points", "fit_shock_line",
           "detect_oblique_shock", "detect_oblique_shock_ransac",
           "register_detector", "get_detector", "available_detectors", "detect"]

_DETECTORS = {}


def register_detector(name):
    """Decorator: register a detector under ``name`` for get_detector/detect."""
    def deco(fn):
        _DETECTORS[name] = fn
        return fn
    return deco


def get_detector(name="oblique_line"):
    if name not in _DETECTORS:
        raise KeyError(f"unknown detector '{name}'; have {available_detectors()}")
    return _DETECTORS[name]


def available_detectors():
    return sorted(_DETECTORS)


def schlieren(rho, dx, dy):
    """Numerical schlieren field |grad rho| (same shape and namespace as rho)."""
    xp = backend.array_namespace(rho)
    gy, gx = xp.gradient(rho, dy, dx)
    return xp.hypot(gx, gy)


def shock_points(rho, dx, dy, quantile=0.9):
    """Index coordinates and weights of strong-gradient (shock) cells."""
    xp = backend.array_namespace(rho)
    s = schlieren(rho, dx, dy)
    pos = s[s > 0]
    thr = xp.quantile(pos, quantile) if pos.size else 0.0
    iy, ix = xp.where(s >= thr)
    return ix, iy, s[iy, ix]


def fit_shock_line(x_pts, y_pts, weights=None):
    """Weighted least-squares line y = m x + b via the normal equations.

    Uses A^T W A without forming the dense N x N weight matrix, so it scales to
    the large point sets that fine or 3D fields produce.
    """
    x = backend.to_numpy(x_pts)
    y = backend.to_numpy(y_pts)
    w = np.ones_like(x) if weights is None else backend.to_numpy(weights)
    A = np.column_stack([x, np.ones_like(x)])
    ATA = A.T @ (w[:, None] * A)
    ATy = A.T @ (w * y)
    m, b = np.linalg.solve(ATA, ATy)
    return float(m), float(b)


def _angle_and_foot(m, b):
    beta = float(np.rad2deg(np.arctan(abs(m))))
    x_foot = float(-b / m) if m != 0 else float("nan")
    return {"beta_deg": beta, "x_foot": x_foot, "slope": float(m), "intercept": float(b)}


@register_detector("oblique_line")
def detect_oblique_shock(rho, x, y, quantile=0.9):
    """Default detector: weighted line fit to the schlieren ridge.

    Returns dict with beta_deg, x_foot, slope, intercept.
    """
    dx, dy = x[1] - x[0], y[1] - y[0]
    ix, iy, w = shock_points(rho, dx, dy, quantile)
    xs = backend.to_numpy(x)[backend.to_numpy(ix)]
    ys = backend.to_numpy(y)[backend.to_numpy(iy)]
    m, b = fit_shock_line(xs, ys, w)
    return _angle_and_foot(m, b)


@register_detector("oblique_ransac")
def detect_oblique_shock_ransac(rho, x, y, quantile=0.9, residual_threshold=None):
    """Robust detector: RANSAC line fit that rejects off-ridge outliers.

    More tolerant of boundary-layer and expansion gradients than the plain fit,
    which is what real (turbulent) fields need. Same return schema.
    """
    from sklearn.linear_model import RANSACRegressor
    dx, dy = x[1] - x[0], y[1] - y[0]
    ix, iy, _ = shock_points(rho, dx, dy, quantile)
    xs = backend.to_numpy(x)[backend.to_numpy(ix)].reshape(-1, 1)
    ys = backend.to_numpy(y)[backend.to_numpy(iy)]
    r = RANSACRegressor(random_state=0, residual_threshold=residual_threshold).fit(xs, ys)
    return _angle_and_foot(float(r.estimator_.coef_[0]), float(r.estimator_.intercept_))


def detect(field, method="oblique_line", **kw):
    """Dispatch detection on a field dict {rho, x, y} by method name."""
    return get_detector(method)(field["rho"], field["x"], field["y"], **kw)
