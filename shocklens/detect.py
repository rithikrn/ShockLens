"""Shock detection from a density field.

Numerical schlieren is |grad rho|. The shock shows up as a bright ridge; fitting
a line to that ridge (weighted by gradient magnitude) gives the shock angle and,
where it meets the wall, the shock-foot location. v0.1 assumes one dominant
oblique shock with no competing gradients, which is what the synthetic case and
clean tutorials (forwardStep, wedge) provide. Boundary-layer and expansion
gradients are a v0.2 robustness problem.
"""

from __future__ import annotations

import numpy as np

__all__ = ["schlieren", "shock_points", "fit_shock_line", "detect_oblique_shock"]


def schlieren(rho, dx, dy):
    """Numerical schlieren field |grad rho| (same shape as rho)."""
    gy, gx = np.gradient(rho, dy, dx)
    return np.hypot(gx, gy)


def shock_points(rho, dx, dy, quantile=0.9):
    """Coordinates and weights of strong-gradient (shock) cells.

    Returns (ix, iy, weight) on grid-index coordinates so callers can map to
    physical space with their own x, y arrays.
    """
    s = schlieren(rho, dx, dy)
    thr = np.quantile(s[s > 0], quantile) if np.any(s > 0) else 0.0
    iy, ix = np.where(s >= thr)
    return ix, iy, s[iy, ix]


def fit_shock_line(x_pts, y_pts, weights=None):
    """Weighted least-squares line y = m x + b through shock points."""
    w = np.ones_like(x_pts) if weights is None else weights
    W = np.diag(w)
    A = np.column_stack([x_pts, np.ones_like(x_pts)])
    m, b = np.linalg.solve(A.T @ W @ A, A.T @ W @ y_pts)
    return float(m), float(b)


def detect_oblique_shock(rho, x, y, quantile=0.9):
    """Recover shock angle (deg) and wall foot location from a density field.

    Returns dict with beta_deg, x_foot, slope, intercept.
    """
    dx, dy = x[1] - x[0], y[1] - y[0]
    ix, iy, w = shock_points(rho, dx, dy, quantile)
    xs, ys = x[ix], y[iy]
    m, b = fit_shock_line(xs, ys, w)
    beta = np.rad2deg(np.arctan(abs(m)))
    x_foot = -b / m if m != 0 else float("nan")     # y = 0 crossing
    return {"beta_deg": beta, "x_foot": float(x_foot),
            "slope": m, "intercept": b}
