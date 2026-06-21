"""Track a shock over a sequence of fields.

This is the unsteady side of SBLI and the scalability path: point it at a list of
field snapshots (or a directory of VTK timesteps) and it returns the shock-foot
trajectory x_s(t) and the low-frequency breathing rate. The per-frame work reuses
detect.detect_oblique_shock, so tracking is just a loop, which means it scales to
as many timesteps as you can read.
"""

from __future__ import annotations

import numpy as np

from . import detect, separation

__all__ = ["track_fields", "track_vtk_series"]


def track_fields(times, fields, quantile=0.9):
    """Track shock foot and angle across in-memory field dicts.

    Each field is a dict with rho, x, y (as produced by synthetic or io). Returns
    a dict with t, x_foot, beta_deg arrays and the breathing frequency.
    """
    times = np.asarray(times, dtype=float)
    x_foot, beta = [], []
    for f in fields:
        got = detect.detect_oblique_shock(f["rho"], f["x"], f["y"], quantile)
        x_foot.append(got["x_foot"])
        beta.append(got["beta_deg"])
    x_foot, beta = np.asarray(x_foot), np.asarray(beta)

    fs = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 1.0
    f_breathing = separation.dominant_frequency(x_foot - x_foot.mean(), fs)
    return {"t": times, "x_foot": x_foot, "beta_deg": beta,
            "f_breathing": f_breathing, "fs": fs}


def track_vtk_series(paths, times=None, field="rho", nx=200, ny=120):
    """Track a shock across a list of VTK files (e.g. foamToVTK timesteps)."""
    from . import io
    fields = []
    for p in paths:
        raw = io.read_vtk_slice(p, fields=(field,))
        grid, x, y = io.to_uniform_grid(raw["points"], raw[field], nx, ny)
        fields.append({"rho": grid, "x": x, "y": y})
    if times is None:
        times = np.arange(len(paths), dtype=float)
    return track_fields(times, fields)
