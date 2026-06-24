"""Read compressible fields from VTK output.

Uses PyVista, so anything it can open works: OpenFOAM foamToVTK output, or VTK
written by another solver. That matters because high-speed labs often run their
own codes, not OpenFOAM. PyVista is an optional dependency imported lazily, so
the synthetic path and the analysers run without it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["read_vtk_slice", "to_uniform_grid", "FIELD_ALIASES"]

# Common names the same quantity goes by across solvers, so a request for "rho"
# still finds "density", "Rho", etc. Extend per solver as needed.
FIELD_ALIASES = {
    "rho": ["rho", "Rho", "density", "Density", "rhoMean"],
    "p": ["p", "P", "pressure", "Pressure", "pMean"],
    "U": ["U", "u", "velocity", "Velocity", "UMean"],
    "T": ["T", "Temperature", "temperature"],
    "nut": ["nut", "nuTilda", "muT"],
}


def _resolve(name, data):
    """Find the array for a canonical field name under any known alias."""
    for alias in FIELD_ALIASES.get(name, [name]):
        if alias in data:
            return data[alias]
    raise KeyError(f"field '{name}' not found; tried {FIELD_ALIASES.get(name, [name])}, "
                   f"have {list(data.keys())}")


def read_vtk_slice(path, fields=("rho", "p"), z=None):
    """Read a 2D slice of cell data into a dict of named point arrays.

    Field names are resolved through FIELD_ALIASES, so output from different
    solvers works without renaming. Returns {"points": (N,2), field: (N,...)}.
    For 3D data pass ``z`` to take a plane near that height; else the mid-plane.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading VTK needs pyvista: pip install shocklens[vtk]"
                          ) from exc

    mesh = pv.read(path)
    cell_names = {a for f in fields for a in FIELD_ALIASES.get(f, [f])}
    if cell_names & set(mesh.cell_data.keys()):
        centres = mesh.cell_centers()
        pts, data = np.asarray(centres.points), mesh.cell_data
    else:
        pts, data = np.asarray(mesh.points), mesh.point_data

    zc = pts[:, 2]
    z = float(np.median(zc)) if z is None else z
    keep = np.abs(zc - z) <= (np.ptp(zc) / 50 + 1e-9) if np.ptp(zc) > 0 else slice(None)
    out = {"points": pts[keep, :2]}
    for f in fields:
        out[f] = np.asarray(_resolve(f, data))[keep]
    return out


def to_uniform_grid(points, values, nx, ny):
    """Interpolate scattered values onto a uniform grid (linear + nearest fill)."""
    from scipy.interpolate import griddata
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), nx)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), ny)
    gy, gx = np.meshgrid(y, x, indexing="ij")
    tgt = np.column_stack([gx.ravel(), gy.ravel()])
    lin = griddata(points, values, tgt, method="linear")
    nn = griddata(points, values, tgt, method="nearest")
    grid = np.where(np.isnan(lin), nn, lin).reshape(ny, nx)
    return grid.astype(np.float32), x, y
