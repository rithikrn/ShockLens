"""Read compressible fields from VTK output.

Uses PyVista, so anything it can open works: OpenFOAM foamToVTK output, or VTK
written by another solver. That matters because high-speed labs often run their
own codes, not OpenFOAM. PyVista is an optional dependency imported lazily, so
the synthetic path and the analysers run without it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["read_vtk_slice", "to_uniform_grid"]


def read_vtk_slice(path, fields=("rho", "p"), z=None):
    """Read a 2D slice of cell data into a dict of named point arrays.

    Returns {"points": (N,2), field: (N,...)}. For 3D data pass ``z`` to take a
    plane near that height; otherwise the mid-plane is used.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading VTK needs pyvista: pip install shocklens[vtk]"
                          ) from exc

    mesh = pv.read(path)
    if all(f in mesh.cell_data for f in fields):
        centres = mesh.cell_centers()
        pts, data = np.asarray(centres.points), mesh.cell_data
    else:
        pts, data = np.asarray(mesh.points), mesh.point_data

    zc = pts[:, 2]
    z = float(np.median(zc)) if z is None else z
    keep = np.abs(zc - z) <= (np.ptp(zc) / 50 + 1e-9) if np.ptp(zc) > 0 else slice(None)
    out = {"points": pts[keep, :2]}
    for f in fields:
        out[f] = np.asarray(data[f])[keep]
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
