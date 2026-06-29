import numpy as np
import pyvista as pv

import shocklens as sl
from shocklens.config import run_case


def _field_vtk(path):
    f = sl.synthetic.compression_ramp_field(mach=2.5, theta_deg=15.0)
    gy, gx = np.meshgrid(f["y"], f["x"], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    c = pv.PolyData(pts)
    c["rho"] = f["rho"].ravel()
    c.save(path)
    return f


def _wall_vtk(path, f):
    # inclined ramp wall (y = tan(15)*x for x>0) with TWO z-layers (2D slab)
    x = np.linspace(f["x"][0], f["x"][-1], 200)
    y = np.where(x > 0, np.tan(np.deg2rad(15)) * x, 0.0)
    cf = sl.synthetic.ramp_cf_profile(x, f["x_sep"], f["x_reatt"])
    tang = np.column_stack([np.gradient(x), np.gradient(y)])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    tau = (cf * 2.8)[:, None] * tang                 # wall shear along the tangent
    p = 1.0 + 1.5 * (x > f["x_sep"])                 # pressure rise across interaction
    xx = np.concatenate([x, x])                      # duplicate z-layers
    yy = np.concatenate([y, y])
    z = np.concatenate([np.zeros_like(x), np.full_like(x, 0.05)])
    w = pv.PolyData(np.column_stack([xx, yy, z]))
    w["wallShearStress"] = np.vstack([np.column_stack([tau, np.zeros(len(x))])] * 2)
    w["p"] = np.concatenate([p, p])
    w.save(path)


def _probe(path):
    t = np.arange(0, 30, 1 / 200.0)
    p = sl.synthetic.wall_pressure_signal(t, f_breathing=0.7, noise=0.04)
    np.savetxt(path, np.column_stack([t, p, p]))


def test_full_pipeline_rich(tmp_path):
    vtk, wall, probe = tmp_path / "c.vtk", tmp_path / "w.vtk", tmp_path / "p"
    f = _field_vtk(str(vtk))
    _wall_vtk(str(wall), f)
    _probe(str(probe))
    cfg = {"name": "t", "vtk": str(vtk), "fields": ["rho"], "detector": "oblique_ransac",
           "nx": 320, "ny": 180, "mach": 2.5, "theta_deg": 15.0,
           "wall_vtk": str(wall), "cf_denom": 2.8, "probe_file": str(probe), "probe_col": 1}
    out = run_case(cfg, outdir=str(tmp_path / "r"))
    # rich shock strength
    assert out["shock"]["pressure_ratio"] > 1
    assert out["shock"]["beta_error_deg"] < 2
    # separation recovered through tangent projection + z-dedup
    assert out["separation"]["separated"] is True
    assert out["separation"]["cf_min"] < 0
    # wall-pressure rise from the wall patch
    assert out["wall_pressure_rise"]["rise_ratio"] > 1
    # unsteady spectrum
    assert abs(out["wall_pressure_spectrum"]["breathing_freq"] - 0.7) < 0.2
    # all figures present
    assert {"contour", "schlieren", "overlay", "cf", "wall_pressure_x",
            "wall_pressure_spectrum"} <= set(out["figures"])


def test_missing_optionals_warn_not_crash(tmp_path):
    vtk = tmp_path / "c.vtk"
    _field_vtk(str(vtk))
    out = run_case({"name": "t", "vtk": str(vtk), "fields": ["rho"]})
    assert "shock" in out
    assert any("separation not computed" in w for w in out["warnings"])
    assert any("wall-pressure spectrum not computed" in w for w in out["warnings"])
