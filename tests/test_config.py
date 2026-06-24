import numpy as np
import pyvista as pv

import shocklens as sl


def test_run_case_recovers_angle(tmp_path):
    f = sl.synthetic.oblique_shock_field(nx=300, ny=160, mach=5.0, theta_deg=15.0,
                                         x_corner=0.0, xlim=(-0.15, 0.30),
                                         ylim=(0.0, 0.15))
    gy, gx = np.meshgrid(f["y"], f["x"], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cloud = pv.PolyData(pts)
    cloud["rho"] = f["rho"].ravel()
    p = tmp_path / "case.vtk"
    cloud.save(p)

    cfg = {"name": "t", "vtk": str(p), "fields": ["rho"],
           "mach": 5.0, "theta_deg": 15.0, "nx": 300, "ny": 160}
    out = sl.config.run_case(cfg)
    assert out["beta_error_deg"] < 2.0
    assert "beta_theory_deg" in out
