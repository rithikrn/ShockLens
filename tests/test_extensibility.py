import numpy as np

import shocklens as sl


def test_both_detectors_recover_angle():
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0)
    for method in ("oblique_line", "oblique_ransac"):
        got = sl.detect.detect(f, method=method)
        assert abs(got["beta_deg"] - f["beta_deg"]) < 2.5, method


def test_ransac_survives_scattered_outliers():
    # Scattered gradient spikes (numerical noise) should be rejected; a coherent
    # competing line would not be, and that honestly needs multi-shock labelling.
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=20.0)
    rho = f["rho"].copy()
    rng = np.random.default_rng(0)
    iy = rng.integers(0, rho.shape[0], 150)
    ix = rng.integers(0, rho.shape[1], 150)
    rho[iy, ix] += rng.uniform(0.3, 0.6, 150)      # scattered spikes
    field = {"rho": rho, "x": f["x"], "y": f["y"]}
    got = sl.detect.detect(field, method="oblique_ransac")
    assert abs(got["beta_deg"] - f["beta_deg"]) < 3.0


def test_detector_registry():
    names = sl.detect.available_detectors()
    assert {"oblique_line", "oblique_ransac"} <= set(names)
    try:
        sl.detect.get_detector("nope")
        assert False
    except KeyError:
        pass


def test_backend_namespace_is_numpy_for_numpy():
    assert sl.backend.array_namespace(np.zeros(3)) is np


def test_line_fit_scales_without_dense_weight_matrix():
    # Many points must not blow up (regression on the old O(N^2) np.diag path).
    n = 50000
    x = np.linspace(0, 1, n)
    y = 0.5 * x + 0.1
    m, b = sl.detect.fit_shock_line(x, y, np.ones(n))
    assert abs(m - 0.5) < 1e-6 and abs(b - 0.1) < 1e-6


def test_separation_field_spanwise():
    x = np.linspace(0, 1, 200)
    rows = [sl.synthetic.ramp_cf_profile(x, 0.4 + 0.02 * k, 0.6 + 0.02 * k)
            for k in range(4)]
    out = sl.separation.separation_field(x, np.array(rows))
    assert out["L_sep"].shape == (4,)
    assert np.all(out["L_sep"] > 0.15)


def test_field_aliases_resolve(tmp_path):
    import pyvista as pv
    f = sl.synthetic.oblique_shock_field()
    gy, gx = np.meshgrid(f["y"], f["x"], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cloud = pv.PolyData(pts)
    cloud["density"] = f["rho"].ravel()          # aliased name, not "rho"
    p = tmp_path / "aliased.vtk"
    cloud.save(p)
    raw = sl.io.read_vtk_slice(str(p), fields=("rho",))   # ask for canonical
    assert raw["rho"].shape[0] == pts.shape[0]
