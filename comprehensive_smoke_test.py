"""End-to-end smoke test, including the VTK read path the unit tests skip.

Writes a synthetic oblique-shock field to a real .vtk, reads it back through the
PyVista path, regrids, and checks the detector recovers the analytic angle.

Run: python comprehensive_smoke_test.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

import shocklens as sl

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def write_shock_vtk(path, mach, theta):
    f = sl.synthetic.oblique_shock_field(nx=240, ny=140, mach=mach, theta_deg=theta)
    gy, gx = np.meshgrid(f["y"], f["x"], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cloud = pv.PolyData(pts)
    cloud["rho"] = f["rho"].ravel()
    cloud["p"] = f["p"].ravel()
    cloud.save(path)
    return f["beta_deg"]


def main():
    tmp = Path(tempfile.mkdtemp())

    # 1. Detector vs analytic angle across conditions.
    ok = True
    for mach, theta in [(2.0, 8.0), (3.0, 15.0), (4.0, 22.0), (5.0, 25.0)]:
        f = sl.synthetic.oblique_shock_field(mach=mach, theta_deg=theta)
        got = sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])
        ok = ok and abs(got["beta_deg"] - f["beta_deg"]) < 2.0
    check("Shock-angle detection matches theta-beta-M theory", ok)

    # 2. Real VTK round trip: write, read, regrid, detect.
    vtk = tmp / "shock.vtk"
    beta_true = write_shock_vtk(vtk, mach=3.0, theta=15.0)
    raw = sl.io.read_vtk_slice(vtk, fields=("rho", "p"))
    rho, x, y = sl.io.to_uniform_grid(raw["points"], raw["rho"], 200, 120)
    got = sl.detect.detect_oblique_shock(rho, x, y)
    check("VTK read -> regrid -> detect recovers angle", abs(got["beta_deg"] - beta_true) < 2.5,
          f"{got['beta_deg']:.1f} vs {beta_true:.1f}")

    # 3. Separation + reattachment from a ramp Cf.
    xx = np.linspace(0, 1, 400)
    cf = sl.synthetic.ramp_cf_profile(xx, 0.42, 0.61)
    sep = sl.separation.separation_points(xx, cf)
    check("Separation/reattachment recovered from Cf",
          abs(sep["x_sep"] - 0.42) < 0.01 and abs(sep["x_reatt"] - 0.61) < 0.01,
          f"L_sep={sep['L_sep']:.3f}")

    # 4. Shock-breathing frequency from a Welch PSD.
    fs = 200.0
    t = np.arange(0, 60, 1 / fs)
    p = sl.synthetic.wall_pressure_signal(t, f_breathing=0.8, noise=0.05)
    f0 = sl.separation.dominant_frequency(p, fs)
    check("Shock-breathing frequency from PSD", abs(f0 - 0.8) < 0.1, f"{f0:.3f} Hz")

    # 5. Sparse-sensor prediction generalises to interior cases.
    data = sl.synthetic.make_sbli_dataset(
        ramp_angles=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    test_idx = {3, 6}
    train = [c for i, c in enumerate(data) if i not in test_idx]
    test = [data[i] for i in test_idx]
    model = sl.SparseSensorModel("L_sep").fit(train)
    r2 = sl.metrics.regression_scores(model.predict(test),
                                      [c["L_sep"] for c in test])["r2"]
    check("Sparse sensors predict separation length", r2 > 0.5, f"R2={r2:.2f}")

    # 6. Most-informative sensor is near the interaction, not the inlet.
    imp = model.sensor_importance()
    check("Sensor-importance is concentrated, not uniform",
          imp.max() > 1.5 * imp.mean(), f"max/mean={imp.max()/imp.mean():.1f}")

    # 7. Track an oscillating shock and recover the breathing frequency.
    t, fields, truth = sl.synthetic.oblique_shock_timeseries(
        n_frames=200, fs=100.0, f_breathing=2.0)
    tr = sl.track.track_fields(t, fields)
    check("Shock tracking recovers breathing frequency",
          abs(tr["f_breathing"] - 2.0) < 0.25, f"{tr['f_breathing']:.2f} Hz")

    # 8. Model round-trips through disk unchanged.
    mp = tmp / "model.joblib"
    model.save(mp)
    reloaded = sl.SparseSensorModel.load(mp)
    same = np.allclose(model.predict(test), reloaded.predict(test))
    check("Trained model save/load is exact", same)

    # 9. The real wedge case (Mach 5, 15 deg) through the actual extract CLI.
    #    OpenFOAM produces the VTK on your machine; here we generate the exact
    #    M=5/theta=15 field on the wedge domain and run the installed CLI on it.
    import subprocess
    beta_theory = sl.synthetic.oblique_beta(5.0, 15.0)
    wf = sl.synthetic.oblique_shock_field(
        nx=300, ny=160, mach=5.0, theta_deg=15.0, x_corner=0.0,
        xlim=(-0.15242, 0.3048), ylim=(0.0, 0.1524))
    gy, gx = np.meshgrid(wf["y"], wf["x"], indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cloud = pv.PolyData(pts)
    cloud["rho"] = wf["rho"].ravel()
    cloud["p"] = wf["p"].ravel()
    wvtk = tmp / "wedge15Ma5.vtk"
    cloud.save(wvtk)
    out = subprocess.run(["shocklens", "extract", str(wvtk), "--nx", "300", "--ny", "160"],
                         capture_output=True, text=True)
    import json as _json
    got = _json.loads(out.stdout)
    check("Wedge case (M5, 15deg) via real extract CLI matches theory",
          abs(got["beta_deg"] - beta_theory) < 2.0,
          f"detected {got['beta_deg']:.1f} vs theory {beta_theory:.1f}")

    # 10. Forecast shock motion from extracted trajectory.
    t2, fields2, _ = sl.synthetic.oblique_shock_timeseries(n_frames=300, fs=100.0)
    series = sl.track.track_fields(t2, fields2)["x_foot"]
    fc = sl.forecast.ShockForecaster(window=10).fit(series[:200])
    from shocklens.forecast import windowed_xy
    _, yv = windowed_xy(series, 10)
    n_test = len(yv) - 190
    r2 = sl.metrics.regression_scores(fc.forecast(series)[-n_test:], yv[-n_test:])["r2"]
    check("Shock-motion forecaster predicts held-out trajectory", r2 > 0.6, f"R2={r2:.2f}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
