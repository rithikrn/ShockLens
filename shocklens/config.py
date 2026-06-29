"""Run a whole case from one small YAML file, so porting is one file and no script.
"""

from __future__ import annotations

import os

import numpy as np

from . import detect, io, rankinehugoniot, separation, synthetic

__all__ = ["load_case", "run_case"]


def load_case(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _shock(cfg):
    fields = cfg.get("fields", ["rho"])
    raw = io.read_vtk_slice(cfg["vtk"], fields=tuple(fields))
    grid, x, y = io.to_uniform_grid(raw["points"], raw[fields[0]],
                                    cfg.get("nx", 300), cfg.get("ny", 160))
    field = {"rho": grid, "x": x, "y": y, "dx": x[1] - x[0], "dy": y[1] - y[0]}
    got = detect.detect(field, method=cfg.get("detector", "oblique_line"))
    return field, got


def _patch_fetch(mesh, name):
    """Coordinates and array for a patch field, handling point or cell data."""
    if name in mesh.point_data:
        return np.asarray(mesh.points), np.asarray(mesh[name])
    if name in mesh.cell_data:
        return np.asarray(mesh.cell_centers().points), np.asarray(mesh.cell_data[name])
    return None, None


def _collapse_by_x(x, arr):
    """Average duplicate-x points (2D-slab front/back z-layers), sorted by x."""
    key = np.round(x, 6)
    ux = np.unique(key)
    out = np.array([arr[key == u].mean(axis=0) for u in ux])
    xs = np.array([x[key == u].mean() for u in ux])
    o = np.argsort(xs)
    return xs[o], out[o]


def _wall_profile(cfg):
    """Tangent-projected Cf and (if present) wall pressure along the wall patch."""
    import pyvista as pv
    m = pv.read(cfg["wall_vtk"])
    shear_name = cfg.get("wall_shear_field", "wallShearStress")
    pts, tau = _patch_fetch(m, shear_name)
    if pts is None:
        raise KeyError(f"{shear_name} not on wall_vtk")
    x0, y0 = pts[:, 0], pts[:, 1]
    x, tau = _collapse_by_x(x0, tau)
    _, y = _collapse_by_x(x0, y0)
    tx, ty = np.gradient(x), np.gradient(y)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    tau_t = tau[:, 0] * tx + tau[:, 1] * ty            # project onto wall tangent
    cf = tau_t / float(cfg["cf_denom"])
    pw = None
    for pname in ("p", cfg.get("wall_pressure_field", "p")):
        _, parr = _patch_fetch(m, pname)
        if parr is not None:
            _, pw = _collapse_by_x(x0, parr)
            break
    return x, cf, pw


def _wall_pressure_time(cfg):
    d = np.loadtxt(cfg["probe_file"])
    t = d[:, 0]
    p = d[:, int(cfg.get("probe_col", 1))]
    fs = 1.0 / np.mean(np.diff(t))
    pf = p - p.mean()
    return t, p, fs, separation.pressure_rms(pf), separation.dominant_frequency(pf, fs)


def run_case(cfg, outdir=None):
    """Turn a case dict into a rich results summary, with figures and honesty checks.

    Required: vtk. Optional: name, fields, detector, nx, ny, mach, theta_deg,
    wall_vtk + cf_denom (separation + wall-pressure rise), probe_file + probe_col
    (unsteady wall pressure), outdir (write figures). Never raises on a missing
    optional input, it records a warning instead.
    """
    outdir = outdir or cfg.get("outdir")
    warnings = []
    field, got = _shock(cfg)
    out = {"case": cfg.get("name", cfg["vtk"]),
           "detector": cfg.get("detector", "oblique_line"),
           "shock": {"beta_deg": round(got["beta_deg"], 3),
                     "x_foot": round(got["x_foot"], 4)}}

    # shock strength + theta-beta-M honesty check
    if "mach" in cfg:
        Mn1 = cfg["mach"] * np.sin(np.deg2rad(got["beta_deg"]))
        if Mn1 > 1:
            jump = rankinehugoniot.oblique_shock(cfg["mach"], beta_deg=got["beta_deg"])
            out["shock"]["pressure_ratio"] = round(jump["p_ratio"], 3)
            out["shock"]["density_ratio"] = round(jump["rho_ratio"], 3)
            out["shock"]["M2"] = round(jump["M2"], 3)
        if "theta_deg" in cfg:
            beta_theory = float(synthetic.oblique_beta(cfg["mach"], cfg["theta_deg"]))
            err = abs(got["beta_deg"] - beta_theory)
            out["shock"]["beta_theory_deg"] = round(beta_theory, 3)
            out["shock"]["beta_error_deg"] = round(err, 3)
            if err > 5:
                warnings.append("shock angle is >5 deg from theta-beta-M; expected for "
                                "a separated viscous ramp, but check mesh resolution if "
                                "you expected an attached inviscid shock")

    # separation + wall-pressure rise from the wall patch
    x = cf = pw = None
    if cfg.get("wall_vtk") and cfg.get("cf_denom"):
        try:
            x, cf, pw = _wall_profile(cfg)
            sep = separation.separation_points(x, cf)
            out["separation"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in sep.items()}
            out["separation"]["cf_min"] = round(float(np.min(cf)), 5)
            if not sep["separated"]:
                warnings.append("no separation found (Cf never goes negative); raise the "
                                "ramp angle or Reynolds number if you expected a bubble")
            if np.mean(cf) < 0:
                warnings.append("Cf is mostly negative; your solver may use the opposite "
                                "wall-shear sign convention (negate cf_denom)")
            if pw is not None:
                span = x[-1] - x[0]
                p_up = float(np.mean(pw[x < x[0] + 0.2 * span]))
                p_peak = float(np.max(pw))
                out["wall_pressure_rise"] = {"p_upstream": round(p_up, 5),
                                             "p_peak": round(p_peak, 5),
                                             "rise_ratio": round(p_peak / p_up, 3)}
            else:
                warnings.append("wall pressure distribution not found on wall_vtk; "
                                "pressure rise skipped")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"separation skipped: could not read wall_vtk ({e})")
    else:
        warnings.append("separation not computed: add wall_vtk + cf_denom to the case file")

    # unsteady wall pressure from probes
    tt = pp = fs = None
    if cfg.get("probe_file"):
        try:
            tt, pp, fs, rms, fb = _wall_pressure_time(cfg)
            out["wall_pressure_spectrum"] = {"rms": round(rms, 5),
                                             "breathing_freq": round(fb, 4),
                                             "fs": round(fs, 2)}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"wall-pressure spectrum skipped: could not read probe_file ({e})")
    else:
        warnings.append("wall-pressure spectrum not computed: add probe_file to the case file")

    # figures
    if outdir:
        from . import plots
        os.makedirs(outdir, exist_ok=True)
        figs = {"contour": plots.plot_field(field, "rho", f"{outdir}/contour_rho.png",
                                            title=f"{out['case']}: density", shock=got),
                "schlieren": plots.plot_schlieren(field, f"{outdir}/schlieren.png"),
                "overlay": plots.plot_detection_overlay(field, got,
                                                       f"{outdir}/shock_overlay.png")}
        if x is not None and cf is not None:
            figs["cf"] = plots.plot_cf(x, cf, f"{outdir}/skin_friction.png")
            if pw is not None:
                sep = out.get("separation", {})
                figs["wall_pressure_x"] = plots.plot_wall_distribution(
                    x, pw, f"{outdir}/wall_pressure_x.png",
                    x_sep=sep.get("x_sep"), x_reatt=sep.get("x_reatt"))
        if tt is not None:
            figs["wall_pressure_spectrum"] = plots.plot_wall_pressure(
                tt, pp, fs, f"{outdir}/wall_pressure_spectrum.png")
        out["figures"] = figs

    out["warnings"] = warnings
    return out
