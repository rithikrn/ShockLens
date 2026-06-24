"""Run a case from a small YAML description, so porting to your own data is one file.

A case file points at a VTK (or a produced foamToVTK file), names the density
field if your solver calls it something unusual, and optionally gives the
freestream Mach and wedge/ramp angle so the detected shock angle is checked
against theta-beta-M. One command in, a reproducible scorecard out. The case file
doubles as the reproducibility record: commit it and anyone re-runs your result.
"""

from __future__ import annotations

from . import detect, io, synthetic

__all__ = ["load_case", "run_case"]


def load_case(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def run_case(cfg):
    """Extract SBLI events from the VTK named in a case dict; return a scorecard.

    Recognised keys: vtk (required), name, fields (default ["rho"]), detector
    (default "oblique_line"), nx, ny, and optional mach + theta_deg for the
    theta-beta-M angle check.
    """
    fields = cfg.get("fields", ["rho"])
    raw = io.read_vtk_slice(cfg["vtk"], fields=tuple(fields))
    grid, x, y = io.to_uniform_grid(raw["points"], raw[fields[0]],
                                    cfg.get("nx", 300), cfg.get("ny", 160))
    got = detect.detect({"rho": grid, "x": x, "y": y},
                        method=cfg.get("detector", "oblique_line"))
    out = {"case": cfg.get("name", cfg["vtk"]),
           "detector": cfg.get("detector", "oblique_line"),
           "beta_deg": round(got["beta_deg"], 3),
           "x_foot": round(got["x_foot"], 4)}
    if "mach" in cfg and "theta_deg" in cfg:
        beta_theory = synthetic.oblique_beta(cfg["mach"], cfg["theta_deg"])
        out["beta_theory_deg"] = round(float(beta_theory), 3)
        out["beta_error_deg"] = round(abs(got["beta_deg"] - beta_theory), 3)
    return out
