"""ShockLens: turn shock-dominated CFD into interpretable SBLI events.

OpenFOAM (or any VTK) field -> physics extraction -> sparse-sensor ML -> SBLI
scorecard. Runs offline on synthetic, ground-truth-labelled data.

Quickstart::

    import shocklens as sl

    # detect an oblique shock and check the angle against theory
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0)
    got = sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])
    print(got["beta_deg"], "vs true", f["beta_deg"])

    # predict separation length from sparse wall-pressure sensors
    data = sl.synthetic.make_sbli_dataset()
    model = sl.SparseSensorModel("L_sep").fit(data[:-2])
    print(model.predict(data[-2:]))
"""

from __future__ import annotations

from . import backend, config, detect, forecast, io, metrics, separation, synthetic, track
from .models import SparseSensorModel, build_xy

__version__ = "0.1.0"

__all__ = ["backend", "config", "detect", "forecast", "io", "metrics",
           "separation", "synthetic", "track", "SparseSensorModel", "compression_ramp_field", "build_xy",
           "__version__"]
