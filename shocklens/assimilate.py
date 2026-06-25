"""Assimilate velocity and pressure from a density-gradient (BOS/schlieren) field.

This is the physics-first version of the BOS-PINN idea. Experimental SBLI work
can measure the density gradient cheaply (BOS), but velocity is hard near a shock
because PIV seeding and laser sheets are disrupted there. The literature recovers
the velocity and pressure by training a RANS/Euler-constrained PINN on the BOS
density gradient. The weak point of that approach is the shock itself, where the
network oscillates.

ShockLens inverts the priority. The shock is located by physics (tracking the
gradient ridge) and the jump across it is set by the exact Rankine-Hugoniot
relations, not by the network. The smooth regions on each side are then filled
from the upstream state and the exact downstream state. That baseline is exact
for a piecewise-uniform oblique shock and is the leading-order field a PINN would
refine; the optional `shocklens.pinn` module is that refinement, with the tracked
shock and the R-H jump entering as hard constraints.

References: assimilation of SBLI velocity/pressure from BOS via RANS-constrained
PINNs (validated on Mach 2.28 DNS and Mach 2.0 wind-tunnel data); Jagtap et al.,
J. Comput. Phys. 466 (2022) for the XPINN-from-Schlieren inverse problem; Liu et
al., J. Sci. Comput. 2023 for R-H as a hard PINN constraint.
"""

from __future__ import annotations

import numpy as np

from . import detect, rankinehugoniot

__all__ = ["ShockAssimilator"]


class ShockAssimilator:
    """Recover (rho, u, v, p) from a density-gradient field plus the upstream state."""

    def __init__(self, gamma=1.4, detector="oblique_ransac"):
        self.gamma = gamma
        self.detector = detector

    def assimilate(self, field, upstream):
        """field: {rho, x, y, dx, dy}; upstream: {rho1, u1, p1}.

        Returns recovered fields and the tracked shock. The density field stands
        in for the BOS-derived gradient; track on |grad rho| either way.
        """
        x, y = np.asarray(field["x"]), np.asarray(field["y"])
        got = detect.detect(field, method=self.detector)

        rho1, u1, p1 = upstream["rho1"], upstream["u1"], upstream["p1"]
        ds = rankinehugoniot.downstream_state(rho1, u1, p1,
                                              beta_deg=got["beta_deg"],
                                              gamma=self.gamma)

        gy, gx = np.meshgrid(y, x, indexing="ij")
        signed = gy - (got["slope"] * gx + got["intercept"])
        # the downstream (post-shock) side is the higher-density side
        side = signed > 0
        rho = np.asarray(field["rho"])
        if rho[side].mean() < rho[~side].mean():
            side = ~side  # ensure `side` marks the post-shock region

        rho_r = np.where(side, ds["rho2"], rho1).astype(np.float32)
        u_r = np.where(side, ds["u2"], u1).astype(np.float32)
        v_r = np.where(side, ds["v2"], 0.0).astype(np.float32)
        p_r = np.where(side, ds["p2"], p1).astype(np.float32)

        return {"rho": rho_r, "u": u_r, "v": v_r, "p": p_r,
                "downstream": ds, "beta_deg": got["beta_deg"],
                "post_shock_mask": side, "x": x, "y": y}

    @staticmethod
    def rms_error(recovered, truth):
        """Relative RMS error per field between recovered and truth dicts."""
        out = {}
        for k in ("u", "v", "p", "rho"):
            if k in truth:
                a, b = np.asarray(recovered[k]), np.asarray(truth[k])
                denom = np.sqrt(np.mean(b**2)) or 1.0
                out[k] = float(np.sqrt(np.mean((a - b)**2)) / denom)
        return out
