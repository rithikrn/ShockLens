"""Background-Oriented Schlieren (BOS) handling.

BOS measures the apparent displacement of a textured background seen through a
flow. That displacement is proportional to the line-of-sight integral of the
transverse refractive-index gradient, and through the Gladstone-Dale relation
(n - 1 = G * rho) the refractive-index gradient is proportional to the density
gradient. So a BOS image gives you grad(rho), and nothing else, which is exactly
the single quantity the SBLI-PINN assimilation literature trains on (e.g. the
Mach 2.28 DNS / Mach 2.0 wind-tunnel study that recovers full velocity and
pressure from BOS density gradients alone).

This module is the bridge: `synthetic_bos` makes a BOS-like displacement field
from a density field (for offline testing), and `density_gradient_from_bos`
inverts it back to grad(rho). Both are plain numpy, so the experimental-data
path needs no solver and no GPU.
"""

from __future__ import annotations

import numpy as np

GLADSTONE_DALE_AIR = 2.3e-4  # m^3/kg, near-STP visible light

__all__ = ["synthetic_bos", "density_gradient_from_bos", "schlieren_from_bos"]


def synthetic_bos(rho, dx, dy, gladstone_dale=GLADSTONE_DALE_AIR,
                  sensitivity=1.0, noise=0.0, seed=0):
    """Forward BOS model: density field -> apparent (disp_x, disp_y).

    Displacement is proportional to grad(n) = gladstone_dale * grad(rho), scaled
    by an optical sensitivity (the path length times geometry factor lumped into
    one number). `noise` adds Gaussian pixel noise to mimic a real measurement.
    """
    gy, gx = np.gradient(np.asarray(rho, dtype=float), dy, dx)
    k = gladstone_dale * sensitivity
    disp_x, disp_y = k * gx, k * gy
    if noise > 0:
        rng = np.random.default_rng(seed)
        amp = noise * np.hypot(disp_x, disp_y).std()
        disp_x = disp_x + rng.normal(0, amp, disp_x.shape)
        disp_y = disp_y + rng.normal(0, amp, disp_y.shape)
    return disp_x.astype(np.float32), disp_y.astype(np.float32)


def density_gradient_from_bos(disp_x, disp_y, gladstone_dale=GLADSTONE_DALE_AIR,
                              sensitivity=1.0):
    """Inverse: BOS displacement -> (drho/dx, drho/dy).

    Divides out the optical constant. The result is the data term the
    assimilator fits in the smooth regions; the shock itself is handled by the
    exact jump conditions, not by differentiating across it.
    """
    k = gladstone_dale * sensitivity
    return np.asarray(disp_x) / k, np.asarray(disp_y) / k


def schlieren_from_bos(disp_x, disp_y):
    """Magnitude |displacement|, a schlieren-like field for shock tracking."""
    return np.hypot(np.asarray(disp_x), np.asarray(disp_y))
