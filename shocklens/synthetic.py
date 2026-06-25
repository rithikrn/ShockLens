"""Synthetic compressible and SBLI fields with known ground truth.

Everything here runs with no solver installed, and every case carries the exact
answer the extractors are supposed to recover: the oblique-shock angle from the
theta-beta-M relation, and the separation/reattachment points of a ramp. That
makes the test suite a physics check, not a curve-fit. Swap in real OpenFOAM or
solver VTK later via shocklens.io.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4

__all__ = ["oblique_beta", "oblique_shock_field", "oblique_shock_timeseries",
           "ramp_cf_profile", "wall_pressure_signal", "make_sbli_dataset"]


def oblique_beta(mach, theta_deg, gamma=GAMMA, weak=True):
    """Shock angle beta (deg) for deflection theta at upstream Mach, weak root."""
    theta = np.deg2rad(theta_deg)

    def f(beta):
        return (2 / np.tan(beta) * (mach**2 * np.sin(beta)**2 - 1)
                / (mach**2 * (gamma + np.cos(2 * beta)) + 2) - np.tan(theta))

    mu = np.arcsin(1.0 / mach)              # Mach angle, lower bound
    lo, hi = (mu + 1e-4, np.deg2rad(65)) if weak else (np.deg2rad(65), np.deg2rad(89.9))
    return np.rad2deg(brentq(f, lo, hi))


def _oblique_jumps(mach, beta_deg, gamma=GAMMA):
    mn1 = mach * np.sin(np.deg2rad(beta_deg))
    rho_ratio = (gamma + 1) * mn1**2 / ((gamma - 1) * mn1**2 + 2)
    p_ratio = 1 + 2 * gamma / (gamma + 1) * (mn1**2 - 1)
    return rho_ratio, p_ratio


def oblique_shock_field(nx=200, ny=120, mach=3.0, theta_deg=15.0,
                        x_corner=0.2, shock_width=0.01, gamma=GAMMA,
                        xlim=(0.0, 1.0), ylim=(0.0, 0.6)):
    """2D density/pressure field for a wedge-induced oblique shock.

    The shock springs from (x_corner, 0) at the analytic angle beta. Returns a
    dict with rho, p, x, y, and the ground-truth beta_deg and foot location.
    """
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    beta = oblique_beta(mach, theta_deg, gamma)
    rho_ratio, p_ratio = _oblique_jumps(mach, beta, gamma)

    # Signed distance to the shock line through the corner; tanh = captured jump.
    nrm = np.array([-np.sin(np.deg2rad(beta)), np.cos(np.deg2rad(beta))])
    sd = (gx - x_corner) * nrm[0] + gy * nrm[1]
    s = 0.5 * (1 - np.tanh(sd / shock_width))      # 1 behind shock, 0 ahead
    downstream = (gx > x_corner)
    s = s * downstream

    rho = 1.0 + (rho_ratio - 1.0) * s
    p = 1.0 + (p_ratio - 1.0) * s
    return {"rho": rho.astype(np.float32), "p": p.astype(np.float32),
            "x": xs, "y": ys, "dx": xs[1] - xs[0], "dy": ys[1] - ys[0],
            "beta_deg": float(beta), "x_corner": float(x_corner),
            "mach": float(mach), "theta_deg": float(theta_deg)}


def ramp_cf_profile(x, x_sep, x_reatt, cf_attached=2e-3, sigma_frac=1.0):
    """Skin friction that crosses zero exactly at x_sep and x_reatt.

    A Gaussian dip on a flat attached level, scaled so Cf = 0 at both points and
    negative between them.
    """
    centre = 0.5 * (x_sep + x_reatt)
    half = 0.5 * (x_reatt - x_sep)
    sigma = sigma_frac * half
    amp = cf_attached * np.exp((half / sigma) ** 2)      # forces Cf(x_sep)=0
    return cf_attached - amp * np.exp(-((x - centre) / sigma) ** 2)


def wall_pressure_signal(t, p_mean=2.0, f_breathing=0.5, amp=0.15, noise=0.03,
                         seed=0):
    """Wall-pressure time series with low-frequency shock breathing plus noise."""
    rng = np.random.default_rng(seed)
    return (p_mean + amp * np.sin(2 * np.pi * f_breathing * t)
            + noise * rng.standard_normal(t.shape))


def oblique_shock_timeseries(n_frames=200, fs=100.0, x0=0.3, amp=0.04,
                             f_breathing=2.0, mach=3.0, theta_deg=15.0,
                             nx=160, ny=100):
    """A sequence of oblique-shock fields whose foot oscillates sinusoidally.

    The corner moves as x0 + amp*sin(2*pi*f*t), so a tracker run over the frames
    should recover both the trajectory and the breathing frequency. Returns
    (times, fields, ground_truth).
    """
    t = np.arange(n_frames) / fs
    x_corner = x0 + amp * np.sin(2 * np.pi * f_breathing * t)
    fields = [oblique_shock_field(nx=nx, ny=ny, mach=mach, theta_deg=theta_deg,
                                  x_corner=float(xc)) for xc in x_corner]
    truth = {"t": t, "x_foot": x_corner, "f_breathing": f_breathing, "fs": fs}
    return t, fields, truth

def compression_ramp_field(nx=240, ny=140, mach=3.0, theta_deg=20.0,
                           x_corner=0.5, x_sep=0.40, x_reatt=0.62,
                           shock_width=0.012, bubble_depth=0.15, gamma=GAMMA,
                           xlim=(0.0, 1.0), ylim=(0.0, 0.5)):
    """A 2D compression-ramp SBLI field: separation shock plus a wall bubble.

    Unlike the inviscid wedge, this carries the full interaction: a separation
    shock that springs ahead of the corner (from x_sep) at the theta-beta-M
    angle, and a shallow near-wall recirculation bubble between x_sep and
    x_reatt. Ground truth includes beta_deg, x_sep, x_reatt and a matching Cf,
    so the flagship example exercises shock detection AND separation at once.
    """
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    beta = oblique_beta(mach, theta_deg, gamma)
    rho_ratio, p_ratio = _oblique_jumps(mach, beta, gamma)

    nrm = np.array([-np.sin(np.deg2rad(beta)), np.cos(np.deg2rad(beta))])
    sd = (gx - x_sep) * nrm[0] + gy * nrm[1]
    s = 0.5 * (1 - np.tanh(sd / shock_width)) * (gx > x_sep)
    rho = 1.0 + (rho_ratio - 1.0) * s
    p = 1.0 + (p_ratio - 1.0) * s

    centre, half = 0.5 * (x_sep + x_reatt), 0.5 * (x_reatt - x_sep)
    bubble = np.exp(-((gx - centre) / half) ** 2) * np.exp(-(gy / 0.03) ** 2)
    rho = rho - bubble_depth * bubble

    cf = ramp_cf_profile(xs, x_sep, x_reatt)
    return {"rho": rho.astype(np.float32), "p": p.astype(np.float32),
            "x": xs, "y": ys, "dx": xs[1] - xs[0], "dy": ys[1] - ys[0],
            "cf": cf, "beta_deg": float(beta), "x_corner": float(x_corner),
            "x_sep": float(x_sep), "x_reatt": float(x_reatt),
            "mach": float(mach), "theta_deg": float(theta_deg)}


def make_sbli_dataset(ramp_angles=None, n_sensors=12, seed=0):
    """Ensemble of compression-ramp cases for the sparse-sensor ML baseline.

    Stronger ramp angle -> longer separation and a more upstream shock foot.
    Each case gives wall-pressure sensors (the input) and the SBLI targets.
    """
    if ramp_angles is None:
        ramp_angles = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 200)
    sensors_x = np.linspace(0.1, 0.9, n_sensors)
    rows = []
    for ang in ramp_angles:
        # Empirical-ish scalings (monotonic in ramp angle), plus small noise.
        L_sep = 0.02 * (ang - 6) + rng.normal(0, 0.003)
        x_sep = 0.55 - 0.012 * (ang - 6)
        x_reatt = x_sep + L_sep
        x_shock = x_sep - 0.05
        cf = ramp_cf_profile(x, x_sep, x_reatt)
        # Wall pressure: upstream plateau, rise through the interaction.
        p_w = 1.0 + (1.0 + 0.06 * (ang - 6)) * 0.5 * (1 + np.tanh((x - x_sep) / 0.05))
        p_w = p_w + rng.normal(0, 0.01, x.shape)
        sensors = np.interp(sensors_x, x, p_w)
        rows.append({"ramp_angle": ang, "x": x, "cf": cf, "p_wall": p_w,
                     "sensors_x": sensors_x, "sensors": sensors.astype(np.float32),
                     "x_sep": x_sep, "x_reatt": x_reatt, "L_sep": L_sep,
                     "x_shock": x_shock})
    return rows
