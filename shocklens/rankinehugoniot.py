"""Exact Rankine-Hugoniot jump relations across a shock.

These are closed-form gas-dynamics, not learned. Given an upstream state and a
shock angle, the downstream density, pressure, temperature, velocity, and Mach
number follow exactly from conservation of mass, momentum, and energy. This is
the physics that the assimilator enforces *at* the shock, so the network never
has to represent the discontinuity it is worst at. Validated against the standard
normal-shock tables (e.g. M1=2, gamma=1.4 gives p2/p1=4.5, rho2/rho1=2.667,
T2/T1=1.6875, M2=0.5774).

References for using R-H as a hard constraint in shock-aware PINNs: Liu et al.,
"Discontinuity Computing Using Physics-Informed Neural Networks", J. Sci. Comput.
2023; the review arXiv:2503.17379 (2025) states the R-H conditions as the
inviscid conservation laws across the discontinuity.
"""

from __future__ import annotations

import numpy as np

from .synthetic import oblique_beta

GAMMA = 1.4

__all__ = ["normal_shock", "oblique_shock", "downstream_state",
           "shock_speed", "rh_residual", "oblique_beta"]


def normal_shock(M1, gamma=GAMMA):
    """Normal-shock jump ratios for upstream normal Mach M1 (M1 >= 1).

    Returns M2 and the static ratios p2/p1, rho2/rho1, T2/T1, plus the
    stagnation-pressure ratio p02/p01.
    """
    M1 = float(M1)
    if M1 < 1.0:
        raise ValueError("normal-shock upstream Mach must be >= 1")
    g = gamma
    M2 = np.sqrt((1 + (g - 1) / 2 * M1**2) / (g * M1**2 - (g - 1) / 2))
    p_ratio = 1 + 2 * g / (g + 1) * (M1**2 - 1)
    rho_ratio = (g + 1) * M1**2 / ((g - 1) * M1**2 + 2)
    T_ratio = p_ratio / rho_ratio
    p0_ratio = (rho_ratio**(g / (g - 1))
                * ((g + 1) / (2 * g * M1**2 - (g - 1)))**(1 / (g - 1)))
    return {"M2": float(M2), "p_ratio": float(p_ratio),
            "rho_ratio": float(rho_ratio), "T_ratio": float(T_ratio),
            "p0_ratio": float(p0_ratio)}


def oblique_shock(M1, theta_deg=None, beta_deg=None, gamma=GAMMA):
    """Oblique-shock state for deflection theta or shock angle beta.

    Give exactly one of theta_deg (flow turn) or beta_deg (shock angle). Returns
    beta_deg, theta_deg, the downstream Mach M2, and the static jump ratios.
    """
    if (theta_deg is None) == (beta_deg is None):
        raise ValueError("give exactly one of theta_deg or beta_deg")
    if beta_deg is None:
        beta_deg = oblique_beta(M1, theta_deg, gamma)
    beta = np.deg2rad(beta_deg)
    Mn1 = M1 * np.sin(beta)
    ns = normal_shock(Mn1, gamma)
    Mn2 = ns["M2"]
    if theta_deg is None:
        # recover the turn angle from beta via the theta-beta-M relation
        g = gamma
        tan_theta = (2 / np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1)
                     / (M1**2 * (g + np.cos(2 * beta)) + 2))
        theta_deg = float(np.rad2deg(np.arctan(tan_theta)))
    M2 = Mn2 / np.sin(beta - np.deg2rad(theta_deg))
    return {"beta_deg": float(beta_deg), "theta_deg": float(theta_deg),
            "M2": float(M2), "p_ratio": ns["p_ratio"],
            "rho_ratio": ns["rho_ratio"], "T_ratio": ns["T_ratio"]}


def downstream_state(rho1, u1, p1, theta_deg=None, beta_deg=None, gamma=GAMMA,
                     a1=None):
    """Full downstream primitive state across an oblique shock.

    Upstream is (rho1, u1, p1) with u1 the freestream speed along +x. Returns the
    downstream density, pressure, temperature ratio, speed magnitude, the (u, v)
    components after a deflection theta, the Mach numbers, and the shock angle.
    """
    a1 = np.sqrt(gamma * p1 / rho1) if a1 is None else a1
    M1 = u1 / a1
    sh = oblique_shock(M1, theta_deg=theta_deg, beta_deg=beta_deg, gamma=gamma)
    rho2 = rho1 * sh["rho_ratio"]
    p2 = p1 * sh["p_ratio"]
    a2 = np.sqrt(gamma * p2 / rho2)
    speed2 = sh["M2"] * a2
    th = np.deg2rad(sh["theta_deg"])
    return {"rho2": float(rho2), "p2": float(p2), "T_ratio": sh["T_ratio"],
            "speed2": float(speed2), "u2": float(speed2 * np.cos(th)),
            "v2": float(speed2 * np.sin(th)), "M1": float(M1), "M2": sh["M2"],
            "beta_deg": sh["beta_deg"], "theta_deg": sh["theta_deg"]}


def shock_speed(u_left, u_right, flux):
    """Shock speed from neighbouring states for a scalar conservation law.

    The Rankine-Hugoniot speed s = [f] / [u] = (f(uR) - f(uL)) / (uR - uL).
    For Burgers (flux f(u) = u**2/2) this is the average (uL + uR)/2. This is the
    "compute the shock speed from neighbouring states" relation, the honest core
    of the dynamic shock-tracking idea.
    """
    du = u_right - u_left
    if du == 0:
        return 0.0
    return float((flux(u_right) - flux(u_left)) / du)


def rh_residual(state_left, state_right, normal, gamma=GAMMA):
    """Mass/momentum/energy flux mismatch across a face (zero when R-H holds).

    Each state is (rho, u, v, p); normal is the unit shock normal (nx, ny). Used
    as the hard interface constraint a shock-aware PINN drives to zero, and as a
    standalone check on any candidate jump.
    """
    nx, ny = normal
    out = []
    for (rho, u, v, p) in (state_left, state_right):
        un = u * nx + v * ny
        e = p / ((gamma - 1) * rho) + 0.5 * (u**2 + v**2)
        out.append(np.array([rho * un,
                             rho * un * u + p * nx,
                             rho * un * v + p * ny,
                             un * (rho * e + p)]))
    return out[1] - out[0]
