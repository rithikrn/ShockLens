import numpy as np

from shocklens import rankinehugoniot as rh


def test_normal_shock_matches_tables():
    # Textbook M1=2, gamma=1.4
    out = rh.normal_shock(2.0)
    assert abs(out["p_ratio"] - 4.5) < 1e-6
    assert abs(out["rho_ratio"] - 2.6667) < 1e-3
    assert abs(out["T_ratio"] - 1.6875) < 1e-3
    assert abs(out["M2"] - 0.57735) < 1e-4
    assert abs(out["p0_ratio"] - 0.72087) < 1e-4


def test_oblique_consistent_with_theta_beta_M():
    # beta from theta, then theta back from beta, must round-trip
    M1, theta = 3.0, 20.0
    o = rh.oblique_shock(M1, theta_deg=theta)
    assert abs(o["theta_deg"] - theta) < 1e-3
    back = rh.oblique_shock(M1, beta_deg=o["beta_deg"])
    assert abs(back["theta_deg"] - theta) < 1e-2
    assert 1.0 < o["M2"] < M1            # weak oblique stays supersonic here


def test_downstream_state_self_consistent():
    rho1, p1, gamma = 1.0, 1.0, 1.4
    M1 = 3.0
    u1 = M1 * np.sqrt(gamma * p1 / rho1)
    ds = rh.downstream_state(rho1, u1, p1, theta_deg=20.0)
    # density and pressure rise, flow deflects by theta
    assert ds["rho2"] > rho1 and ds["p2"] > p1
    assert abs(np.rad2deg(np.arctan2(ds["v2"], ds["u2"])) - ds["theta_deg"]) < 1e-3


def test_burgers_shock_speed_is_average():
    s = rh.shock_speed(2.0, 0.0, flux=lambda u: 0.5 * u**2)
    assert abs(s - 1.0) < 1e-12       # (uL+uR)/2 = 1


def test_rh_residual_zero_for_true_jump():
    # A normal shock state pair should give ~zero R-H flux mismatch along x.
    rho1, p1, gamma = 1.0, 1.0, 1.4
    M1 = 2.0
    a1 = np.sqrt(gamma * p1 / rho1)
    u1 = M1 * a1
    ns = rh.normal_shock(M1)
    rho2, p2 = rho1 * ns["rho_ratio"], p1 * ns["p_ratio"]
    u2 = rho1 * u1 / rho2            # mass conservation
    res = rh.rh_residual((rho1, u1, 0.0, p1), (rho2, u2, 0.0, p2), (1.0, 0.0))
    assert np.max(np.abs(res)) < 1e-6
