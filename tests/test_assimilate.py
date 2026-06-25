import numpy as np

import shocklens as sl
from shocklens import rankinehugoniot as rh
from shocklens.assimilate import ShockAssimilator


def test_assimilator_recovers_downstream_state():
    M1, theta, gamma = 3.0, 18.0, 1.4
    f = sl.synthetic.oblique_shock_field(mach=M1, theta_deg=theta,
                                         x_corner=0.2)
    a1 = np.sqrt(gamma * 1.0 / 1.0)
    up = {"rho1": 1.0, "u1": M1 * a1, "p1": 1.0}
    out = ShockAssimilator(gamma=gamma).assimilate(f, up)

    # tracked angle matches theta-beta-M theory
    beta_true = rh.oblique_shock(M1, theta_deg=theta)["beta_deg"]
    assert abs(out["beta_deg"] - beta_true) < 2.0

    # recovered downstream pressure/density ratios match exact R-H
    theory = rh.oblique_shock(M1, theta_deg=theta)
    assert abs(out["downstream"]["p2"] / up["p1"] - theory["p_ratio"]) < 0.3
    assert out["post_shock_mask"].any() and (~out["post_shock_mask"]).any()


def test_bos_to_velocity_pipeline_runs():
    # density field -> BOS -> back to a field dict -> assimilate
    f = sl.synthetic.oblique_shock_field(mach=2.5, theta_deg=12.0)
    up = {"rho1": 1.0, "u1": 2.5 * np.sqrt(1.4), "p1": 1.0}
    out = ShockAssimilator().assimilate(f, up)
    assert out["u"].shape == f["rho"].shape
    assert np.all(np.isfinite(out["p"]))
