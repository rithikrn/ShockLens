import shocklens as sl


def test_oblique_beta_matches_known_value():
    # M=2, theta=10deg -> beta ~ 39.3deg (standard gas-dynamics tables).
    beta = sl.synthetic.oblique_beta(2.0, 10.0)
    assert abs(beta - 39.3) < 0.5


def test_detector_recovers_shock_angle():
    for mach, theta in [(2.5, 12.0), (3.0, 15.0), (4.0, 20.0)]:
        f = sl.synthetic.oblique_shock_field(mach=mach, theta_deg=theta)
        got = sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])
        assert abs(got["beta_deg"] - f["beta_deg"]) < 2.0, (mach, theta)


def test_detector_recovers_shock_foot():
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0, x_corner=0.2)
    got = sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])
    assert abs(got["x_foot"] - 0.2) < 0.05


def test_schlieren_peaks_on_the_shock():
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0)
    s = sl.detect.schlieren(f["rho"], f["dx"], f["dy"])
    assert s.max() > 5 * s.mean()
