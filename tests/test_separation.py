import numpy as np
import shocklens as sl


def test_separation_points_recovered():
    x = np.linspace(0, 1, 400)
    cf = sl.synthetic.ramp_cf_profile(x, x_sep=0.4, x_reatt=0.6)
    sep = sl.separation.separation_points(x, cf)
    assert sep["separated"]
    assert abs(sep["x_sep"] - 0.4) < 0.01
    assert abs(sep["x_reatt"] - 0.6) < 0.01
    assert abs(sep["L_sep"] - 0.2) < 0.02


def test_attached_flow_reports_no_separation():
    x = np.linspace(0, 1, 200)
    cf = np.full_like(x, 2e-3)
    assert not sl.separation.separation_points(x, cf)["separated"]


def test_dominant_frequency_recovered():
    fs = 100.0
    t = np.arange(0, 60, 1 / fs)
    p = sl.synthetic.wall_pressure_signal(t, f_breathing=0.5, noise=0.02)
    f0 = sl.separation.dominant_frequency(p, fs)
    assert abs(f0 - 0.5) < 0.1
