import os
import tempfile

import shocklens as sl


def test_compression_ramp_field_has_shock_and_separation():
    f = sl.synthetic.compression_ramp_field(mach=3.0, theta_deg=20.0)
    # shock angle recovered from the field
    got = sl.detect.detect(f, method="oblique_line")
    assert abs(got["beta_deg"] - f["beta_deg"]) < 2.0
    # the case genuinely separates, and the Cf agrees with ground truth
    sep = sl.separation.separation_points(f["x"], f["cf"])
    assert sep["separated"]
    assert abs(sep["x_sep"] - f["x_sep"]) < 0.02
    assert abs(sep["x_reatt"] - f["x_reatt"]) < 0.02


def test_overlay_figure_written():
    from shocklens import plots
    f = sl.synthetic.compression_ramp_field()
    got = sl.detect.detect(f)
    p = plots.plot_detection_overlay(f, got,
                                     os.path.join(tempfile.mkdtemp(), "ov.png"))
    assert os.path.getsize(p) > 0


def test_both_detectors_agree_on_ramp():
    f = sl.synthetic.compression_ramp_field()
    a = sl.detect.detect(f, method="oblique_line")["beta_deg"]
    b = sl.detect.detect(f, method="oblique_ransac")["beta_deg"]
    assert abs(a - b) < 1.0
