import shocklens as sl


def test_tracker_recovers_breathing_frequency():
    t, fields, truth = sl.synthetic.oblique_shock_timeseries(
        n_frames=120, fs=200.0, f_breathing=2.0)
    tr = sl.track.track_fields(t, fields)
    assert abs(tr["f_breathing"] - 2.0) < 0.25


def test_tracker_follows_foot_motion():
    t, fields, truth = sl.synthetic.oblique_shock_timeseries(
        n_frames=80, amp=0.04, f_breathing=1.5)
    tr = sl.track.track_fields(t, fields)
    # detected foot range should be close to the imposed +/- amp swing
    swing = tr["x_foot"].max() - tr["x_foot"].min()
    assert swing > 0.04
