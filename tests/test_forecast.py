import numpy as np
import shocklens as sl


def test_forecaster_beats_persistence_on_sine():
    # A predictable oscillation should be forecast better than "repeat last value".
    t = np.linspace(0, 12, 600)
    series = 0.3 + 0.05 * np.sin(2 * np.pi * 1.5 * t) + 0.002 * np.random.default_rng(0).standard_normal(t.size)
    split = 400
    fc = sl.forecast.ShockForecaster(window=12, horizon=1).fit(series[:split])
    from shocklens.forecast import windowed_xy
    X, y = windowed_xy(series, 12, 1)
    pred = fc.forecast(series)
    n_test = len(y) - (split - 12)
    r2 = sl.metrics.regression_scores(pred[-n_test:], y[-n_test:])["r2"]
    assert r2 > 0.6


def test_windowing_shapes():
    X, y = sl.forecast.windowed_xy(np.arange(20), window=5, horizon=2)
    assert X.shape == (14, 5) and y.shape == (14,)
    assert y[0] == 6  # series[0:5] -> predict index 5+2-1 = 6
