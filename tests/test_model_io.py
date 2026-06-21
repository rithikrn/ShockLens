import tempfile, os
import shocklens as sl


def test_model_save_load_roundtrip():
    data = sl.synthetic.make_sbli_dataset()
    m = sl.SparseSensorModel("L_sep").fit(data)
    p0 = m.predict(data[:2])
    path = os.path.join(tempfile.mkdtemp(), "m.joblib")
    m.save(path)
    m2 = sl.SparseSensorModel.load(path)
    p1 = m2.predict(data[:2])
    assert (abs(p0 - p1) < 1e-9).all()
    assert m2.target == "L_sep"


def test_plots_write_files():
    import numpy as np
    from shocklens import plots
    d = tempfile.mkdtemp()
    f = sl.synthetic.oblique_shock_field()
    out = plots.plot_schlieren(f, os.path.join(d, "s.png"))
    assert os.path.getsize(out) > 0
