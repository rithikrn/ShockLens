import shocklens as sl


def test_sparse_sensor_model_learns():
    data = sl.synthetic.make_sbli_dataset(
        ramp_angles=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    test_idx = {3, 6}                       # interior, so RF interpolates
    train = [c for i, c in enumerate(data) if i not in test_idx]
    test = [data[i] for i in test_idx]
    model = sl.SparseSensorModel("L_sep").fit(train)
    pred = model.predict(test)
    r2 = sl.metrics.regression_scores(pred, [c["L_sep"] for c in test])["r2"]
    assert r2 > 0.5


def test_sensor_importance_shape():
    data = sl.synthetic.make_sbli_dataset()
    model = sl.SparseSensorModel("x_shock").fit(data)
    assert model.sensor_importance().shape[0] == data[0]["sensors"].shape[0]
