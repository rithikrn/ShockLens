"""ShockLens CLI: shocklens <command>.

demo     full offline pipeline on synthetic data (no solver, no VTK)
extract  detect a shock in a VTK file and report angle + foot
train    fit the sparse-sensor model on synthetic ramps and score it
info     environment info
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_demo(args):
    import shocklens as sl

    f = sl.synthetic.oblique_shock_field(mach=args.mach, theta_deg=args.theta)
    got = sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])

    data = sl.synthetic.make_sbli_dataset()
    test = [data[3], data[6]]
    train = [c for i, c in enumerate(data) if i not in (3, 6)]
    model = sl.SparseSensorModel("L_sep").fit(train)
    pred = model.predict(test)
    reg = sl.metrics.regression_scores(pred, [c["L_sep"] for c in test])

    case = sl.synthetic.make_sbli_dataset()[4]
    sep = sl.separation.separation_points(case["x"], case["cf"])

    print(json.dumps({
        "shock_angle_detected": round(got["beta_deg"], 2),
        "shock_angle_true": round(f["beta_deg"], 2),
        "shock_foot_detected": round(got["x_foot"], 3),
        "separation": {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in sep.items()},
        "L_sep_prediction_r2": round(reg["r2"], 3),
        "L_sep_prediction_mae": round(reg["mae"], 4),
    }, indent=2))


def _cmd_extract(args):
    import shocklens as sl

    raw = sl.io.read_vtk_slice(args.vtk, fields=("rho",))
    rho, x, y = sl.io.to_uniform_grid(raw["points"], raw["rho"], args.nx, args.ny)
    got = sl.detect.detect_oblique_shock(rho, x, y)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in got.items()}, indent=2))


def _cmd_train(args):
    import shocklens as sl

    data = sl.synthetic.make_sbli_dataset()
    test = [data[3], data[6]]
    train = [c for i, c in enumerate(data) if i not in (3, 6)]
    model = sl.SparseSensorModel(args.target).fit(train)
    pred = model.predict(test)
    reg = sl.metrics.regression_scores(pred, [c[args.target] for c in test])
    print(json.dumps({"target": args.target, **{k: round(v, 4)
                                                 for k, v in reg.items()}}, indent=2))


def _cmd_track(args):
    import shocklens as sl
    t, fields, truth = sl.synthetic.oblique_shock_timeseries(
        n_frames=args.frames, f_breathing=args.freq)
    tr = sl.track.track_fields(t, fields)
    print(json.dumps({
        "frames": len(t),
        "breathing_freq_detected": round(tr["f_breathing"], 3),
        "breathing_freq_true": truth["f_breathing"],
        "x_foot_range": [round(float(tr["x_foot"].min()), 3),
                         round(float(tr["x_foot"].max()), 3)],
    }, indent=2))


def _cmd_plot(args):
    import shocklens as sl
    from shocklens import plots
    out = args.outdir
    import os
    os.makedirs(out, exist_ok=True)

    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0)
    case = sl.synthetic.make_sbli_dataset()[4]
    import numpy as np
    t = np.arange(0, 30, 1 / 200.0)
    p = sl.synthetic.wall_pressure_signal(t, f_breathing=0.6)
    tt, fields, _ = sl.synthetic.oblique_shock_timeseries(n_frames=200)
    tr = sl.track.track_fields(tt, fields)

    paths = [
        plots.plot_schlieren(f, f"{out}/schlieren.png"),
        plots.plot_cf(case["x"], case["cf"], f"{out}/skin_friction.png"),
        plots.plot_wall_pressure(t, p, 200.0, f"{out}/wall_pressure.png"),
        plots.plot_shock_trajectory(tr, f"{out}/shock_trajectory.png"),
    ]
    print("wrote:\n  " + "\n  ".join(paths))


def _cmd_forecast(args):
    import shocklens as sl
    # Track an oscillating shock, then forecast its motion one step ahead.
    t, fields, truth = sl.synthetic.oblique_shock_timeseries(
        n_frames=300, fs=100.0, f_breathing=2.0)
    tr = sl.track.track_fields(t, fields)
    series = tr["x_foot"]
    split = len(series) * 2 // 3
    fc = sl.forecast.ShockForecaster(window=args.window).fit(series[:split])
    pred = fc.forecast(series)
    # score the held-out tail
    from shocklens.forecast import windowed_xy
    _, y = windowed_xy(series, args.window)
    n_test = len(y) - (split - args.window)
    r2 = sl.metrics.regression_scores(pred[-n_test:], y[-n_test:])["r2"]
    print(json.dumps({"window": args.window,
                      "forecast_r2_heldout": round(r2, 3),
                      "next_step_prediction": round(fc.predict_next(series[:split]), 4)},
                     indent=2))


def _cmd_info(args):
    import shocklens as sl
    print(f"shocklens {sl.__version__}")
    try:
        import pyvista as pv
        print(f"pyvista   {pv.__version__}")
    except ImportError:
        print("pyvista   not installed (VTK reading disabled)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="shocklens", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="offline synthetic pipeline")
    d.add_argument("--mach", type=float, default=3.0)
    d.add_argument("--theta", type=float, default=15.0)
    d.set_defaults(func=_cmd_demo)

    e = sub.add_parser("extract", help="detect a shock in a VTK file")
    e.add_argument("vtk")
    e.add_argument("--nx", type=int, default=200)
    e.add_argument("--ny", type=int, default=120)
    e.set_defaults(func=_cmd_extract)

    t = sub.add_parser("train", help="fit + score the sparse-sensor model")
    t.add_argument("--target", default="L_sep", choices=["L_sep", "x_shock", "x_sep"])
    t.set_defaults(func=_cmd_train)

    k = sub.add_parser("track", help="track an oscillating shock over time")
    k.add_argument("--frames", type=int, default=200)
    k.add_argument("--freq", type=float, default=2.0)
    k.set_defaults(func=_cmd_track)

    pl = sub.add_parser("plot", help="render the standard figures to a folder")
    pl.add_argument("--outdir", default="figures")
    pl.set_defaults(func=_cmd_plot)

    fc = sub.add_parser("forecast", help="forecast shock motion from its history")
    fc.add_argument("--window", type=int, default=8)
    fc.set_defaults(func=_cmd_forecast)

    i = sub.add_parser("info", help="environment info")
    i.set_defaults(func=_cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
