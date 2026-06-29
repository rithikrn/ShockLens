"""Figures for SBLI results.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import detect, separation  # noqa: E402

__all__ = ["plot_schlieren", "plot_cf", "plot_wall_pressure",
           "plot_shock_trajectory", "plot_sensor_importance",
           "plot_parity", "plot_forecast_compare", "plot_detection_overlay",
           "plot_assimilation", "plot_field", "plot_wall_distribution"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_schlieren(field, path, title="Numerical schlieren"):
    s = detect.schlieren(field["rho"], field["dx"], field["dy"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(s, origin="lower", cmap="gray_r",
              extent=[field["x"][0], field["x"][-1], field["y"][0], field["y"][-1]],
              aspect="auto")
    ax.set(xlabel="x", ylabel="y", title=title)
    return _save(fig, path)


def plot_cf(x, cf, path, title="Skin friction and separation"):
    sep = separation.separation_points(x, cf)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(x, cf, color="k")
    ax.axhline(0, color="0.6", lw=0.8)
    if sep["separated"]:
        ax.axvspan(sep["x_sep"], sep["x_reatt"], color="tab:red", alpha=0.15)
        ax.axvline(sep["x_sep"], color="tab:red", ls="--", lw=1)
        ax.axvline(sep["x_reatt"], color="tab:green", ls="--", lw=1)
        ax.set_title(f"{title}  (L_sep = {sep['L_sep']:.3f})")
    ax.set(xlabel="x", ylabel="Cf")
    return _save(fig, path)


def plot_wall_pressure(t, p, fs, path, title="Wall pressure and spectrum"):
    f, pxx = separation.psd(p, fs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    a1.plot(t, p, color="k", lw=0.8)
    a1.set(xlabel="t", ylabel="p_wall")
    a2.semilogy(f, pxx, color="tab:blue")
    a2.set(xlabel="frequency", ylabel="PSD")
    a2.axvline(separation.dominant_frequency(p, fs), color="tab:red", ls="--", lw=1)
    fig.suptitle(title)
    return _save(fig, path)


def plot_shock_trajectory(track, path, title="Shock-foot trajectory"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(track["t"], track["x_foot"], color="tab:purple")
    ax.set(xlabel="t", ylabel="x_foot",
           title=f"{title}  (f_breathing = {track['f_breathing']:.2f})")
    return _save(fig, path)


def plot_sensor_importance(importance, sensors_x, path, title="Sensor importance"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(importance)), importance, color="tab:orange")
    ax.set_xticks(range(len(sensors_x)))
    ax.set_xticklabels([f"{x:.2f}" for x in sensors_x], rotation=45, fontsize=7)
    ax.set(xlabel="sensor x", ylabel="importance", title=title)
    return _save(fig, path)


def plot_parity(y_true, y_pred, y_std, path, title="Predicted vs true"):
    """Parity plot: predicted against true with +/-1 sigma error bars."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", ls="--",
            lw=1, label="perfect (y = x)")
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", color="tab:blue",
                ecolor="tab:gray", capsize=3, label="prediction +/- 1 sigma")
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ax.set(xlabel="true", ylabel="predicted", title=f"{title}  (MAE = {mae:.4f})")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, path)


def plot_forecast_compare(t, y_true, y_pred, path, title="Forecast vs truth"):
    """Two panels: true and predicted overlaid, plus the residual underneath."""
    t, y_true, y_pred = np.asarray(t), np.asarray(y_true), np.asarray(y_pred)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(t, y_true, color="k", lw=1.3, label="true")
    a1.plot(t, y_pred, color="tab:red", ls="--", lw=1.3, label="predicted")
    a1.set(ylabel="shock foot x", title=title)
    a1.legend(fontsize=8)
    a2.plot(t, y_pred - y_true, color="tab:purple", lw=1)
    a2.axhline(0, color="0.6", lw=0.8)
    a2.set(xlabel="t", ylabel="error")
    return _save(fig, path)


def plot_detection_overlay(field, detected, path,
                           title="Shock detection: true vs detected"):
    """Schlieren background with the true and detected shock lines overlaid.

    The field-level check: if the dashed detected line sits on the bright ridge
    and on the solid true line, the extraction is right. The angle error is in
    the title. This is the figure that proves the physics, not a metric table.
    """
    s = detect.schlieren(field["rho"], field["dx"], field["dy"])
    x = np.asarray(field["x"])
    y0, y1 = float(field["y"][0]), float(field["y"][-1])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.imshow(s, origin="lower", cmap="gray_r",
              extent=[x[0], x[-1], y0, y1], aspect="auto")

    yd = detected["slope"] * x + detected["intercept"]
    ax.plot(x, yd, color="tab:red", ls="--", lw=1.8,
            label=f"detected ({detected['beta_deg']:.1f} deg)")

    extra = ""
    if "beta_deg" in field:
        x0 = field.get("x_sep", field.get("x_corner", 0.0))
        m = np.tan(np.deg2rad(field["beta_deg"]))
        ax.plot(x, m * (x - x0), color="tab:green", lw=1.3,
                label=f"true ({field['beta_deg']:.1f} deg)")
        extra = f"  (error {abs(detected['beta_deg'] - field['beta_deg']):.2f} deg)"

    ax.set(xlabel="x", ylabel="y", title=title + extra)
    ax.set_ylim(y0, y1)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, path)


def plot_assimilation(recovered, path, field="u",
                      title="Assimilated field (recovered from density gradient)"):
    """Show a recovered field (u, v, p, or rho) with the tracked shock implied.

    The headline output: a velocity or pressure field reconstructed from only the
    density gradient, the quantity BOS gives and PIV struggles to get near a shock.
    """
    x, y = np.asarray(recovered["x"]), np.asarray(recovered["y"])
    arr = np.asarray(recovered[field])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(arr, origin="lower", cmap="viridis",
                   extent=[x[0], x[-1], y[0], y[-1]], aspect="auto")
    fig.colorbar(im, ax=ax, label=field)
    ax.set(xlabel="x", ylabel="y",
           title=f"{title}  (beta={recovered['beta_deg']:.1f} deg)")
    return _save(fig, path)


def plot_field(field, key, path, title=None, shock=None):
    """Filled contour of a field (rho, p, ...), optionally with the shock line."""
    x, y = np.asarray(field["x"]), np.asarray(field["y"])
    arr = np.asarray(field[key])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    c = ax.contourf(x, y, arr, levels=30, cmap="viridis")
    fig.colorbar(c, ax=ax, label=key)
    if shock is not None:
        ax.plot(x, shock["slope"] * x + shock["intercept"], "r--", lw=1.3,
                label=f"shock {shock['beta_deg']:.1f} deg")
        ax.legend(loc="upper left", fontsize=8)
    ax.set(xlabel="x", ylabel="y", title=title or f"{key} contours")
    ax.set_ylim(float(y[0]), float(y[-1]))
    return _save(fig, path)


def plot_wall_distribution(x, p, path, x_sep=None, x_reatt=None,
                           title="Wall pressure distribution"):
    """Wall pressure versus streamwise distance, with the interaction marked."""
    x, p = np.asarray(x), np.asarray(p)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(x, p, "-", lw=1.6, color="#1f77b4")
    if x_sep is not None:
        ax.axvline(x_sep, ls="--", color="#d62728", lw=1, label="separation")
    if x_reatt is not None:
        ax.axvline(x_reatt, ls="--", color="#2ca02c", lw=1, label="reattachment")
    if x_sep is not None or x_reatt is not None:
        ax.legend(loc="upper left", fontsize=8)
    ax.set(xlabel="x", ylabel="wall pressure", title=title)
    ax.grid(alpha=0.3)
    return _save(fig, path)
