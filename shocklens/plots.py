"""Figures for SBLI results.

matplotlib with the Agg backend, so it runs headless (CI, HPC). Every function
saves a PNG and returns its path. These are the visuals that make extracted
quantities legible: a schlieren image, a Cf curve with separation marked, a
wall-pressure spectrum, and a shock trajectory.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import detect, separation  # noqa: E402

__all__ = ["plot_schlieren", "plot_cf", "plot_wall_pressure",
           "plot_shock_trajectory", "plot_sensor_importance"]


def plot_schlieren(field, path, title="Numerical schlieren"):
    s = detect.schlieren(field["rho"], field["dx"], field["dy"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(s, origin="lower", cmap="gray_r",
              extent=[field["x"][0], field["x"][-1], field["y"][0], field["y"][-1]],
              aspect="auto")
    ax.set(xlabel="x", ylabel="y", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


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
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_wall_pressure(t, p, fs, path, title="Wall pressure and spectrum"):
    f, pxx = separation.psd(p, fs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    a1.plot(t, p, color="k", lw=0.8); a1.set(xlabel="t", ylabel="p_wall")
    a2.semilogy(f, pxx, color="tab:blue"); a2.set(xlabel="frequency", ylabel="PSD")
    a2.axvline(separation.dominant_frequency(p, fs), color="tab:red", ls="--", lw=1)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_shock_trajectory(track, path, title="Shock-foot trajectory"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(track["t"], track["x_foot"], color="tab:purple")
    ax.set(xlabel="t", ylabel="x_foot",
           title=f"{title}  (f_breathing = {track['f_breathing']:.2f})")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_sensor_importance(importance, sensors_x, path,
                           title="Sensor importance"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(importance)), importance, color="tab:orange")
    ax.set_xticks(range(len(sensors_x)))
    ax.set_xticklabels([f"{x:.2f}" for x in sensors_x], rotation=45, fontsize=7)
    ax.set(xlabel="sensor x", ylabel="importance", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path
