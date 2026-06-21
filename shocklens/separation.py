"""Separation and wall-signal analysis.

separation_points finds where Cf crosses zero: positive-to-negative is
separation, negative-to-positive is reattachment, and the gap is the separation
length, the clearest single SBLI severity metric. The wall-pressure helpers give
the unsteady side: RMS load and the dominant shock-breathing frequency from a
Welch PSD.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

__all__ = ["separation_points", "pressure_rms", "dominant_frequency", "psd"]


def _zero_crossings(x, f, rising):
    out = []
    for i in range(len(f) - 1):
        a, b = f[i], f[i + 1]
        if (rising and a < 0 <= b) or (not rising and a >= 0 > b):
            # linear interpolation to the crossing
            out.append(x[i] + (x[i + 1] - x[i]) * (0 - a) / (b - a))
    return out


def separation_points(x, cf):
    """Locate separation, reattachment, and bubble length from Cf(x).

    Returns dict with x_sep, x_reatt, L_sep (NaN if the flow stays attached).
    """
    sep = _zero_crossings(x, cf, rising=False)   # Cf: + -> -
    reatt = _zero_crossings(x, cf, rising=True)   # Cf: - -> +
    if not sep or not reatt:
        return {"x_sep": float("nan"), "x_reatt": float("nan"),
                "L_sep": 0.0, "separated": False}
    x_sep = sep[0]
    x_reatt = next((r for r in reatt if r > x_sep), reatt[-1])
    return {"x_sep": float(x_sep), "x_reatt": float(x_reatt),
            "L_sep": float(x_reatt - x_sep), "separated": True}


def pressure_rms(p_signal):
    """RMS of the fluctuating wall pressure."""
    return float(np.std(p_signal))


def psd(p_signal, fs):
    """Welch power spectral density, zero-padded for finer peak localisation.

    Returns (frequencies, power).
    """
    nperseg = min(1024, len(p_signal))
    nfft = 4 * nperseg
    return welch(p_signal, fs=fs, nperseg=nperseg, nfft=nfft)


def dominant_frequency(p_signal, fs):
    """Frequency of the largest PSD peak above DC."""
    f, pxx = psd(p_signal, fs)
    if len(f) < 2:
        return 0.0
    k = 1 + int(np.argmax(pxx[1:]))
    return float(f[k])
