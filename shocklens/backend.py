"""Array backend indirection.

Detectors and field ops get their array module from the data they are handed, so
the same code runs on numpy today and on a GPU array library (cupy) tomorrow with
no edits: pass cupy arrays in, get cupy work out. Only the tiny final solves drop
to numpy. The GPU path is structurally supported here; validate it on hardware
before relying on it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["array_namespace", "to_numpy"]


def array_namespace(*arrays):
    """Return the array module (numpy, or cupy if the inputs are cupy arrays)."""
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy as cp
            return cp
    return np


def to_numpy(a):
    """Bring an array back to numpy regardless of where it lives."""
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except ImportError:
        pass
    return np.asarray(a)
