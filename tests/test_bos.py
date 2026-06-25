import numpy as np

import shocklens as sl
from shocklens import bos


def test_bos_roundtrip_recovers_density_gradient():
    f = sl.synthetic.oblique_shock_field(mach=3.0, theta_deg=15.0)
    dx_disp, dy_disp = bos.synthetic_bos(f["rho"], f["dx"], f["dy"])
    gx, gy = bos.density_gradient_from_bos(dx_disp, dy_disp)
    gy_true, gx_true = np.gradient(f["rho"].astype(float), f["dy"], f["dx"])
    assert np.allclose(gx, gx_true, atol=1e-4)
    assert np.allclose(gy, gy_true, atol=1e-4)


def test_schlieren_from_bos_peaks_on_shock():
    f = sl.synthetic.oblique_shock_field()
    dx_disp, dy_disp = bos.synthetic_bos(f["rho"], f["dx"], f["dy"])
    s = bos.schlieren_from_bos(dx_disp, dy_disp)
    assert s.max() > 5 * s.mean()        # gradient concentrated at the shock
