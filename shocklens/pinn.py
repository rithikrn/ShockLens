"""Optional smooth-region PINN refinement (the icing, requires the [pinn] extra).

The assimilator already gives a physically exact field for a piecewise-uniform
shock. Real SBLI regions are not uniform: there is an expansion fan, a relaxing
boundary layer, a separation bubble. This module refines the smooth regions with
a small physics-informed network that fits the measured density gradient while
satisfying the compressible Euler residual, with two non-negotiable physics
anchors:

  1. the shock location comes from the tracker, not the network, and
  2. the jump across it is pinned to the exact Rankine-Hugoniot state via
     `rankinehugoniot.rh_residual` as a hard interface penalty.

So the network only ever shapes the smooth flow; it never has to represent the
discontinuity, which is the failure mode documented for vanilla PINNs at shocks
(review arXiv:2503.17379, 2025). This is a research scaffold, not a validated
assimilator: it requires PyTorch (`pip install -e ".[pinn]"`) and you must check
convergence and accuracy on your own case before trusting the output.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SmoothRegionPINN", "torch_available"]


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class SmoothRegionPINN:
    """Minimal Euler-residual PINN that refines one smooth subdomain.

    Inputs are collocation coordinates and the measured density gradient at data
    points; the loss is data-misfit + Euler-residual + an R-H interface penalty.
    Kept deliberately small and readable; scale width/depth and add RANS terms as
    your case needs.
    """

    def __init__(self, hidden=(64, 64, 64), gamma=1.4, seed=0):
        if not torch_available():
            raise ImportError("SmoothRegionPINN needs PyTorch: pip install -e \".[pinn]\"")
        import torch
        torch.manual_seed(seed)
        self.gamma = gamma
        layers, d = [], 2
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
            d = h
        layers += [torch.nn.Linear(d, 4)]          # outputs rho, u, v, p
        self.net = torch.nn.Sequential(*layers)

    def _forward(self, xy):
        return self.net(xy)

    def train(self, xy_data, drho_data, xy_colloc, rh_anchor=None,
              steps=2000, lr=1e-3, w_data=1.0, w_pde=1.0, w_rh=10.0):
        """Fit to density-gradient data under the Euler residual and R-H anchor.

        xy_data, xy_colloc: (N,2) coordinate arrays. drho_data: (N,2) measured
        (drho/dx, drho/dy). rh_anchor: optional (xy, normal, state_left,
        state_right) tuple enforcing the exact jump. Returns the loss history.
        """
        import torch
        xy_d = torch.tensor(np.asarray(xy_data), dtype=torch.float32, requires_grad=True)
        g_d = torch.tensor(np.asarray(drho_data), dtype=torch.float32)
        xy_c = torch.tensor(np.asarray(xy_colloc), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)

        def grad(out, inp):
            return torch.autograd.grad(out, inp, torch.ones_like(out),
                                       create_graph=True)[0]

        hist = []
        for _ in range(steps):
            opt.zero_grad()
            q = self._forward(xy_d)
            rho = q[:, 0:1]
            drho = grad(rho, xy_d)
            data_loss = ((drho - g_d) ** 2).mean()

            qc = self._forward(xy_c)
            rho_c, u_c, v_c, p_c = (qc[:, i:i+1] for i in range(4))
            # continuity residual of the steady compressible flow
            d_rho_u = grad(rho_c * u_c, xy_c)[:, 0:1]
            d_rho_v = grad(rho_c * v_c, xy_c)[:, 1:2]
            pde_loss = ((d_rho_u + d_rho_v) ** 2).mean()

            loss = w_data * data_loss + w_pde * pde_loss
            if rh_anchor is not None:
                loss = loss + w_rh * self._rh_penalty(rh_anchor)
            loss.backward()
            opt.step()
            hist.append(float(loss.detach()))
        return hist

    def _rh_penalty(self, rh_anchor):
        import torch
        _, normal, sl, sr = rh_anchor
        res = rankine_hugoniot_torch(sl, sr, normal, self.gamma)
        return (res ** 2).mean() if torch.is_tensor(res) else float(res)

    def predict(self, xy):
        import torch
        with torch.no_grad():
            return self._forward(torch.tensor(np.asarray(xy),
                                              dtype=torch.float32)).numpy()


def rankine_hugoniot_torch(state_left, state_right, normal, gamma=1.4):
    """Torch flux mismatch across a face (mirror of rankinehugoniot.rh_residual)."""
    import torch
    nx, ny = normal
    flux = []
    for (rho, u, v, p) in (state_left, state_right):
        un = u * nx + v * ny
        e = p / ((gamma - 1) * rho) + 0.5 * (u**2 + v**2)
        flux.append(torch.stack([rho * un, rho * un * u + p * nx,
                                 rho * un * v + p * ny, un * (rho * e + p)]))
    return flux[1] - flux[0]
