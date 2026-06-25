# Assimilation: velocity and pressure from a density gradient

Experimental SBLI campaigns can measure the density gradient cheaply with
Background-Oriented Schlieren (BOS), but velocity is hard to get near the shock:
PIV seeding and laser sheets are disrupted exactly where the interaction matters.
Recent work recovers velocity and pressure by training a RANS- or Euler-
constrained physics-informed neural network on the BOS density gradient alone,
validated against a Mach 2.28 DNS and a Mach 2.0 wind-tunnel case. The known weak
spot of that approach is the shock itself, where a network oscillates.

ShockLens inverts the priority, in line with its physics-first rule.

1. Track the shock from the gradient ridge (`detect`, outlier-resistant RANSAC line fit).
2. Set the jump across it with the exact Rankine-Hugoniot relations
   (`rankinehugoniot`), not the network.
3. Fill the smooth regions from the upstream state and the exact downstream
   state (`assimilate.ShockAssimilator`).

That baseline is exact for a piecewise-uniform oblique shock and is the leading-
order field a PINN would refine. The optional `shocklens.pinn` module is that
refinement: a small Euler-residual network that shapes only the smooth flow,
with the tracked shock and the R-H jump entering as hard constraints, so the
network never has to represent the discontinuity.

## Run it

```bash
shocklens assimilate --mach 3.0 --theta 18 --noise 0.02
```

Prints the tracked shock angle and the recovered downstream Mach, pressure ratio,
and velocity, and writes `figures/assimilated_u.png`. The BOS forward/inverse
path (`bos.synthetic_bos`, `bos.density_gradient_from_bos`) stands in for an
experimental image offline; on real data, feed your measured gradient instead.

## Why this is not a black box

The shock location and the jump are physics. The network, when used, only
interpolates the smooth regions and is pinned at the interface by Rankine-
Hugoniot. Every recovered number traces back to a conservation law or a tracked
coordinate.

## References

- Assimilating mean velocity fields of an SBLI from BOS via a RANS-constrained
  PINN, validated on Mach 2.28 DNS and Mach 2.0 wind-tunnel data.
- Z. Mao, A. D. Jagtap, G. E. Karniadakis, inverse supersonic-flow problems from
  Schlieren with XPINNs, J. Comput. Phys. 466, 111402 (2022).
- S. Cai et al., velocity/pressure from tomographic BOS via PINNs (espresso cup),
  J. Fluid Mech. 915, A102 (2021).
- L. Liu et al., Discontinuity Computing Using PINNs (R-H as a hard constraint),
  J. Sci. Comput. 2023; review arXiv:2503.17379 (2025) for the smoothness bias.
