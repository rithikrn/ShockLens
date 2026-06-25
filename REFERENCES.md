# References

The work ShockLens sits next to, and why. Paraphrased; follow the DOIs for the
originals.

## The enduring problem

- D. S. Dolling, "Fifty Years of Shock-Wave/Boundary-Layer Interaction Research:
  What Next?", AIAA Journal 39(8), 2001. The review that set the agenda; the
  low-frequency unsteadiness of shock-induced separation is still open.
- B. Ganapathisubramani, N. T. Clemens, D. S. Dolling, "Low-frequency dynamics of
  shock-induced separation in a compression ramp interaction," J. Fluid Mech.
  636, 397-425, 2009. The unsteadiness study ShockLens' PSD targets.

## High-fidelity data is cheap to generate, expensive to interpret

- M. Bernardini et al., "STREAmS: a high-fidelity accelerated solver for DNS of
  compressible turbulent flows," Comput. Phys. Commun. 263, 107906, 2021; and
  STREAmS-2.0, CPC 285, 108644, 2023. GPU DNS of SBLI: terabytes of fields that
  still need event-level interpretation.

## ML as a PDE solver struggles at shocks

- "Challenges and Advancements in Modeling Shock Fronts with Physics-Informed
  Neural Networks: A Review and Benchmarking Study," arXiv:2503.17379, 2025.
  Vanilla PINNs are biased toward smooth functions and oscillate at
  discontinuities.
- Z. Mao, A. D. Jagtap, G. E. Karniadakis, "Physics-informed neural networks for
  inverse problems in supersonic flows," J. Comput. Phys. 466, 111402, 2022.
  Recovers fields from Schlieren density gradients with XPINN domain decomposition
  and entropy conditions.
- L. Liu et al., "Discontinuity Computing Using Physics-Informed Neural Networks,"
  J. Sci. Comput. 2023. Adds the Rankine-Hugoniot relation as a hard constraint to
  capture shocks, the idea ShockLens enforces exactly instead of learning.

## ML on extracted quantities works

- "Deep learning reconstruction of pressure fluctuations in supersonic
  shock-boundary layer interaction," Physics of Fluids 35(7), 076117, 2023.
  Reconstructs near-wall pressure from sparse signals.
- K. Fukami, R. Maulik, et al., "Global field reconstruction from sparse sensors
  with Voronoi-tessellation-assisted deep learning," Nature Machine Intelligence
  3, 945-951, 2021.
- J.-C. Loiseau, B. R. Noack, S. L. Brunton, "Sparse reduced-order modelling:
  sensor-based dynamics to full-state estimation," J. Fluid Mech. 844, 459-490,
  2018.

## Assimilation from Background-Oriented Schlieren

- Assimilating mean velocity fields of an SBLI from BOS density gradients with a
  RANS-constrained PINN, validated on a Mach 2.28 DNS and a Mach 2.0 wind-tunnel
  case. The premise behind `shocklens.assimilate`.
- S. Cai et al., "Flow over an espresso cup: inferring 3-D velocity and pressure
  fields from tomographic BOS via physics-informed neural networks," J. Fluid
  Mech. 915, A102, 2021.

## Control needs the events as state and reward

- "Control of Hypersonic Shock-Wave/Laminar-Boundary-Layer Interaction Using Deep
  Reinforcement Learning," AIAA Journal, 2025, doi:10.2514/1.J065230. A laminar
  compression ramp in OpenFOAM where the wall-pressure coefficient is the state
  and skin-friction separation is the reward, the quantities ShockLens extracts.

