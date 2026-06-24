# compressionRamp_2D  (flagship)

The canonical 2D SBLI case, and the one that exercises the whole tool. Unlike the
wedge (a clean shock, no separation), a compression ramp at a strong enough angle
**separates**, so this example shows every extractor at once:

- shock detection with a field-level true-vs-detected overlay,
- separation and reattachment from Cf, and the separation length,
- wall-pressure RMS and the low-frequency breathing spectrum.

This is what "shows the range of the tool" means: one case, four physics events,
each validated.

## What's new here vs the other examples

| Example | What it adds |
|---------|--------------|
| forwardStep_Ma3 | shock capture / numerical schlieren on an inviscid shock-rich flow |
| wedge_oblique | a clean oblique shock with an exact theta-beta-M angle check |
| **compressionRamp_2D** | **the full SBLI: separation + reattachment + L_sep + breathing PSD, plus the shock overlay** |

## Run it offline now (no solver)

```bash
shocklens overlay --outdir figures     # field-level true-vs-detected shock overlay
shocklens demo                          # shock + separation + L_sep scorecard
```

Offline, the synthetic ramp field reproduces a separation shock (springing ahead
of the corner) and a wall recirculation bubble, with known ground truth, so the
overlay error and the separation length are checkable. Expect the detected shock
angle within a few hundredths of a degree of theory, and L_sep ~ 0.22.

Figures: `docs/figures/shock_overlay.png`, `ramp_separation.png`,
`ramp_wall_pressure.png`.

## Produce real VTK (recipe, verified against the wedge tutorial)

There is no stock OpenFOAM compression-ramp SBLI tutorial, but the validated
`wedge_oblique/case` is one edit away from being a viscous separating ramp. The
geometry is already a 15-degree corner. To make it separate:

1. Viscosity on. In `constant/thermophysicalProperties`, set `mu` from `0` to a
   value giving a laminar interaction Reynolds number (e.g. `mu 2e-4`, which with
   the non-dimensional `rho=1`, `U=2.5`, corner length ~0.3 gives Re ~ 4000).
2. No-slip wall. In every `0/` field, change the ramp patch (`obstacle`) and the
   upstream `bottom` patch from `slip`/`symmetryPlane` to a no-slip wall
   (`U`: `fixedValue (0 0 0)`; `p`, `T`: `zeroGradient` for an adiabatic wall).
3. Drop the Mach. In `0/U`, set the inlet to `(2.5 0 0)` (Mach 2.5 separates more
   readily than Mach 5).
4. Refine the near-wall mesh in `system/blockMeshDict` (grade toward the wall) so
   the boundary layer and bubble are resolved.
5. `./Allrun`, then `shocklens run-case case.yaml`.

Verify it actually separates for your angle and Re before trusting the numbers;
this is a starting setup, not a validated result. Your lab's solver works too,
the VTK path is solver-agnostic.

## Port to your own ramp

Edit `case.yaml` (`vtk`, `fields`, `mach`, `theta_deg`) and run
`shocklens run-case case.yaml`. See `docs/PORTING.md`.
