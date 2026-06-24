# Porting ShockLens to your own case

Four steps, no code.

1. Run your simulation and export VTK (OpenFOAM: `foamToVTK`). Anything PyVista
   can read works, so you are not tied to OpenFOAM.
2. Copy `examples/wedge_oblique/case.yaml` to `mycase.yaml` and edit:
   - `vtk`: path to your `.vtk`
   - `fields`: your density field name (`rho`, `Density`, ...; aliases resolve,
     and an unknown name is tried literally)
   - `mach`, `theta_deg`: optional; turns on the theta-beta-M angle check
   - `detector`: `oblique_ransac` if the field is noisy or turbulent
3. `shocklens run-case mycase.yaml`
4. Read the scorecard JSON. With `mach`/`theta_deg` set you also get the detected
   vs theoretical shock angle and the error.

That is the whole port. The case file is also your reproducibility record: commit
it and anyone re-runs your result with one command.
