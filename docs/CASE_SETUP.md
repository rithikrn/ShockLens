# Setting up your case for ShockLens

ShockLens reads three things. The shock needs the density field. Separation and the
wall-pressure rise need the wall patch. The breathing spectrum needs a time series
of wall pressure. Here is exactly what to add to your OpenFOAM case so all three
exist, and how to point `case.yml` at them.

If a piece is missing, ShockLens still runs and just warns, so you can start with
the field only and add the rest later.

## What produces each input

| ShockLens wants | You add | case.yml key |
|---|---|---|
| density field (shock, contours, schlieren) | nothing, `foamToVTK` writes rho/U/p | `vtk` |
| skin friction + wall pressure (separation, pressure rise) | `wallShearStress` function object | `wall_vtk`, `cf_denom` |
| unsteady wall pressure (RMS, breathing rate) | `probes` function object | `probe_file`, `probe_col` |

## 1. The field (always)

Run `foamToVTK` after the solver. It writes the internal field to
`VTK/<case>_<t>.vtk` and every boundary patch to `VTK/<patch>/<patch>_<t>.vtk`.
Density comes for free. Point `vtk:` at the internal-field file.

## 2. Wall shear and wall pressure (for separation)

Add this to `system/controlDict` so the wall patch carries `wallShearStress` (and
`p` rides along automatically):

```
functions
{
    wallShearStress
    {
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        patches         (bottom);          // your wall patch name
        writeControl    writeTime;
    }
}
```

After `foamToVTK`, the wall patch file `VTK/bottom/bottom_<t>.vtk` has both
`wallShearStress` and `p`. Point `wall_vtk:` at it.

`cf_denom` is the dynamic pressure that turns wall shear into skin friction:
`cf_denom = 0.5 * rho_inf * U_inf^2`. For the bundled ramp (rho_inf = 1.4,
U_inf = 2) that is 2.8. ShockLens projects the shear onto the local wall tangent,
so an inclined or curved ramp is handled correctly, not just a flat plate.

## 3. Time-resolved wall pressure (for the breathing spectrum)

Add a `probes` function object with a few points just above the wall:

```
    wallPressureProbes
    {
        type            probes;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (p);
        probeLocations
        (
            (0.05 1e-4 0)
            (0.10 1e-4 0)
            (0.15 1e-4 0)
        );
    }
```

This writes `postProcessing/wallPressureProbes/0/p`, one column of time then one
per probe. Point `probe_file:` at it and set `probe_col` to the probe under the
interaction (1 is the first probe after the time column). Set `writeInterval` small
enough to resolve the unsteadiness if you want a clean spectrum.

## The case.yml that ties it together

```yaml
name: myCase
vtk: case/VTK/case_8.vtk
fields: [rho]
detector: oblique_ransac
nx: 320
ny: 180
mach: 2.0
theta_deg: 15.0
wall_vtk: case/VTK/bottom/bottom_8.vtk
cf_denom: 2.8
probe_file: case/postProcessing/wallPressureProbes/0/p
probe_col: 3
```

Then one command does everything:

```bash
shocklens run-case myCase.yml --outdir results
```

You get the results summary printed and the figures written to `results/`. The YAML
is the only per-case file, so this works for any case, not just the ramp.

## Later: heat flux for transition

When you move to transition work, add a `wallHeatFlux` function object the same
way. ShockLens does not use it yet, but it is the natural next wall quantity, so
writing it now means the data is there when the regime classifier lands.
