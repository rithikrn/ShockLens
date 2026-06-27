# compressionRamp_2D: laminar SBLI compression corner

The flagship case: a laminar boundary layer running into a 15-degree compression
corner at Mach 2, which separates and forms a shock-boundary-layer interaction.
Unlike the inviscid wedge (shock only), this case exercises the whole tool: the
shock, the separation bubble, and the unsteady wall pressure.

This is a complete, self-contained OpenFOAM case under `case/`. It is built from
the validated wedge tutorial, so the numerics are the same proven `rhoCentralFoam`
setup. It has not been run on this machine, so treat it as a starting case:
run it, pass the checks in "Verify it separated" below, and tune the angle, the
Reynolds number, or the mesh if it does not behave.

## The setup, and why these numbers

The case is non-dimensional with sound speed `a = 1` (the wedge convention, set by
`molWeight 11640.3`, `Cp 2.5`). The freestream you need for post-processing:

| Quantity | Value |
|---|---|
| Mach number | 2.0 (`U = (2 0 0)`, since `a = 1`) |
| Density `rho_inf` | 1.4  (= `p/(R T)` with `p=1`, `T=1`, `R=0.7143`) |
| Pressure `p_inf`, Temperature `T_inf` | 1, 1 |
| Viscosity `mu` | 2.8e-4 |
| Plate length `L` (inlet to corner) | 1.0 |
| Reynolds number `Re_L = rho U L / mu` | 1.0e4 |
| Ramp angle | 15 degrees |
| Cf denominator `0.5 rho_inf U_inf^2` | 2.8 |

Three choices matter. The 15-degree ramp gives an attached oblique shock: the
maximum deflection at Mach 2 is 22.97 degrees, so 15 sits safely below it and the
shock angle is about 45 degrees, not a detached bow shock. `Re_L = 1e4` keeps the
boundary layer laminar and thick enough (delta around 0.05 to 0.1) to resolve and
to separate under the corner's adverse pressure gradient. And the corner shock
reaches the top boundary near `x = 1`, reflects, and leaves through the outlet at
`x = 1.5` before it can fall back on the interaction, so the top `slip` boundary
does not pollute the result.

## Files

```
case/
├── 0/                 U, p, T  (freestream M=2; no-slip wall; slip top)
├── constant/          thermophysicalProperties (mu, Pr), turbulenceProperties (laminar)
├── system/
│   ├── blockMeshDict  flat plate [-1,0] + 15 deg ramp [0,1.5], y graded to the wall
│   ├── controlDict    rhoCentralFoam + wallShearStress + near-wall pressure probes
│   ├── fvSchemes      Kurganov flux, vanLeer reconstruction
│   └── fvSolution     diagonal for the explicit fields, smoothSolver for U and e
├── Allrun             blockMesh -> checkMesh -> rhoCentralFoam -> foamToVTK
└── Allclean
```

## Run it

```bash
cd case
./Allrun
```

This meshes, checks the mesh, runs the solver, and writes VTK. `rhoCentralFoam` is
explicit, so the run is many thousands of small time steps; expect it to take a
while. Watch `log.rhoCentralFoam` for the residuals to settle.

## Verify it separated (do not skip this)

1. Mesh is valid: `checkMesh` reports no errors. If `blockMesh` complains about a
   face orientation, reverse the four vertices of that face in `blockMeshDict`. If
   the cells cluster at the top instead of the wall, flip the y grading from
   `(1 8 1)` to `(1 0.125 1)`.
2. There is a shock: open the latest time in ParaView, or run the shock extract
   below, and confirm a single oblique density jump from near the corner.
3. It actually separated: build the `Cf` profile (next section) and confirm `Cf`
   goes negative over a region. If it stays positive everywhere, the corner did
   not separate. Increase the ramp angle toward 18 to 20 degrees (stay under 22.97
   at Mach 2), or lower `mu` to raise the Reynolds number, and rerun.

## Feed the results to ShockLens

After `foamToVTK`, the internal field is `case/VTK/case_<t>.vtk` and each patch is
under `case/VTK/<patch>/`. Newer OpenFOAM writes `.vtm`/`.vtu`; point ShockLens at
whatever was written.

### Shock location and angle (density field)

```bash
shocklens extract case/VTK/case_8.vtk --nx 320 --ny 180
```

Or edit `case.yml` so `vtk:` points at the file, then
`shocklens run-case examples/compressionRamp_2D/case.yml`. The `theta-beta-M`
check it prints is approximate here: separation pushes the shock upstream and
changes its angle, so for the viscous ramp the detected angle will not match the
inviscid 45 degrees exactly. The exact validation belongs to the wedge case; here
the number is a sanity bound.

For the field-level overlay on this real field:

```python
import shocklens as sl
from shocklens import io, plots
raw = io.read_vtk_slice("case/VTK/case_8.vtk", fields=("rho",))
rho, x, y = io.to_uniform_grid(raw["points"], raw["rho"], 320, 180)
field = {"rho": rho, "x": x, "y": y, "dx": x[1]-x[0], "dy": y[1]-y[0]}
got = sl.detect.detect(field, method="oblique_ransac")
plots.plot_detection_overlay(field, got, "ramp_overlay.png")
```

### Separation, reattachment, L_sep (skin friction)

Skin friction is the streamwise wall shear over the freestream dynamic pressure,
and the denominator for this case is 2.8.

```python
import numpy as np, pyvista as pv, shocklens as sl
wss = pv.read("case/VTK/bottom/bottom_8.vtk")      # the wall patch
x = wss.points[:, 0]
tau_wx = wss["wallShearStress"][:, 0]              # streamwise component
o = np.argsort(x); x, tau_wx = x[o], tau_wx[o]
cf = tau_wx / 2.8                                  # 0.5 * rho_inf * U_inf^2
print(sl.separation.separation_points(x, cf))      # x_sep, x_reatt, L_sep, separated
sl.plots.plot_cf(x, cf, "ramp_cf.png")
```

If `Cf` is negative everywhere the flow looks attached, your solver writes wall
shear with the opposite sign; use `cf = -cf`.

### Wall-pressure loading and breathing frequency (probes)

```python
import numpy as np, shocklens as sl
d = np.loadtxt("case/postProcessing/wallPressureProbes/0/p")  # cols: time, p0..p4
t = d[:, 0]
p = d[:, 3]                                         # probe at x=0.1, under the interaction
fs = 1.0 / np.mean(np.diff(t))
pf = p - p.mean()
print("RMS:", sl.separation.pressure_rms(pf))
print("breathing freq:", sl.separation.dominant_frequency(pf, fs))
sl.plots.plot_wall_pressure(t, p, fs, "ramp_wall_pressure.png")
```

### Shock motion over time (sequence of fields)

```python
import glob, shocklens as sl
paths = sorted(glob.glob("case/VTK/case_*.vtk"))
tr = sl.track.track_vtk_series(paths, field="rho", nx=320, ny=180)
print("breathing freq from shock motion:", tr["f_breathing"])
```

For the full generic workflow and gotchas, see `../../docs/TUTORIAL.md`.

## Caveats

This case is unrun here, so the mesh resolution, the time step, and whether it
separates for these exact numbers are yours to confirm. The top `slip` boundary
reflects the corner shock (by design it exits before the interaction, but if you
shorten the domain that stops being true). Once you have a separated, converged
result, commit the `case.yml` and the produced numbers as the reproducibility
record.
