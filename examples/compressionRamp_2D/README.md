# compressionRamp_2D: laminar SBLI compression corner

The main case. A laminar boundary layer hits a 15 degree corner at Mach 2,
separates, and throws a shock. The wedge only gives you a shock; this one gives
you the whole interaction: shock, separation bubble, and unsteady wall pressure.

It's a full OpenFOAM case under `case/`, built from the validated wedge so the
numerics are the same. Heads up: I haven't run it yet, so treat it as a starting
point. Run it, check it separated (below), and nudge the angle or Reynolds number
if it doesn't.

## The numbers

Non-dimensional, sound speed `a = 1`. What you need for post-processing:

| Thing | Value |
|---|---|
| Mach | 2.0 (`U = (2 0 0)`) |
| `rho_inf` | 1.4 |
| `p_inf`, `T_inf` | 1, 1 |
| `mu` | 2.8e-4 |
| `Re_L` (L = 1) | 1e4 |
| Ramp angle | 15 degrees |
| Cf denominator `0.5 rho U^2` | 2.8 |

Why these: 15 degrees stays under the Mach 2 detachment limit (22.97), so the
shock is a clean attached oblique, not a bow shock. Re 1e4 keeps the boundary
layer laminar and thick enough to separate. The corner shock bounces off the top
and leaves through the outlet before it can hit the interaction.

## Run it

```bash
cd case
./Allrun        # blockMesh -> checkMesh -> rhoCentralFoam -> foamToVTK
```

It's an explicit solver, so it's a lot of small steps. Give it time.

## Did it separate? Check this

1. `checkMesh` is clean. If blockMesh complains about a face, reverse that face's
   four vertices. If cells bunch at the top instead of the wall, flip the y
   grading to `(1 0.125 1)`.
2. There's one oblique shock from near the corner (look in ParaView, or run the
   extract below).
3. Cf goes negative somewhere. That's the bubble. If it never does, it didn't
   separate, so bump the angle to 18 or 20 degrees (stay under 22.97) or drop `mu`.

## Feed it to ShockLens

Internal field is `case/VTK/case_<t>.vtk`; the wall patch is under
`case/VTK/bottom/`.

Shock angle:

```bash
shocklens extract case/VTK/case_8.vtk --nx 320 --ny 180
```

Separation (Cf from wall shear, denominator 2.8):

```python
import numpy as np, pyvista as pv, shocklens as sl
wss = pv.read("case/VTK/bottom/bottom_8.vtk")
x = wss.points[:, 0]; tau = wss["wallShearStress"][:, 0]
o = np.argsort(x); x, tau = x[o], tau[o]
cf = tau / 2.8
print(sl.separation.separation_points(x, cf))   # flip sign if Cf looks inverted
```

Wall pressure and breathing rate:

```python
import numpy as np, shocklens as sl
d = np.loadtxt("case/postProcessing/wallPressureProbes/0/p")
t, p = d[:, 0], d[:, 3]            # probe under the interaction
fs = 1 / np.mean(np.diff(t)); pf = p - p.mean()
print(sl.separation.pressure_rms(pf), sl.separation.dominant_frequency(pf, fs))
```

Shock motion over time:

```python
import glob, shocklens as sl
tr = sl.track.track_vtk_series(sorted(glob.glob("case/VTK/case_*.vtk")),
                               field="rho", nx=320, ny=180)
print(tr["f_breathing"])
```

Full workflow and gotchas are in `../../docs/TUTORIAL.md`.

## One note

The `run-case` theta-beta-M check is only a rough guide here. Separation shifts
the shock off the inviscid 45 degrees, so don't read a few degrees of mismatch as
a bug. The exact angle check lives with the wedge.
