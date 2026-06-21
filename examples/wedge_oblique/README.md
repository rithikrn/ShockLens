# wedge_oblique

The exact OpenFOAM `rhoCentralFoam/wedge15Ma5` tutorial (from OpenFOAM-7),
unmodified, under `case/`. Mach 5 inviscid flow over a 15-degree wedge. The
corner sits at the origin and throws a single oblique shock, whose angle is
fixed by the theta-beta-M relation, so this case validates the detector against
gas dynamics.

For M = 5 and theta = 15 deg the weak-shock angle is beta = 24.3 deg
(`shocklens.synthetic.oblique_beta(5.0, 15.0)`).

## Files

```
case/
├── 0/            T, U, p   (freestream: U=(5 0 0), a=1, so Mach 5)
├── constant/     thermophysicalProperties, turbulenceProperties
├── system/       blockMeshDict (the 15 deg wedge), controlDict, fvSchemes, fvSolution
├── Allrun        blockMesh -> rhoCentralFoam -> foamToVTK
└── Allclean
run_and_extract.sh   solve, then run `shocklens extract` on the result
```

The case is the stock tutorial. The only additions are `Allrun`/`Allclean`
(OpenFOAM-7 ships this case without them) and the extract wrapper.

## Run the real pipeline

With OpenFOAM sourced and ShockLens installed:

```bash
./run_and_extract.sh
```

That solves the case, exports VTK, and prints the detected shock angle next to
the theoretical value. Expect the detected angle within about a degree of 24.3.

## Run the detector offline (no OpenFOAM)

The smoke test reproduces this case's physics without the solver: it builds the
M=5 / 15-degree field on the wedge domain, writes a VTK, and runs the extract
CLI. See `../../comprehensive_smoke_test.py`, check 9.

```python
import shocklens as sl
f = sl.synthetic.oblique_shock_field(mach=5.0, theta_deg=15.0, x_corner=0.0,
                                     xlim=(-0.15242, 0.3048), ylim=(0.0, 0.1524))
print(sl.detect.detect_oblique_shock(f["rho"], f["x"], f["y"])["beta_deg"])  # ~24.3
```

## Note on versions

These files are from OpenFOAM-7 (openfoam.org). On a newer Foundation release or
on ESI OpenFOAM (.com), `turbulenceProperties` may be `momentumTransport` and a
couple of schemes may have moved; the geometry and freestream are unchanged.
