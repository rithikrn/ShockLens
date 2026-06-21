# forwardStep_Ma3

The Mach 3 supersonic forward-facing step from the OpenFOAM tutorials. The step
throws off shocks and reflections, which makes it a good shock-capture and
schlieren baseline before moving to a real boundary-layer interaction.

```bash
cp -r $FOAM_TUTORIALS/compressible/rhoCentralFoam/forwardStep ./case
(cd case && ./Allrun && foamToVTK)
shocklens extract case/VTK/case_*.vtk --nx 360 --ny 120
```

Notes:
- This case has multiple shocks; v0.1 fits the single dominant ridge. Treat the
  reported angle as the leading-shock estimate. Multi-shock labelling is v0.2.
- Numerical schlieren (`shocklens.detect.schlieren`) is the clearest view here.
