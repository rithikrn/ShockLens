# Changelog

## 0.1.0
First release. Offline, solver-free end-to-end pipeline.

- Shock detection: numerical schlieren, shock-line fit, angle and foot location,
  validated against the theta-beta-M relation.
- Separation analysis: separation/reattachment from Cf, separation length.
- Wall signals: pressure RMS, Welch PSD, shock-breathing frequency.
- Time tracking: shock-foot trajectory over a sequence of fields, breathing rate.
- ML: random-forest prediction of SBLI targets from sparse wall sensors (save/load),
  and a gradient-boosting forecaster that predicts future shock motion from its
  recent history (early-warning signal).
- IO: VTK reader (OpenFOAM or any solver) plus regridding.
- Example: the exact OpenFOAM rhoCentralFoam/wedge15Ma5 case (M5, 15 deg wedge).
- Figures: schlieren, Cf with separation, wall-pressure spectrum, shock trajectory.
- CLI: demo, extract, train, track, forecast, plot, info.
- 15 unit tests plus a 10-check comprehensive smoke test that runs the real
  wedge case physics through the extract CLI.
