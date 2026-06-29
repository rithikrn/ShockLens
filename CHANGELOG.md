# Changelog

## 0.1.1

`run-case` is now the whole pipeline from one YAML, no per-case script.

- One command turns a case.yml into: shock angle and strength (pressure and
  density ratio, downstream Mach), separation (x_sep, x_reatt, L_sep, Cf_min),
  wall-pressure rise, the breathing spectrum, six figures, and a results summary.
- Reliable on real walls: skin friction is projected onto the local wall tangent
  (inclined or curved ramps, not just flat plates), 2D-slab duplicate points are
  collapsed, and point- or cell-data VTK patches both work.
- Honesty checks: warns on missing inputs, no separation, odd Cf sign, or a shock
  angle far from theory, instead of failing silently.
- New optional case.yml keys: wall_vtk, cf_denom, wall_shear_field, probe_file,
  probe_col, outdir. See docs/CASE_SETUP.md for what to add to your OpenFOAM case.
- Note: run-case output is now grouped (shock / separation / wall_pressure_*).

## 0.1.0

First release. Runs offline end to end, no solver needed.

- Shock detection, separation from Cf, wall-pressure RMS/PSD, shock-breathing rate.
- Shock tracking over time.
- ML: sparse-sensor prediction, plus a forecaster for shock motion.
- Assimilation: exact Rankine-Hugoniot jumps, BOS handling, and a tracker that
  recovers velocity and pressure from a density gradient (optional PINN refiner).
- Examples: the inviscid wedge (exact angle check) and a laminar compression ramp.
- CLI: demo, extract, train, track, forecast, mlcard, report, overlay, assimilate, run-case, info.
- 35 tests plus a 10-check smoke test on the real wedge physics.
