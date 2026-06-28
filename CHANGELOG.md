# Changelog

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
