# compressionRamp_2D  (v0.2)

The first real SBLI case: a 2D compression ramp where an oblique shock meets a
boundary layer and separates it. This is where ShockLens moves from shock capture
to interaction analysis.

Targets to extract:
- separation and reattachment points from `Cf` (`separation.separation_points`)
- separation length `L_sep`
- wall-pressure plateau and RMS
- shock-foot motion and breathing frequency over time

The offline stand-in is available now:

```python
import numpy as np, shocklens as sl
x = np.linspace(0, 1, 400)
cf = sl.synthetic.ramp_cf_profile(x, x_sep=0.42, x_reatt=0.61)
print(sl.separation.separation_points(x, cf))
```

A runnable OpenFOAM compression-ramp case (IDDES, recycling/rescaling inflow) is
the v0.2 deliverable; OpenFOAM has been used for exactly this (Aerospace 10(10),
892, 2023).
