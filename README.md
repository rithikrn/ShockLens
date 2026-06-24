# ShockLens

**Turn shock-dominated CFD into interpretable SBLI events, then predict them from sparse sensors.**

ShockLens reads a compressible flow field (OpenFOAM `foamToVTK` output, or any
VTK), automatically extracts the shock-boundary-layer-interaction quantities that
actually matter, shock location and angle, separation and reattachment points,
separation length, wall-pressure loading, unsteady shock-breathing frequency, and
trains a small model to predict those quantities from a handful of wall-pressure
sensors. It runs end to end offline on synthetic, ground-truth-labelled data, so
you can install it and see the whole pipeline work before touching a solver.

The ML is deliberately small. v0.1 is a random-forest baseline over sparse
sensors. Temporal models and neural operators come later, once the extraction is
solid. The contribution is the extraction-and-benchmark layer, not the model.

## Where it fits in your workflow

ShockLens does not run CFD. It post-processes fields a simulation already
produced. The loop is: run the case once in OpenFOAM (or any solver, or an
experiment) and export VTK, then ShockLens reads that VTK, extracts the SBLI
events, and trains on them. Because it reads generic VTK, it is not tied to
OpenFOAM. The bundled synthetic data lets you run the whole tool with no
simulation at all, which is how the demo, tests, and CI work.

```
solver run (once)  ->  VTK  ->  ShockLens: extract events  ->  ML on sparse sensors  ->  scorecard + figures
```

## Figures

`shocklens plot` renders these from the offline data; on real cases they come
from your fields.

| | |
|---|---|
| ![schlieren](docs/figures/schlieren.png) | ![skin friction](docs/figures/skin_friction.png) |
| Numerical schlieren of the oblique shock | Cf with the separation bubble marked |
| ![wall pressure](docs/figures/wall_pressure.png) | ![trajectory](docs/figures/shock_trajectory.png) |
| Wall-pressure signal and its spectrum | Shock-foot trajectory over time |

## The machine learning

ShockLens has two models, both deliberately light (no GPU, no deep-learning
stack in v0.1):

1. Sparse-sensor prediction. A random forest maps a handful of wall-pressure
   sensor readings to an SBLI target (separation length, shock position). This is
   the "predict the interaction from limited measurement" task that matters when
   you cannot see the whole field, as in experiments.
2. Shock-motion forecasting. A gradient-boosting model predicts where the shock
   goes next from its recent trajectory. That is the early-warning signal for
   separation and unstart. Run `shocklens forecast`.

The physics-extraction modules are not a detour from the ML, they are its data
engine: you cannot learn shock or separation behaviour without first turning raw
fields into clean, labelled trajectories, which is exactly what `detect`,
`separation`, and `track` produce. As fidelity rises (LES/DNS, real cases), the
ML grows: temporal deep models (LSTM/TCN), field reconstruction from sensors via
operator learning, and a forecaster that can sit inside a control loop. The
extraction layer keeps producing the labels those models train on.

```bash
shocklens forecast        # predict future shock position from its history
```

## Motivation

Shock-boundary-layer interaction limits high-speed flight: a small shock
displacement can trigger separation, large wall-pressure fluctuations, buffet,
thermal and structural loading, and inlet unstart. Its low-frequency unsteadiness
has been an open problem since Dolling's 2001 "what next" review and remains one.
This is the focus of the CASL group at FSU, whose SBLI work spans wedge/Mach
studies, 3D interactions, and hypersonic boundary layers.

The field is moving fast, and the activity sorts into three camps:

- ML as a PDE solver. PINNs and neural operators promise mesh-free solutions, but
  they are biased toward smooth functions and oscillate at discontinuities, so
  shocks remain a hard, actively-patched case (review: arXiv:2503.17379, 2025;
  Jagtap et al., J. Comput. Phys. 466, 2022). Asking a network to *be* the shock
  solver fights its own architecture.
- ML on extracted quantities. Sparse near-wall pressure can reconstruct SBLI
  pressure fields (Physics of Fluids 35, 076117, 2023), and sensor-based ROMs
  recover full state from a few probes (Loiseau, Noack & Brunton, J. Fluid Mech.
  844, 2018; Fukami et al., Nat. Mach. Intell. 3, 2021). These work because they
  operate on physically-meaningful, low-dimensional signals.
- Control. Deep RL suppresses shock-induced separation on a laminar
  compression-ramp SBLI in OpenFOAM, using the wall-pressure coefficient as the
  state and skin-friction-based separation as the reward (AIAA Journal, 2025,
  doi:10.2514/1.J065230). The controller consumes exactly the events ShockLens
  extracts.

Meanwhile GPU DNS solvers (STREAmS, Comput. Phys. Commun. 2021/2023) now generate
SBLI fields faster than anyone can interpret them. Full reference list in
`REFERENCES.md`.

## The gap

> The methods that work, sparse-sensor ROMs and RL control, all consume the same
> interpretable SBLI events: shock location, separation length, wall-pressure
> spectra. But there is no simple, reproducible, solver-agnostic tool that
> reliably *extracts those events* from a field and serves them to the ML layer.
> Everyone re-implements ad-hoc extraction.

ShockLens fills that gap, and it scales from 2D tutorials to 3D LES.

## Design philosophy: physics is the cake, ML is the icing

The shock is found by physics, not by a network. Detection, separation, and
spectra are gradient, zero-crossing, and PSD operations validated against closed-
form answers (the theta-beta-M angle, known Cf crossings). They do not learn, so
they do not hallucinate, and they work the first time on a case they have never
seen. ML only ever acts on the *outputs* of that physics, low-dimensional, labelled
events, never on raw pixels in place of the physics:

- sparse sensors -> separation length / shock position (with uncertainty),
- shock trajectory -> short-horizon forecast (early warning of separation/unstart),
- and, later, those same events as the state/reward for an RL controller, or as
  hard constraints on a PINN super-resolver.

This ordering is deliberate. It is why the tool is trustworthy on real data, why
the ML stays small and honest, and why GPU/PINN/RL are additions on a solid base
rather than a black box asked to do everything. The numbers the ML reports on the
bundled synthetic data are pipeline checks, not evidence of predictive power;
that evidence comes from real cases, and the tool is built to measure it honestly.

## What it extracts

| Quantity | Why it matters |
|---|---|
| Shock-foot location and angle | shock position, oscillation, theta-beta-M validation |
| Separation point `x_sep` | onset of shock-induced separation |
| Reattachment point `x_reatt` | bubble recovery |
| Separation length `L_sep` | clearest single SBLI severity metric |
| Wall pressure and RMS | structural loading, buffet, inlet stability |
| Skin friction `Cf` | separation/reattachment indicator |
| Shock-breathing frequency | low-frequency unsteadiness from a Welch PSD |
| Numerical schlieren | shock visualisation and an ML input field |

## Repository layout

```text
shocklens/
├── shocklens/
│   ├── synthetic.py    # ground-truth oblique-shock + ramp data (offline, testable)
│   ├── detect.py       # numerical schlieren, shock-line fit, angle + foot
│   ├── separation.py   # Cf zero-crossings, RMS, PSD, breathing frequency
│   ├── track.py        # shock-foot trajectory over time (batch / unsteady)
│   ├── models.py       # sparse-sensor random-forest baseline (save/load)
│   ├── forecast.py     # gradient-boosting shock-motion forecaster
│   ├── plots.py        # schlieren, Cf, wall-pressure, trajectory figures
│   ├── metrics.py      # SBLI scorecard
│   ├── io.py           # VTK reader (OpenFOAM or any solver) + regrid
│   └── cli.py          # demo | extract | train | track | forecast | plot | info
├── examples/           # forwardStep, wedge, compression-ramp workflows
├── docs/figures/       # rendered example figures
├── tests/              # physics checks against known answers
├── comprehensive_smoke_test.py
├── CITATION.cff
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Case suite

Each example adds something the previous one does not, so the suite shows the
tool's range rather than three variations of one check.

| Case | Adds | Source | Status |
|---|---|---|---|
| `compressionRamp_2D` (flagship) | full SBLI: separation + reattachment + L_sep + breathing PSD + shock overlay | compression-ramp SBLI | v0.1 offline, real-run recipe |
| `wedge_oblique` | exact theta-beta-M angle validation | analytic + `rhoCentralFoam/wedge15Ma5` (real tutorial) | v0.1 |
| `forwardStep_Ma3` | shock capture / numerical schlieren | `rhoCentralFoam/forwardStep` | v0.1 |
| `compressionRamp_3D_LES` | spanwise corrugation, bubble breathing | 3D LES/DES | v0.3 |

## Install

```bash
git clone https://github.com/yourusername/shocklens.git
cd shocklens
pip install -e .              # core: numpy, scipy, scikit-learn
pip install -e ".[vtk]"       # adds pyvista for reading solver output
pip install -e ".[dev]"       # adds pytest
```

No PyTorch needed for v0.1.

## Quickstart, no solver required

```bash
shocklens demo            # extraction + prediction scorecard
shocklens track           # follow an oscillating shock, recover the breathing rate
shocklens plot            # write the figures above to ./figures
```

`shocklens demo` prints the offline scorecard:

```json
{
  "shock_angle_detected": 32.24,
  "shock_angle_true": 32.24,
  "shock_foot_detected": 0.2,
  "separation": {"x_sep": 0.43, "x_reatt": 0.629, "L_sep": 0.199, "separated": true},
  "L_sep_prediction_r2": 0.96,
  "L_sep_prediction_mae": 0.0109
}
```

The detected shock angle matches the theta-beta-M value because the synthetic
field is built from it; that is the test, not a coincidence.

## Reproducibility

Everything offline is deterministic under fixed seeds, so `shocklens demo` and
the test suite produce the same numbers on any machine. The core depends only on
numpy, scipy, and scikit-learn, with no GPU and no compiled extensions, so it
installs the same way everywhere. To reproduce the full check:

```bash
pip install -e ".[dev]"
pytest -q                       # 15 physics + pipeline tests
python comprehensive_smoke_test.py   # includes the VTK round trip
```

Trained models persist with `model.save(path)` / `SparseSensorModel.load(path)`.

## On real solver output

```bash
# after running an OpenFOAM rhoCentralFoam case and foamToVTK:
shocklens extract path/to/case/VTK/case_100.vtk --nx 300 --ny 160
```

See each example's README for the full workflow.

## Why this over a generic neural-operator surrogate

A full-field FNO surrogate predicts everything and is judged by RMSE, which is a
poor signal near a shock and a crowded research space. ShockLens predicts the
specific SBLI events that drive loading and control, which is more interpretable,
cheaper to train, and a sharper contribution. It also gets more valuable as
fidelity rises: the richer the LES/DNS field, the more there is to extract.

# References

The work ShockLens sits next to, and why. Paraphrased; follow the DOIs for the
originals.

## The enduring problem

- D. S. Dolling, "Fifty Years of Shock-Wave/Boundary-Layer Interaction Research:
  What Next?", AIAA Journal 39(8), 2001. The agenda-setting review; the
  low-frequency unsteadiness of shock-induced separation is still open.
- B. Ganapathisubramani, N. T. Clemens, D. S. Dolling, "Low-frequency dynamics of
  shock-induced separation in a compression ramp interaction," J. Fluid Mech.
  636, 397-425, 2009. The canonical unsteadiness study ShockLens' PSD targets.

## High-fidelity data is now cheap to generate, expensive to interpret

- M. Bernardini et al., "STREAmS: a high-fidelity accelerated solver for DNS of
  compressible turbulent flows," Comput. Phys. Commun. 263, 107906, 2021; and
  STREAmS-2.0, CPC 285, 108644, 2023. GPU-accelerated DNS of SBLI: terabytes of
  fields that still need event-level interpretation.
- N. Goffart, B. Tartinville, S. Pirozzoli, "From High-Fidelity High-Order to
  Reduced-Order Modeling for Unsteady Shock Wave/Boundary Layer Interactions,"
  2025. The high-fidelity-to-ROM direction ShockLens serves.

## ML as a PDE solver struggles at shocks (why physics-first extraction)

- "Challenges and Advancements in Modeling Shock Fronts with Physics-Informed
  Neural Networks: A Review and Benchmarking Study," arXiv:2503.17379, 2025.
  Vanilla PINNs are biased toward smooth functions and oscillate at
  discontinuities; capturing shocks needs special machinery.
- A. D. Jagtap, Z. Mao, N. Adams, G. E. Karniadakis, "Physics-informed neural
  networks for inverse problems in supersonic flows," J. Comput. Phys. 466,
  111402, 2022. State of the art, and still fighting the smoothness bias.

## ML on extracted quantities works (what ShockLens feeds)

- "Deep learning reconstruction of pressure fluctuations in supersonic
  shock-boundary layer interaction," Physics of Fluids 35(7), 076117, 2023.
  Reconstructs near-wall pressure from sparse signals: ShockLens' sparse-sensor
  premise, demonstrated.
- K. Fukami, R. Maulik, et al., "Global field reconstruction from sparse sensors
  with Voronoi-tessellation-assisted deep learning," Nature Machine Intelligence
  3, 945-951, 2021.
- J.-C. Loiseau, B. R. Noack, S. L. Brunton, "Sparse reduced-order modelling:
  sensor-based dynamics to full-state estimation," J. Fluid Mech. 844, 459-490,
  2018.

## Control needs the events as state and reward

- "Control of Hypersonic Shock-Wave/Laminar-Boundary-Layer Interaction Using Deep
  Reinforcement Learning," AIAA Journal, 2025, doi:10.2514/1.J065230. A laminar
  compression ramp in OpenFOAM where the wall-pressure coefficient is the RL
  state and the skin-friction-based separation is the reward, the exact
  quantities ShockLens extracts, with a reported ~40% separation reduction.
- E. Parish, D. S. Ching, et al., "Data-driven turbulent Prandtl number modeling
  for hypersonic shock-boundary-layer interactions," AIAA Journal, 2024.

# Roadmap

ShockLens v0.1 is a 2D shock-event-extraction skeleton with a baseline ML layer.
The point of the architecture is that each expansion axis below is a *natural
addition* at a defined extension point, not a rewrite. The seams already exist;
filling them is the work.

## Expansion axes and the seams that carry them

| Axis | Extension point (already in place) | What's left to build |
|------|-----------------------------------|----------------------|
| 2D -> 3D detection | detector registry (`detect.register_detector`); `detect()` dispatch | a `shock_surface_3d` detector that fits a surface, not a line |
| 2D -> 3D separation | `separation.separation_field` (spanwise stack of Cf) | wire to a real 3D wall patch |
| Robust / high-fidelity (RANSAC -> LES/DNS) | `oblique_ransac` detector | connected-components + multi-shock labelling for competing structures |
| Different solvers | `io.FIELD_ALIASES` name resolution | add aliases / a reader per solver |
| GPU | `backend.array_namespace` (ops follow the input's array module) | validate the cupy path on hardware |
| Point-wise vs field-wise | detectors return point estimates today | add field-wise outputs (per-cell shock indicator) |
| Pointwise uncertainty | RF tree-spread in `SparseSensorModel` | propagate to detection (ensemble of fits) |

## Near-term (v0.2): make it a real SBLI tool

- One real OpenFOAM run committed end to end (the wedge `Allrun` is wired).
- Real compression-ramp case: `wallShearStress -> Cf -> x_sep/x_reatt/L_sep`.
- Field-level true-vs-detected overlay figure on real VTK.
- Honest real-data ML metrics (expected to drop from the synthetic ~1.0).

## Later (v0.3 / v0.4)

- 3D LES fields: spanwise shock corrugation, bubble breathing, RANS-vs-LES events.
- `shocklens/inlet.py`: scramjet / mixed-compression inlet metrics (shock-train
  and terminal-shock tracking, isolator pressure rise, unstart early-warning,
  exit-plane distortion).

## What this does and does not change

Refactoring to these seams lifts expandability and implementation maturity and
keeps the path to higher novelty open. It does **not** by itself make the tool
novel or widely adopted: that still requires the real-data predictability study.
The seams just make sure that work is additive, not a teardown.

## AI/ML/GPU as physics-constrained icing

Every learned or accelerated component sits on top of the physics extraction and
is constrained by it, never replacing it. That is the rule that keeps the tool
from becoming a black box.

| Addition | How physics constrains it | Grounded in |
|----------|---------------------------|-------------|
| Sparse-sensor prediction (here) | trained on extracted, theory-validated events, not raw fields | Phys. Fluids 35, 076117 (2023); Loiseau et al., JFM 844 (2018) |
| Shock-motion forecasting (here) | input is the detector's trajectory; scored vs persistence | Ganapathisubramani et al., JFM 636 (2009) |
| RL control hook | state and reward are extracted wall pressure / Cf separation | AIAA J. 2025, doi:10.2514/1.J065230 |
| PINN / neural-operator super-resolution | extracted shock location enters as a hard constraint / loss region, sidestepping the smoothness bias that makes PINNs oscillate at shocks | review arXiv:2503.17379 (2025) |
| GPU acceleration | `backend.array_namespace` swaps the array module; physics ops are unchanged | STREAmS, CPC 263 (2021) / 285 (2023) |

The point of the table is the middle column: in each row the learning is anchored
to a quantity the physics already computed and validated, so the model cannot
drift away from the physics it is supposed to respect.

## License

MIT. See [LICENSE](LICENSE).
