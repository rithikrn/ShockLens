#!/bin/bash
# Full real pipeline: OpenFOAM solve -> VTK -> ShockLens extraction.
# Requires a sourced OpenFOAM environment and `pip install shocklens`.
set -euo pipefail
cd "$(dirname "$0")"

( cd case && ./Allrun )                       # blockMesh, rhoCentralFoam, foamToVTK
LAST_VTK=$(ls -t case/VTK/*.vtk | head -1)
echo "Extracting from $LAST_VTK"
shocklens extract "$LAST_VTK" --nx 300 --ny 160
echo
echo "Theory for this case (Mach 5, 15 deg wedge):"
python3 -c "import shocklens as sl; print('  beta =', round(sl.synthetic.oblique_beta(5.0,15.0),2), 'deg')"
