#!/usr/bin/env bash
set -euo pipefail
cd /home/ub/code/det3d
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dl
exec jupyter lab /home/ub/code/det3d/det3d/detection/luna16_transform_journey.ipynb
