#!/usr/bin/env bash
# Binarize LBD lms (0=bg, 1=lesion) for a det project plan.
set -euo pipefail

PROJECT="${1:-bones}"
PLAN_ID="${2:-1}"
LBD_FOLDER="${3:-}"

source "${HOME}/mambaforge/etc/profile.d/conda.sh"
conda activate dl
export FRAN_CONF="${FRAN_CONF:-/s/fran_storage/conf}"

argv=(
  python /home/ub/code/det3d/det3d/run/preprocessing/binarize_lbd_lms.py
  --project "${PROJECT}"
  --plan-id "${PLAN_ID}"
)
if [[ -n "${LBD_FOLDER}" ]]; then
  argv+=(--lbd-folder "${LBD_FOLDER}")
fi

exec "${argv[@]}"
