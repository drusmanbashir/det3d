#!/usr/bin/env bash
# Submit det training on HPC (bones defaults; override via positional args to train_det.sh).
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" == --patch-size || "$arg" == --patch-size=* ]]; then
    echo "Training removed --patch-size. Set patch dimensions in the workbook." >&2
    exit 2
  fi
done

HPC_CLI="${HPC_CLI_ROOT:-/home/ub/code/agent/agent/hpc/cli}"
export PYTHONPATH="/home/ub/code/agent:/home/ub/code/agent/agent/hpc${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT="${1:-bones}"
PLAN_NUM="${2:-1}"
ARCH="${3:-retinanet}"
EPOCHS="${4:-}"
BS="${5:-}"

argv=(
  "${HPC_CLI}/hpc_submit_poll_fetch.sh"
  /home/ub/code/det3d/det3d/run/training/train_det.sh
  "${PROJECT}"
  "${PLAN_NUM}"
  0
  "${BS}"
  "${EPOCHS}"
  0
  false
  false
  true
  0.0
  ""
  none
  ""
  ""
  true
  1
  ""
  false
  false
  3
  1
  1
  python
  true
  100
  1.0
  ""
  "${ARCH}"
  false
  ""
)

exec "${argv[@]}"
