#!/bin/bash
# det3d end to end: create the project if missing, preprocess + BoxC002 plan, then train. Defaults: LIDC plan 1.
# Tweak via env:  PLAN=2 ./e2e.sh   PROJECT=bones PLAN=3 FOLD=1 ./e2e.sh   PREPROC=0 CUDA_VISIBLE_DEVICES=1 ./e2e.sh
# Extra arguments go to train.py:  ./e2e.sh -e 50 --bs 2 -r my_run
set -euo pipefail

PROJECT="${PROJECT:-lidc}"
PLAN="${PLAN:-1}"
FOLD="${FOLD:-0}"
ARCH="${ARCH:-retinaunet}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
PREPROC="${PREPROC:-1}"          # 0 skips preprocessing and planning
OVERWRITE="${OVERWRITE:-0}"      # 1 passes --overwrite to preprocessing
DATASOURCES="${DATASOURCES:-}"   # only to create a missing project, e.g. "lidc"
MNEMONIC="${MNEMONIC:-}"         # only to create a missing project, e.g. "lungs"
export FRAN_CONF="${FRAN_CONF:-/s/fran_storage/conf}"

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAN_RUN="$(cd "$RUN_DIR/../../../fran/fran/run" && pwd)"
PY="${PY:-$HOME/mambaforge/envs/dl/bin/python}"

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  sed -n '2,4p' "$0" | sed 's/^# //'
  exit 0
fi

if ! "$PY" - "$PROJECT" >/dev/null <<'PY'
import sys
from fran.managers.project import Project
sys.exit(0 if Project(project_title=sys.argv[1]).db.exists() else 1)
PY
then
  if [[ -z "$DATASOURCES" || -z "$MNEMONIC" ]]; then
    echo "Project '$PROJECT' does not exist: set DATASOURCES and MNEMONIC to create it." >&2
    exit 1
  fi
  read -ra ds <<<"$DATASOURCES"
  create=("$PY" "$FRAN_RUN/project/project_init.py" -t "$PROJECT" -m "$MNEMONIC" -n "$NUM_PROCESSES" -ds "${ds[@]}")
  echo "+ ${create[*]}"
  "${create[@]}"
fi

if [[ "$PREPROC" == 1 ]]; then
  preproc=("$PY" "$FRAN_RUN/preproc/analyze_resample.py" --pipeline det -t "$PROJECT" -p "$PLAN" -n "$NUM_PROCESSES")
  [[ "$OVERWRITE" == 1 ]] && preproc+=(--overwrite)
  echo "+ ${preproc[*]}"
  "${preproc[@]}"
fi

train=("$PY" "$RUN_DIR/training/train.py" -t "$PROJECT" -p "$PLAN" -f "$FOLD" --arch "$ARCH" "$@")
echo "+ ${train[*]}"
exec "${train[@]}"
