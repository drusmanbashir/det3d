#!/bin/bash
# det3d end to end: create the project if missing, preprocess + BoxC002 plan, then train. Defaults: LIDC plan 1.
# Tweak via env:  PLAN=2 ./e2e.sh   PROJECT=bones PLAN=3 FOLD=1 ./e2e.sh   PREPROC=0 CUDA_VISIBLE_DEVICES=1 ./e2e.sh
# Modern selectors reach both stages:  ./e2e.sh --project lidc --plan sota --pipeline det --plan-file plan.yaml
# Other arguments go to train.py:  ./e2e.sh -e 50 --bs 2 -r my_run
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  sed -n '2,5p' "$0" | sed 's/^# //'
  exit 0
fi

for arg in "$@"; do
  if [[ "$arg" == --patch-size || "$arg" == --patch-size=* ]]; then
    echo "Training removed --patch-size. Set patch dimensions in the workbook." >&2
    exit 2
  fi
done

PROJECT="${PROJECT:-lidc}"
PLAN="${PLAN:-1}"
FOLD="${FOLD:-0}"
ARCH="${ARCH:-}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
PREPROC="${PREPROC:-1}"          # 0 skips preprocessing and planning
OVERWRITE="${OVERWRITE:-0}"      # 1 passes --overwrite to preprocessing
DATASOURCES="${DATASOURCES:-}"   # only to create a missing project, e.g. "lidc"
MNEMONIC="${MNEMONIC:-}"         # only to create a missing project, e.g. "lungs"
export FRAN_CONF="${FRAN_CONF:-/s/fran_storage/conf}"

TRAIN_ARGS=()
PLAN_FILE="${PLAN_FILE:-}"
MODERN_SELECTORS=0
[[ -n "$PLAN_FILE" ]] && MODERN_SELECTORS=1
declare -A selector_values=()
while (($#)); do
  key="$1"
  value=""
  case "$key" in
    --project|-t|--project-title|--plan|-p|--plan-num|--pipeline|--plan-file)
      if (($# < 2)); then echo "$key requires a value" >&2; exit 2; fi
      value="$2"
      shift 2
      ;;
    --project=*|--project-title=*|--plan=*|--plan-num=*|--pipeline=*|--plan-file=*)
      value="${key#*=}"
      key="${key%%=*}"
      shift
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      continue
      ;;
  esac
  case "$key" in
    --project|-t|--project-title) selector=project; PROJECT="$value" ;;
    --plan|-p|--plan-num) selector=plan; PLAN="$value" ;;
    --pipeline)
      selector=pipeline
      if [[ "$value" != det ]]; then echo "det3d e2e requires --pipeline det" >&2; exit 2; fi
      ;;
    --plan-file) selector=plan_file; PLAN_FILE="$value" ;;
  esac
  MODERN_SELECTORS=1
  if [[ -n "${selector_values[$selector]:-}" && "${selector_values[$selector]}" != "$value" ]]; then
    echo "Conflicting $selector selectors: '${selector_values[$selector]}' and '$value'" >&2
    exit 2
  fi
  selector_values[$selector]="$value"
done

if [[ -n "$PLAN_FILE" ]]; then
  for arg in "${TRAIN_ARGS[@]}"; do
    if [[ "$arg" == --arch || "$arg" == --arch=* ]]; then
      echo "--arch cannot be combined with --plan-file; keep architecture in the workbook plan." >&2
      exit 2
    fi
  done
fi

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAN_RUN="$(cd "$RUN_DIR/../../../fran/fran/run" && pwd)"
PY="${PY:-$HOME/mambaforge/envs/dl/bin/python}"

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
  [[ -n "$PLAN_FILE" ]] && preproc+=(--plan-file "$PLAN_FILE")
  [[ "$OVERWRITE" == 1 ]] && preproc+=(--overwrite)
  echo "+ ${preproc[*]}"
  "${preproc[@]}"
fi

if [[ "$MODERN_SELECTORS" == 1 ]]; then
  train=("$PY" "$RUN_DIR/training/train.py" --pipeline det --project "$PROJECT" --plan "$PLAN")
  [[ -n "$PLAN_FILE" ]] && train+=(--plan-file "$PLAN_FILE")
  train+=("${TRAIN_ARGS[@]}")
else
  train=("$PY" "$RUN_DIR/training/train.py" -t "$PROJECT" -p "$PLAN" -f "$FOLD")
  [[ -n "$ARCH" ]] && train+=(--arch "$ARCH")
  train+=("${TRAIN_ARGS[@]}")
fi
echo "+ ${train[*]}"
exec "${train[@]}"
