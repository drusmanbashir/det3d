#!/usr/bin/env bash
# Local det training (train.py shim prepends --pipeline det).
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" == --patch-size || "$arg" == --patch-size=* ]]; then
    echo "Training removed --patch-size. Set patch dimensions in the workbook." >&2
    exit 2
  fi
done

source "${HOME}/mambaforge/etc/profile.d/conda.sh"
conda activate dl
export PYTHONPATH="/home/ub/code/fran:/home/ub/code/det3d:/home/ub/code/label_analysis:/home/ub/code/utilz:/home/ub/code/localiser${PYTHONPATH:+:${PYTHONPATH}}"

modern=0
has_pipeline=0
declare -A selectors=()
args=("$@")
while ((${#args[@]})); do
  key="${args[0]}"
  value=""
  case "$key" in
    --project|-t|--project-title|--plan|-p|--plan-num|--pipeline|--plan-file)
      ((${#args[@]} >= 2)) || { echo "$key requires a value" >&2; exit 2; }
      value="${args[1]}"
      args=("${args[@]:2}")
      ;;
    --project=*|--project-title=*|--plan=*|--plan-num=*|--pipeline=*|--plan-file=*)
      value="${key#*=}"
      key="${key%%=*}"
      args=("${args[@]:1}")
      ;;
    *) args=("${args[@]:1}"); continue ;;
  esac
  case "$key" in
    --project|-t|--project-title) selector=project; modern=1 ;;
    --plan|-p|--plan-num) selector=plan; modern=1 ;;
    --plan-file) selector=plan_file; modern=1 ;;
    --pipeline)
      selector=pipeline; modern=1; has_pipeline=1
      [[ "$value" == det ]] || { echo "det3d launcher requires --pipeline det" >&2; exit 2; }
      ;;
  esac
  if [[ -n "${selectors[$selector]:-}" && "${selectors[$selector]}" != "$value" ]]; then
    echo "Conflicting $selector selectors: '${selectors[$selector]}' and '$value'" >&2
    exit 2
  fi
  selectors[$selector]="$value"
done

if [[ "$modern" == 1 ]]; then
  modern_argv=(python -u "${DET_TRAIN:-/home/ub/code/det3d/det3d/run/training/train.py}")
  [[ "$has_pipeline" == 1 ]] || modern_argv+=(--pipeline det)
  modern_argv+=("$@")
  exec "${modern_argv[@]}"
fi

PROJECT="${1:-bones}"
PLAN_NUM="${2:-1}"
DEVICES="${3:-0}"
BS="${4:-}"
EPOCHS="${5:-}"
FOLD="${6:-0}"
COMPILED="${7:-false}"
PROFILER="${8:-false}"
WANDB="${9:-true}"
CACHE_RATE="${10:-0.0}"
LR="${11:-}"
RUN_NAME="${12:-}"
DESCRIPTION="${13:-}"
DS_TYPE="${14:-}"
RUN_THROUGH="${15:-true}"
VAL_EVERY_N_EPOCHS="${16:-1}"
TRAIN_INDICES="${17:-}"
BSF="${18:-false}"
DUAL_SSD="${19:-false}"
BATCH_TFMS="${20:-true}"
SNAPSHOT_FREQ="${21:-100}"
VAL_SAMPLING="${22:-1.0}"
TAGS="${23:-}"
ARCH="${24:-}"
TRANSFER="${25:-false}"
RESUME_LR="${26:-}"

DET_TRAIN="${DET_TRAIN:-/home/ub/code/det3d/det3d/run/training/train.py}"

argv=(
  python -u "${DET_TRAIN}"
  --project "${PROJECT}"
  --plan-num "${PLAN_NUM}"
  --devices "${DEVICES}"
  --fold "${FOLD}"
  --compiled "${COMPILED}"
  --profiler "${PROFILER}"
  --wandb "${WANDB}"
  --cache-rate "${CACHE_RATE}"
  --val-sampling "${VAL_SAMPLING}"
  --bsf "${BSF}"
  --batch-tfms "${BATCH_TFMS}"
  --snapshot-freq "${SNAPSHOT_FREQ}"
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}"
  --run-through "${RUN_THROUGH}"
  --transfer "${TRANSFER}"
  --dual-ssd "${DUAL_SSD}"
)

if [[ -n "${ARCH}" ]]; then
  argv+=(--arch "${ARCH}")
fi
if [[ -n "${BS}" ]]; then
  argv+=(--bs "${BS}")
fi
if [[ -n "${EPOCHS}" ]]; then
  argv+=(--epochs "${EPOCHS}")
fi
if [[ -n "${LR}" ]]; then
  argv+=(--learning-rate "${LR}")
fi
if [[ -n "${RESUME_LR}" ]]; then
  argv+=(--resume-lr "${RESUME_LR}")
fi
if [[ -n "${RUN_NAME}" && "${RUN_NAME}" != "none" && "${RUN_NAME}" != "null" ]]; then
  argv+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${DESCRIPTION}" ]]; then
  argv+=(--description "${DESCRIPTION}")
fi
if [[ -n "${DS_TYPE}" ]]; then
  argv+=(--ds-type "${DS_TYPE}")
fi
if [[ -n "${TRAIN_INDICES}" && "${TRAIN_INDICES}" != "none" && "${TRAIN_INDICES}" != "null" ]]; then
  argv+=(--train-indices "${TRAIN_INDICES}")
fi
if [[ -n "${TAGS}" && "${TAGS}" != "none" && "${TAGS}" != "null" ]]; then
  argv+=(--tags "${TAGS}")
fi

exec "${argv[@]}"
