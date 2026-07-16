#!/usr/bin/env bash
# Local det training (train.py shim prepends --pipeline det).
set -euo pipefail

source "${HOME}/mambaforge/etc/profile.d/conda.sh"
conda activate dl
export FRAN_CONF="${FRAN_CONF:-/s/fran_storage/conf}"
export PYTHONPATH="/home/ub/code/fran:/home/ub/code/det3d:/home/ub/code/label_analysis:/home/ub/code/utilz:/home/ub/code/localiser${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT="${1:-bones}"
PLAN_NUM="${2:-1}"
DEVICES="${3:-0}"
BS="${4:-4}"
EPOCHS="${5:-500}"
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
BSF="${18:-true}"
DUAL_SSD="${19:-false}"
BATCH_TFMS="${20:-true}"
SNAPSHOT_FREQ="${21:-100}"
VAL_SAMPLING="${22:-1.0}"
TAGS="${23:-}"
ARCH="${24:-retinanet}"
TRANSFER="${25:-false}"
RESUME_LR="${26:-}"

DET_TRAIN="${DET_TRAIN:-/home/ub/code/det3d/det3d/run/training/train.py}"

argv=(
  python -u "${DET_TRAIN}"
  --project "${PROJECT}"
  --plan-num "${PLAN_NUM}"
  --devices "${DEVICES}"
  --bs "${BS}"
  --fold "${FOLD}"
  --epochs "${EPOCHS}"
  --compiled "${COMPILED}"
  --profiler "${PROFILER}"
  --wandb "${WANDB}"
  --cache-rate "${CACHE_RATE}"
  --arch "${ARCH}"
  --val-sampling "${VAL_SAMPLING}"
  --bsf "${BSF}"
  --batch-tfms "${BATCH_TFMS}"
  --snapshot-freq "${SNAPSHOT_FREQ}"
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}"
  --run-through "${RUN_THROUGH}"
  --transfer "${TRANSFER}"
  --dual-ssd "${DUAL_SSD}"
)

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
