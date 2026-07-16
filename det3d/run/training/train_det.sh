#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1
#SBATCH -t 40:0:0
#SBATCH -J det_train
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j

set -euo pipefail

THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-1}}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dl

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
MAX_RETRIES="${20:-3}"
STEP="${21:-1}"
MIN_BS="${22:-1}"
PYTHON_BIN="${23:-python}"
BATCH_TFMS="${24:-true}"
SNAPSHOT_FREQ="${25:-100}"
VAL_SAMPLING="${26:-1.0}"
TAGS="${27:-}"
ARCH="${28:-retinanet}"
TRANSFER="${29:-false}"
RESUME_LR="${30:-}"

CODE_ROOT="${HPC_CODE_ROOT:-/data/EECS-LITQ/fran_storage/code}"

cmd=(
  "$PYTHON_BIN" -u "${CODE_ROOT}/fran/fran/run/training/train_retry.py"
  --pipeline det
  --project "$PROJECT"
  --plan-num "$PLAN_NUM"
  --devices "$DEVICES"
  --bs "$BS"
  --fold "$FOLD"
  --epochs "$EPOCHS"
  --compiled "$COMPILED"
  --profiler "$PROFILER"
  --wandb "$WANDB"
  --cache-rate "$CACHE_RATE"
  --arch "$ARCH"
  --val-sampling "$VAL_SAMPLING"
  --bsf "$BSF"
  --batch-tfms "$BATCH_TFMS"
  --snapshot-freq "$SNAPSHOT_FREQ"
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS"
  --run-through "$RUN_THROUGH"
  --transfer "$TRANSFER"
  --dual-ssd "$DUAL_SSD"
  --max-retries "$MAX_RETRIES"
  --step "$STEP"
  --min-bs "$MIN_BS"
)

if [[ -n "$LR" ]]; then
  cmd+=(--learning-rate "$LR")
fi
if [[ -n "$RESUME_LR" ]]; then
  cmd+=(--resume-lr "$RESUME_LR")
fi
if [[ -n "$RUN_NAME" && "$RUN_NAME" != "none" && "$RUN_NAME" != "null" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi
if [[ -n "$DESCRIPTION" ]]; then
  cmd+=(--description "$DESCRIPTION")
fi
if [[ -n "$DS_TYPE" ]]; then
  cmd+=(--ds-type "$DS_TYPE")
fi
if [[ -n "$TRAIN_INDICES" && "$TRAIN_INDICES" != "none" && "$TRAIN_INDICES" != "null" ]]; then
  cmd+=(--train-indices "$TRAIN_INDICES")
fi
if [[ -n "$TAGS" && "$TAGS" != "none" && "$TAGS" != "null" ]]; then
  cmd+=(--tags "$TAGS")
fi

"${cmd[@]}"
