#!/usr/bin/env bash
# Hybrid fast LBD benchmark profiles (canonical: det3d/extra/hybrid.py)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROFILE="${1:-smoke}"
GPU="${GPU:-1}"

common=(
  python -u run/training/train_hybrid_fast_lbd.py
  --project lidca
  --plan 4
  --batch-size 4
  --train-mode overwrite
  --gpu "$GPU"
  --wandb true
)

case "$PROFILE" in
  smoke)
    "${common[@]}" \
      --n-train 8 \
      --n-val 4 \
      --epochs 2 \
      --batches-per-epoch 4 \
      --val-batches-per-epoch 2 \
      --val-every-n-epochs 1 \
      --wandb-grid-epoch-freq 1 \
      --permanent-checkpoint-every-n-epochs 99 \
      --exp-id "LIDCA-HYBRID-FAST-LBD-SMOKE" \
      --run-name "LIDCA-HYBRID-FAST-LBD-SMOKE"
    ;;
  n25-e200)
    "${common[@]}" \
      --n-train 25 \
      --epochs 200 \
      --exp-id "LIDCA-HYBRID-FAST-LBD-N25-E200" \
      --run-name "LIDCA-HYBRID-FAST-LBD-N25-E200"
    ;;
  *)
    echo "usage: $0 {smoke|n25-e200}" >&2
    exit 1
    ;;
esac
