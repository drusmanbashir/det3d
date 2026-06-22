#!/usr/bin/env bash
set -euo pipefail
# Native nnDetection LIDC training in conda env `dl`.
# Usage: ./run/training/nndet_train_lidc.sh [-o exp.fold=0] [--sweep]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/nndet_env_dl.sh"
exec nndet_train Task012_LIDC "$@"
