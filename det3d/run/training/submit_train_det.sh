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

modern=0
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
      selector=pipeline; modern=1
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
  exec "${HPC_CLI}/hpc_submit_poll_fetch.sh" \
    /home/ub/code/det3d/det3d/run/training/train_det.sh "$@"
fi

PROJECT="${1:-bones}"
PLAN_NUM="${2:-1}"
ARCH="${3:-}"
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
