#!/usr/bin/env bash
# Ephemeral nnDet prep: stage local data → D3V001_3d.pkl → nndet_conf/plans/{mnemonic}/, delete silo.
# No downloads. See local_plan_sources.yaml and discover_local_plan_sources.py.
set -euo pipefail

MNEMONIC="${1:?usage: generate_nndet_plan_silo.sh liver|kidneys|pancreas|colon|all [--force]}"
FORCE="${2:-}"

FRAN_CONF="${FRAN_CONF:-/s/fran_storage/conf}"
NNDET_ROOT="${NNDET_ROOT:-/home/ub/code/nnDetection}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES_YAML="${SCRIPT_DIR}/local_plan_sources.yaml"
STAGE_PY="${SCRIPT_DIR}/stage_local_msd.py"
PLACEHOLDER_SIZE=789064

source /home/ub/mambaforge/etc/profile.d/conda.sh
conda activate dl
NNDET_CONF="$(python -c "import yaml; print(yaml.safe_load(open('${FRAN_CONF}/config.yaml'))['nndet_conf'])")"
export FRAN_CONF
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export det_num_threads="${det_num_threads:-6}"
export MPLBACKEND=Agg

log() { echo "[nndet_silo $1] $*" >&2; }

run_one() {
  local mnemonic="$1"
  local force="${2:-}"
  local silo_dir="/s/agent_rw/tmp/nndet_silo_${mnemonic}_$$"
  local out_pkl="${NNDET_CONF}/plans/${mnemonic}/D3V001_3d.pkl"

  if [[ -f "${out_pkl}" ]] && [[ "${force}" != "--force" ]]; then
    local sz
    sz="$(stat -c%s "${out_pkl}")"
    if [[ "${sz}" != "${PLACEHOLDER_SIZE}" ]]; then
      log "${mnemonic}" "skip ${out_pkl} (${sz} bytes); use --force to regenerate"
      return 0
    fi
  fi

  local task layout prepare_project prepare_args
  read -r task layout prepare_project prepare_args < <(
    python - <<PY
import yaml
from pathlib import Path
spec = yaml.safe_load(Path("${SOURCES_YAML}").read_text())["${mnemonic}"]
print(spec["task"], spec["layout"], spec.get("prepare_project", ""), spec.get("prepare_args", ""))
PY
  )

  export det_data="${silo_dir}/det_data"
  export det_models="${silo_dir}/det_models"
  mkdir -p "${det_data}" "${det_models}/mlruns" "$(dirname "${out_pkl}")"

  cleanup() {
    if [[ -d "${silo_dir}" ]]; then
      log "${mnemonic}" "delete ephemeral silo ${silo_dir}"
      rm -rf "${silo_dir}"
    fi
  }
  trap cleanup EXIT INT TERM

  log "${mnemonic}" "stage ${task} (${layout})"
  python "${SCRIPT_DIR}/discover_local_plan_sources.py" --sources-yaml "${SOURCES_YAML}" \
    | rg "^${mnemonic} " || true

  local task_dir="${det_data}/${task}"
  python "${STAGE_PY}" \
    --mnemonic "${mnemonic}" \
    --task-dir "${task_dir}" \
    --sources-yaml "${SOURCES_YAML}" >&2

  if [[ "${layout}" == "decathlon" ]]; then
    log "${mnemonic}" "prepare ${prepare_args} via ${prepare_project}"
    cd "${NNDET_ROOT}/projects/${prepare_project}/scripts"
    python prepare.py ${prepare_args} >&2
  elif [[ "${layout}" == "kits_splitted" ]]; then
    log "${mnemonic}" "using local raw_splitted (all cases; skip Task011 210-case prepare cap)"
  else
    log "${mnemonic}" "unsupported layout: ${layout}" >&2
    return 1
  fi

  log "${mnemonic}" "convert_seg2det ${task}"
  cd "${NNDET_ROOT}"
  python "${SCRIPT_DIR}/convert_seg2det_local.py" "${task}" --num_processes "${det_num_threads}"

  log "${mnemonic}" "nndet_prep ${task}"
  nndet_prep "${task}" -np "${det_num_threads}" -npp "${det_num_threads}"

  local src_pkl="${det_data}/${task}/preprocessed/D3V001_3d.pkl"
  if [[ ! -f "${src_pkl}" ]]; then
    log "${mnemonic}" "missing plan pickle: ${src_pkl}" >&2
    return 1
  fi

  cp "${src_pkl}" "${out_pkl}"
  log "${mnemonic}" "installed ${out_pkl} ($(du -h "${out_pkl}" | cut -f1))"
  cleanup
  trap - EXIT INT TERM
}

if [[ "${MNEMONIC}" == "all" ]]; then
  mapfile -t MNEMONICS < <(
    python -c "import yaml; from pathlib import Path; print('\n'.join(yaml.safe_load(Path('${SOURCES_YAML}').read_text()).keys()))"
  )
  for m in "${MNEMONICS[@]}"; do
    run_one "${m}" "${FORCE}"
  done
else
  run_one "${MNEMONIC}" "${FORCE}"
fi

echo "[nndet_silo] done" >&2
