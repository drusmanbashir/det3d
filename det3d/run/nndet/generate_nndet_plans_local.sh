#!/usr/bin/env bash
# Generate nnDet D3V001_3d.pkl for every local MSD source that is on disk.
# Skips mnemonics that already have a non-placeholder plan unless --force.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/generate_nndet_plan_silo.sh" all "${1:-}"
