#!/usr/bin/env bash
# Source after: conda activate dl
NNDET_ROOT="${NNDET_ROOT:-/home/ub/code/nnDetection}"
export det_data="${det_data:-/r/datasets/nndet_data}"
export det_models="${det_models:-/r/datasets/nndet_models}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export det_num_threads="${det_num_threads:-6}"
export det_verbose="${det_verbose:-1}"
export MLFLOW_ALLOW_FILE_STORE=true
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$det_models/mlruns}"
mkdir -p "$det_models/mlruns"
