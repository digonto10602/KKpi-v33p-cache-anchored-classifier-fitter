#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG=${1:-configs/config_v32x_multiL_QC_fitter.in}
# Optional: generate missing GPU coarse caches manually before fitting by uncommenting examples below.
# bash scripts/run_generate_missing_gpu_cache_example.sh 20 000_A1m
# bash scripts/run_generate_missing_gpu_cache_example.sh 20 100_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 20 110_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 20 111_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 20 200_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 24 000_A1m
# bash scripts/run_generate_missing_gpu_cache_example.sh 24 100_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 24 110_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 24 111_A2
# bash scripts/run_generate_missing_gpu_cache_example.sh 24 200_A2
bash run_v32x_multiL_QC_fitter.sh fit "$CONFIG"
bash run_v32x_multiL_QC_fitter.sh spectrum configs/config_v32x_multiL_QC_spectrum_only.in
