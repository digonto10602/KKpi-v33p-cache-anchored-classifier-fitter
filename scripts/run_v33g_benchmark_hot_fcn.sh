#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/config_v33g_multiL_all_irreps_runtime.in}
mkdir -p logs
bash scripts/compile_v33g_all.sh
/usr/bin/time -v stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e "$CFG" fcn-once 2>&1 | tee logs/v33g_benchmark_hot_fcn.log
