#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/config_v33g_multiL_all_irreps_runtime.in}
mkdir -p logs
bash scripts/compile_v33g_all.sh
/usr/bin/time -v stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e "$CFG" spectrum-only 2>&1 | tee logs/v33g_all_irreps_runtime_count_check.log
python3 scripts/check_v33f_truezero_counts.py --summary output_v33g/fit_all_irreps_runtime/debug_v33g_all_irreps_runtime_fit_summary_allL.dat --log logs/v33g_all_irreps_runtime_count_check.log
