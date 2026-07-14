#!/usr/bin/env bash
set -euo pipefail
mkdir -p output_v33f/runtime_configs logs
RUNTIME=output_v33f/runtime_configs/config_v33f_multiL_000A1m_5levels_v33e_v3.runtime.in
bash scripts/prepare_v33f_runtime_config.sh configs/config_v33f_multiL_000A1m_5levels_v33e_v3.in "$RUNTIME"
bash scripts/compile_v33f_all.sh
/usr/bin/time -v stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e "$RUNTIME" spectrum-only 2>&1 | tee logs/v33f_smoke_000A1m_5levels.log
python3 scripts/check_v33f_truezero_counts.py --summary output_v33f/fit_000A1m_5levels/debug_v33f_000A1m_5levels_fit_summary_allL.dat --log logs/v33f_smoke_000A1m_5levels.log
