#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/config_v33g_validate_runtime_vs_v33f.in}
mkdir -p logs
bash scripts/compile_v33g_all.sh
/usr/bin/time -v stdbuf -oL -eL bin/v33g_validate_runtime_vs_v33f "$CFG" 2>&1 | tee logs/v33g_validate_runtime_vs_v33f.log
