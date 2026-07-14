#!/usr/bin/env bash
set -euo pipefail
CFG=${1:?runtime build config required}
mkdir -p logs
bash scripts/compile_v33g_all.sh
/usr/bin/time -v stdbuf -oL -eL bin/v33g_build_runtime_k3basis_cache "$CFG" 2>&1 | tee logs/v33g_build_runtime_cache_all.log
