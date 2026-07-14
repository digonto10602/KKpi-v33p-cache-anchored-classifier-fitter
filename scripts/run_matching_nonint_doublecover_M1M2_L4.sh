#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$ROOT/tools/nonint3body_general_doublecover_package"
CFG="$ROOT/nonint_reference_output/nonint3body_M1M1M2_L4_config.in"
echo "[nonint-match] running double-cover non-interacting tool"
echo "[nonint-match] tool=$TOOL"
echo "[nonint-match] config=$CFG"
cd "$TOOL"
bash run_nonint3body_general_doublecover.sh perf "$CFG"
