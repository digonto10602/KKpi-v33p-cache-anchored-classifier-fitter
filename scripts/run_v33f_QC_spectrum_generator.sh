#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/config_v33f_QC_spectrum_generator_v33e_v3.in}
mkdir -p output_v33f/runtime_configs logs
RUNTIME=output_v33f/runtime_configs/$(basename "$CONFIG" .in).runtime.in
bash scripts/prepare_v33f_runtime_config.sh "$CONFIG" "$RUNTIME"
bash scripts/compile_v33f_all.sh
stdbuf -oL -eL bin/v33f_QC_spectrum_generator "$RUNTIME" spectrum-only 2>&1 | tee logs/v33f_QC_spectrum_generator.log
