#!/usr/bin/env bash
set -euo pipefail
mkdir -p output_v33f/runtime_configs logs
RUNTIME=output_v33f/runtime_configs/config_v33f_multiL_all_irreps_v33e_v3.runtime.in
bash scripts/prepare_v33f_runtime_config.sh configs/config_v33f_multiL_all_irreps_v33e_v3.in "$RUNTIME"
bash scripts/compile_v33f_all.sh
# spectrum-only calls model_for once and is the cleanest 1-FCN benchmark for classifier/projection/determinant scan.
/usr/bin/time -v stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e "$RUNTIME" spectrum-only 2>&1 | tee logs/v33f_benchmark_1fcn_spectrum_only.log
