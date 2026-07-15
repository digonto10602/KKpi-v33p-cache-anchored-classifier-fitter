#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/home/digonto/.local/share/mamba/envs/minuit2-fit/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/time -v stdbuf -oL -eL \
  bin/v33f_k3df_fitter_multiL_v33e \
  output/v33p_gpu_classifier_all_irrep/fitter/fitter_after_L20_111_A2.in fcn-once
