#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p output/v33p_minimal_fitter_000_A1m
export LD_LIBRARY_PATH="/home/digonto/.local/share/mamba/envs/minuit2-fit/lib:${LD_LIBRARY_PATH:-}"
exec /usr/bin/time -v stdbuf -oL -eL \
  bin/v33f_k3df_fitter_multiL_v33e \
  output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_short_minimize.in \
  fit
