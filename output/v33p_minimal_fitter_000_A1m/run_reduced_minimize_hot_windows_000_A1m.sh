#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="/home/digonto/.local/share/mamba/envs/minuit2-fit/lib:${LD_LIBRARY_PATH:-}"
LOG="output/v33p_minimal_fitter_000_A1m/reduced_minimize_hot_windows_000_A1m.log"
set +e
/usr/bin/time -v stdbuf -oL -eL \
  bin/v33f_k3df_fitter_multiL_v33e \
  output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_reduced_minimize.in fit > >(tee "$LOG") 2>&1 &
APP_PID=$!
set -e
while kill -0 "$APP_PID" 2>/dev/null; do
  if rg -q 'model_found=[0-2]/4|fallback_full_scan=[1-9]|full_scan=[1-9]|chi2=nan|chi2=inf|chi2=1e\+100|FCN-call limit exceeded' "$LOG" 2>/dev/null; then
    echo "[reduced-watchdog] abort condition detected; terminating fitter" | tee -a "$LOG"
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    exit 2
  fi
  sleep 0.2
done
wait "$APP_PID"
