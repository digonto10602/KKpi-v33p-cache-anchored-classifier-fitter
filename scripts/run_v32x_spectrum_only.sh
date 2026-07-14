#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=${1:-configs/config_v32x_multiL_QC_spectrum_only.in}
cd "$ROOT"
bash run_v32x_multiL_QC_fitter.sh spectrum "$CONFIG"
