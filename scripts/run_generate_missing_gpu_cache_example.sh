#!/usr/bin/env bash
set -euo pipefail
# Example/manual command for generating one missing coarse cache using bundled v32zu cachegen.
# Normal fitter runs call this automatically when auto_build_missing_coarse=1.
L=${1:-24}
IRREP=${2:-110_A2}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$ROOT/output_v32x_multiL_QC_fitter/manual_gpu_cachegen_L${L}_${IRREP}.sh"
mkdir -p "$(dirname "$TMP")"
cat > "$TMP" <<CFG
Ecm_min=0.26301
Ecm_max=0.36
coarseN=10000
xi=3.444
output_root=/media/digonto/Data/F3inv_cache
irreps=$IRREP
debug=n
use_fused_f2k2=true
validate_fused_f2k2=false
use_kinematic_precompute=true
use_stream_pipeline=true
max_batch_energies=0
gpu_memory_safety_fraction=0.85
use_concurrent_streams=true
concurrent_streams_safety_fraction=0.85
use_multi_gpu=false
use_mixed_precision_f2k2=false
CFG
V32ZU_CONFIG="$TMP" bash "$ROOT/external/v32zu_gpu_cachegen/scripts/run_v32zu_gpu_cachegen.sh" "$L"
