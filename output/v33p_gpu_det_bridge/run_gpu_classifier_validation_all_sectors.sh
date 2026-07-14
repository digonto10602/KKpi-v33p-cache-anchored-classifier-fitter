#!/usr/bin/env bash
set -euo pipefail

# Existing-cache driver only. It is intentionally not run automatically.
ROOT=${GPU_CACHE_ROOT:-/media/digonto/Data/F3inv_cache}
BASE=output/v33p_gpu_det_bridge/sectors
for pair in "20 000_A1m" "20 100_A2" "20 110_A2" "20 111_A2" "20 200_A2" \
            "24 000_A1m" "24 100_A2" "24 110_A2" "24 111_A2" "24 200_A2"; do
  read -r L IRREP <<< "$pair"
  OUT="$BASE/L${L}_${IRREP}"
  mkdir -p "$OUT/gpu_det_grid"
  bin/v33p_cpuassembled_gpu_det_scan \
    --gpu-cache-root "$ROOT" --outdir "$OUT/gpu_det_grid" --Lbyas "$L" --irrep "$IRREP" \
    --Emin 0.26310 --Emax 0.36 --coarseN 20000 --xi 3.444 --threads "${OMP_NUM_THREADS:-16}" \
    --K3iso0 73735.840894011912 --K3iso1 -972421.14060757787 \
    --K3B 347174.05548116949 --K3E -1226756.7068845264 \
    --old-scaling true --complex-read-convention variant_04_real_imag_swapped \
    --hard-hermiticity-check false --gpu-batch-rows 512 --progress-every 500 \
    --cpu-fallback false --validate-subset false
  python3 scripts/run_v33p_classifier_sweep_L20_100_A2.py \
    --grid "$OUT/gpu_det_grid/gpu_det_grid_L${L}_${IRREP}_E026310_0360.csv" \
    --output-dir "$OUT/classifier_sweep"
done
