#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/compile_extract_F3inv_isotropic_v33a.sh
XI=${XI:-3.444}
COARSEN=${COARSEN:-20000}
CACHE_ROOT=${CACHE_ROOT:-/media/digonto/Data/F3inv_cache}
OUTDIR=${OUTDIR:-output_v33a_F3inv_isotropic}
mkdir -p "$OUTDIR"
for L in 20 24; do
  for IR in 000_A1m 100_A2 110_A2 111_A2 200_A2; do
    CACHE="$CACHE_ROOT/v32zu_Lbyas${L}_gpu_cache/v32zu_Lbyas${L}_xi3p444_irrep${IR}_coarse${COARSEN}_F3inv_Vsel_gpu.bin"
    if [[ ! -f "$CACHE" ]]; then
      echo "[extract-all-warning] missing $CACHE"
      continue
    fi
    OUT="$OUTDIR/F3inv_isotropic_Lbyas${L}_${IR}_coarse${COARSEN}.dat"
    ./bin/extract_F3inv_isotropic_from_gpu_cache_v33a "$CACHE" "$OUT" "$L" "$XI" "$IR"
  done
done
python3 scripts/plot_F3inv_isotropic_nscale_v33a.py "$OUTDIR"/*.dat --nscale 100 --ylim -1000000 1000000
