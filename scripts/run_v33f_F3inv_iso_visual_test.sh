#!/usr/bin/env bash
set -euo pipefail
ROOT=${V33E_CACHE_ROOT:?Set V33E_CACHE_ROOT to the directory containing the v33e_coarse20000_sep_f2k2_* folders}
CONFIG=${1:-configs/config_v33f_F3inv_iso_visual_test.in}
RUNTIME=output_v33f/runtime_configs/config_v33f_F3inv_iso_visual_test.runtime.in
mkdir -p output_v33f/runtime_configs logs
python3 - <<PY
from pathlib import Path
src=Path('$CONFIG')
out=Path('$RUNTIME')
text=src.read_text()
lines=[]
for line in text.splitlines():
    if line.strip().startswith('v33e_cache_root'):
        lines.append('v33e_cache_root = $ROOT')
    else:
        lines.append(line)
out.write_text('\n'.join(lines)+'\n')
print('[v33f-runtime-config] wrote', out)
PY
bash scripts/compile_v33f_all.sh
stdbuf -oL -eL bin/v33f_F3inv_iso_visual_test "$RUNTIME" 2>&1 | tee logs/v33f_F3inv_iso_visual_test.log
python3 scripts/plot_F3inv_iso_nscale_with_v3_zeros.py --input-dir output_v33f/F3inv_iso_visual_test --outdir plots/v33f_F3inv_iso_nscale_v3_truezeros --nscale 100 --ylim -1000000 1000000
