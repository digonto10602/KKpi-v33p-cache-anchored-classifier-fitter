#!/usr/bin/env bash
set -euo pipefail
TEMPLATE=${1:?template config required}
OUT=${2:?output config required}
ROOT=${V33E_CACHE_ROOT:?Set V33E_CACHE_ROOT to the directory containing v33e cache folders}
python3 scripts/make_v33e_cache_file_list.py --template "$TEMPLATE" --out "$OUT" --cache-root "$ROOT" --coarseN 20000 --xi 3.444
