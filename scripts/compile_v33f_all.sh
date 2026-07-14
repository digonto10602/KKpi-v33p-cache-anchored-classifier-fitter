#!/usr/bin/env bash
set -euo pipefail
mkdir -p bin logs
THREADS=${OMP_NUM_THREADS:-18}
export OMP_NUM_THREADS="$THREADS"
export EIGEN_INC=${EIGEN_INC:-/usr/include/eigen3}
export MINUIT2_ROOT=${MINUIT2_ROOT:-${CONDA_PREFIX:-}}
CXX=${CXX:-}
if [[ -z "$CXX" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]]; then
    CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
  else
    CXX=g++
  fi
fi
if [[ -n "$MINUIT2_ROOT" ]]; then
  export LD_LIBRARY_PATH="$MINUIT2_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

HAS_MINUIT2=0
if [[ -e "${MINUIT2_ROOT:-}/include/Minuit2/Minuit2/FCNBase.h" && -e "${MINUIT2_ROOT:-}/lib/libMinuit2.so" && -e "${MINUIT2_ROOT:-}/lib/libMinuit2Math.so" ]]; then
  HAS_MINUIT2=1
elif [[ -e /usr/include/Minuit2/FCNBase.h ]] && ldconfig -p 2>/dev/null | grep -q 'libMinuit2\.so' && ldconfig -p 2>/dev/null | grep -q 'libMinuit2Math\.so'; then
  HAS_MINUIT2=1
fi

echo "[v33f compile] CXX=$CXX"
echo "[v33f compile] EIGEN_INC=$EIGEN_INC"
echo "[v33f compile] MINUIT2_ROOT=${MINUIT2_ROOT:-<empty>}"
echo "[v33f compile] HAS_MINUIT2=$HAS_MINUIT2"

"$CXX" -std=c++17 -O3 -fopenmp -I"$EIGEN_INC" -I. -Isource \
  source/v33f_F3inv_iso_visual_test.cpp \
  -o bin/v33f_F3inv_iso_visual_test

if [[ "$HAS_MINUIT2" -eq 1 ]]; then
  "$CXX" -std=c++17 -O3 -fopenmp \
    -I"$EIGEN_INC" -I. -Isource \
    ${MINUIT2_ROOT:+-I$MINUIT2_ROOT/include -I$MINUIT2_ROOT/include/Minuit2} \
    source/qc_fitter_norm_refine_v2_multiL.cpp source/Faddeeva.cc \
    ${MINUIT2_ROOT:+-L$MINUIT2_ROOT/lib} -lMinuit2 -lMinuit2Math \
    -o bin/v33f_k3df_fitter_multiL_v33e
else
  echo "[v33f compile] Minuit2 not found; building spectrum-only binary without fit support"
  "$CXX" -std=c++17 -O3 -fopenmp \
    -DV32F_HAS_MINUIT2=0 -DV32F_DISABLE_MINUIT -I"$EIGEN_INC" -I. -Isource \
    source/qc_fitter_norm_refine_v2_multiL.cpp source/Faddeeva.cc \
    -o bin/v33f_k3df_fitter_multiL_v33e
fi

# Same executable as above, named for the fixed-parameter spectrum-generator workflow.
cp bin/v33f_k3df_fitter_multiL_v33e bin/v33f_QC_spectrum_generator

echo "[v33f compile] done"
echo "  bin/v33f_F3inv_iso_visual_test"
echo "  bin/v33f_k3df_fitter_multiL_v33e"
echo "  bin/v33f_QC_spectrum_generator"
