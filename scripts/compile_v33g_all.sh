#!/usr/bin/env bash
set -euo pipefail
bash scripts/compile_v33f_all.sh

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

echo "[v33g compile] CXX=$CXX"
echo "[v33g compile] EIGEN_INC=$EIGEN_INC"
echo "[v33g compile] MINUIT2_ROOT=${MINUIT2_ROOT:-<empty>}"
echo "[v33g compile] HAS_MINUIT2=$HAS_MINUIT2"

if [[ "$HAS_MINUIT2" -eq 1 ]]; then
  "$CXX" -std=c++17 -O3 -fopenmp \
    -I"$EIGEN_INC" -I. -Isource \
    ${MINUIT2_ROOT:+-I$MINUIT2_ROOT/include -I$MINUIT2_ROOT/include/Minuit2} \
    source/v33g_build_runtime_k3basis_cache.cpp source/Faddeeva.cc \
    ${MINUIT2_ROOT:+-L$MINUIT2_ROOT/lib} -lMinuit2 -lMinuit2Math \
    -o bin/v33g_build_runtime_k3basis_cache
  "$CXX" -std=c++17 -O3 -fopenmp \
    -I"$EIGEN_INC" -I. -Isource \
    ${MINUIT2_ROOT:+-I$MINUIT2_ROOT/include -I$MINUIT2_ROOT/include/Minuit2} \
    source/v33g_validate_runtime_vs_v33f.cpp source/Faddeeva.cc \
    ${MINUIT2_ROOT:+-L$MINUIT2_ROOT/lib} -lMinuit2 -lMinuit2Math \
    -o bin/v33g_validate_runtime_vs_v33f
else
  "$CXX" -std=c++17 -O3 -fopenmp \
    -DV32F_HAS_MINUIT2=0 -DV32F_DISABLE_MINUIT -I"$EIGEN_INC" -I. -Isource \
    source/v33g_build_runtime_k3basis_cache.cpp source/Faddeeva.cc \
    -o bin/v33g_build_runtime_k3basis_cache
  "$CXX" -std=c++17 -O3 -fopenmp \
    -DV32F_HAS_MINUIT2=0 -DV32F_DISABLE_MINUIT -I"$EIGEN_INC" -I. -Isource \
    source/v33g_validate_runtime_vs_v33f.cpp source/Faddeeva.cc \
    -o bin/v33g_validate_runtime_vs_v33f
fi

echo "[v33g compile] done"
echo "  bin/v33g_build_runtime_k3basis_cache"
echo "  bin/v33g_validate_runtime_vs_v33f"
