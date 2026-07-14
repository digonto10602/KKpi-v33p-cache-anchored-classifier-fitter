#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p bin
export EIGEN_INC=${EIGEN_INC:-/usr/include/eigen3}
${CXX:-g++} -std=c++17 -O3 -I"$EIGEN_INC" source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp -o bin/extract_F3inv_isotropic_from_gpu_cache_v33a
