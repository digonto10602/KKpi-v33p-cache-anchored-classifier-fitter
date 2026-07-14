# GPU Cache Reader Fix

## Problem

The combined raw GPU cache file stores raw `cuDoubleComplex` payloads for `F3inv` and `Vsel`. Older reader paths did not reconstruct the payload consistently, which produced:

- `F3inv` hermiticity error near `2`
- projected `QC` hermiticity error near `2`
- nonzero imaginary determinant artifacts
- unstable classifier sign logic

The issue was not the old scaling formula and not the K3df physics.

## Correct format

The target cache is:

```text
v32zu_Lbyas20_xi3p444_irrep100_A2_coarse20000_F3inv_Vsel_gpu.bin
```

The combined raw cache reader must be format-aware and reconstruct the raw payload the same way in every path that consumes it.

## Correct scaling

Old scaling stays at the matrix level:

```cpp
const double const_norm_scale = std::pow(Lbyas * xi, 6.0);
Eigen::MatrixXcd QC = F3inv + K3df;
Eigen::MatrixXcd projected_QC = Vsel.adjoint() * QC * Vsel;
Eigen::MatrixXcd scaled_projected_QC = projected_QC / const_norm_scale;
```

No determinant-level `Nproj` scaling is used.

## Evidence

- row / offset identity passed
- single-payload identity passed
- current / manual `F3inv` max abs diff = `0`
- current / manual `Vsel` max abs diff = `0`
- corrected determinant scan: `max |det_imag| = 0`
- CPU OpenMP subset validation passed

## Relevant code

- `source/qc_fitter_norm_refine_v2_multiL.cpp:627`
- `source/v33g_build_runtime_k3basis_cache.cpp:186`
- `source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp:1`
