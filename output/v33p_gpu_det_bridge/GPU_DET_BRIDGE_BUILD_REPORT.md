# GPU determinant bridge build report

Status: `GPU_DET_BACKEND_REFERENCE_VALIDATION_PASS` for L20/100_A2.

## Build

- Executable: `bin/v33p_cpuassembled_gpu_det_scan`
- CPU source: `source/v33p_cpuassembled_gpu_det_scan.cpp`
- CUDA source: `source/v33p_gpu_det_lu.cu`
- CUDA API: `cublasZgetrfBatched` with complex-double LU; determinant is pivot parity times the product of the diagonal of U.
- Target: `sm_86` RTX 3070.
- The existing legacy/preprojected CUDA LU backend was not included.

## Validation

- Subset: 230 rows, CPU assembly exact, GPU max absolute difference `1.1892064646764559e-17`, zero sign mismatches, zero fallback.
- Full: 20,000 rows, exact Ecm/dimension/sign-change identity, accepted v3/v4 classifier identity, zero fallback.
- No cachegen, cache regeneration, cache copying, or all-sector validation was run.

## Known ceiling

The CPU assembly seam dominates runtime (`4641.6017 s` in the full run); GPU LU took `0.7333 s`. This bridge is a correctness backend, not yet a whole-scan speedup.
