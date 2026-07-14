# GPU determinant backend audit

## Result

`GPU_BACKEND_STATUS = EXISTING_BACKEND_FOUND_BUT_NOT_SAFE_FOR_THIS_CHECKPOINT`

Existing CUDA sources include:

- `source/F3_device_resident_speed_step_v11_v4_cusolver_lu.cu`
- `source/gpu_solvers_batched_streams.cpp`
- historical v32x `GpuBatchedDetBackend` using cuBLAS batched LU

These paths are useful implementation references, but their inputs are
preprojected v32x/runtime rows or legacy fitter structures. They do not load
the corrected combined raw `F3inv_Vsel_gpu.bin` format and do not provide the
required current CSV scan contract.

## Safety decision

The old backend is not used for production classifier or fitter validation.
The trusted CPU/OpenMP v33h scanner remains the reference. No cachegen or GPU
cache regeneration was run.

## Required next patch

Integrate grouped dimensions and double-complex batched LU behind the corrected
raw reader, preserve the exact QC/projection/scaling formulas, write an atomic
temporary CSV, and compare every row/bracket against the CPU reference before
enabling it for fast-track scans.
