# Corrected raw-reader GPU determinant backend build report

## Status

`GPU_DET_BACKEND_REFERENCE_VALIDATION_BLOCKED`

No GPU determinant scanner was built or used for classifier validation.

## Audit result

- `nvcc` is available at `/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc`.
- The local GPU is an NVIDIA GeForce RTX 3070 with 8 GiB VRAM.
- Trusted corrected reader/scanner:
  `source/v33h_patched_gpu_cache_oldscale_det_scan.cpp`.
- Its raw cache reader uses `variant_04_real_imag_swapped`, validates v33e/v32zu
  metadata, reconstructs `F3inv` and `Vsel`, and applies:

```text
QC = F3inv + K3df
projected_QC = Vsel.adjoint() * QC * Vsel
scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6.0)
det = determinant(scaled_projected_QC)
```

However, the determinant in that trusted path is Eigen `PartialPivLU` on the
CPU. It is not a GPU determinant backend.

Existing CUDA/cuSOLVER/cuBLAS files such as
`source/gpu_check/cusolver_batched_varsize_solve_AXeqI.cu` and the F3 GPU
pipeline operate on legacy/preprojected or intermediate representations. They
do not accept the corrected raw-reader output after the trusted CPU projection
and scaling path. Connecting them would risk changing the physics path or
reusing the known-incompatible legacy representation.

## Blocker

There is no existing safe CUDA implementation that takes the corrected raw
reader's assembled, scaled projected matrices and computes only their complex
double determinants. Implementing that bridge would require a new matrix
assembly/data-transfer path and CPU-vs-GPU validation before any production
use. That is outside a safe minimal checkpoint patch.

Therefore:

- no `bin/v33p_corrected_raw_gpu_det_scan` was created;
- no cachegen or cache regeneration was run;
- no GPU determinant validation was run;
- no all-sector GPU classifier validation was started;
- no GPU output was substituted for trusted CPU output.

The trusted CPU hot-window fitter remains usable. GPU work should resume only
with a dedicated implementation that batches the corrected raw-reader output
and proves determinant/bracket/classifier identity against the CPU reference.
