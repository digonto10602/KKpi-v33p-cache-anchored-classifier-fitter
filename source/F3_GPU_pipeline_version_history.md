# F3 GPU Pipeline Version History

This file summarizes the versions developed for the GPU accelerated three-body `F3` / projected determinant pipeline and the major changes in each version.

---

## v9: Inward-shape adaptive zero refinement

**Main file**

- `F3_gpu_omp_cublas_pipeline_v9_inward_shape_refinement.cu`

**Purpose**

- Replace the older threshold-based zero/pole classifier with an inward-shape refinement method.

**Main changes**

- Added adaptive refinement around coarse sign flips.
- Coarse scan only finds candidate sign-flip windows.
- Refined scan uses a local window, for example `i-5` to `i+6`.
- Refined mesh is placed inside the local energy window.
- Classification uses inward gap behavior:
  - If `|f(left)| + |f(right)|` decreases inward, classify as `likely_zero`.
  - If it increases inward, classify as `likely_pole`.
- Removed dependence on:
  - `small_y_threshold`
  - `spike_ratio_threshold`
  - `slope_ratio_threshold`
- Saves only refined candidates classified as `likely_zero`.
- No coarse fallback zeros are saved.

**Limitations**

- Matrix ingredients were still built mostly by CPU/OpenMP.
- GPU was used mainly for linear algebra / inversion / solve steps.

---

## v9 optional HDF5 patch

**Main file**

- `F3_gpu_omp_cublas_pipeline_v9_inward_shape_refinement_optional_hdf5.cu`

**Purpose**

- Avoid requiring HDF5 headers unless matrix saving is explicitly enabled.

**Main changes**

- Made HDF5 matrix saving optional.
- Added support for compiling without:

```cpp
#include <H5Cpp.h>
```

unless:

```bash
-DF3_ENABLE_HDF5_MATRIX_SAVE
```

is provided.

**Compile behavior**

- Without HDF5:

```bash
./build.sh perf
```

- With HDF5:

```bash
./build.sh perf hdf5
```

---

## v10: VRAM-aware hybrid CPU/GPU pipeline

**Main file**

- `F3_gpu_omp_cublas_pipeline_v10_vram_aware_hybrid_pipeline.cu`

**Purpose**

- Make the pipeline VRAM-aware and safer for large energy sweeps.

**Main changes**

- Checks available GPU VRAM.
- Splits matrix chunks to avoid out-of-memory errors.
- Keeps CPU/OpenMP matrix construction:
  - `config_maker_4`
  - `F2`
  - `G`
  - `K2inv`
  - `Vsel`
- Sends matrices to GPU for:
  - `H X1 = F2`
  - `F3 = F2/3 - F2 X1`
  - `F3 X2 = Vsel`
  - `Vsel.adjoint() * X2`
  - determinant of projected `F3^{-1}`
- Keeps inward-shape adaptive zero refinement from v9.
- Optional HDF5 matrix saving retained.

**Limitations**

- Matrix ingredients were still generated on CPU/OpenMP.
- Matrices were copied CPU to GPU for the F3 solve.

---

## v11: VRAM-aware staged GPU-builder version

**Main file**

- `F3_gpu_omp_cublas_pipeline_v11_vram_aware_staged_gpu_builder.cu`

**Purpose**

- Prepare the code structure for eventual GPU-side matrix builders.

**Main changes**

- Added explicit backend enum:

```cpp
enum class F3MatrixBuilderBackend
{
    CpuOpenmpReference = 0,
    FullGpuExperimentalNotAvailable = 1
};
```

- Made clear that full GPU-side ingredient construction was not yet active.
- Still used CPU/OpenMP for:
  - config construction
  - `F2`
  - `G`
  - `K2inv`
  - `Vsel`
- Used GPU for the solve/projection/determinant stage.
- Kept VRAM-aware chunking and inward-shape refinement.

**Limitations**

- No true GPU-side `F2/G/K2/Vsel` construction yet.

---

# GPU-safe ingredient builders

These were developed as companion headers and tests before constructing the full GPU-resident F3 pipeline.

---

## GPU-safe `config_maker_4`

**Main file**

- `functions_gpu_config_maker4.cuh`

**Purpose**

- Make `config_maker_4` GPU-safe for many energy points.

**Main changes**

- Rewrote config selection logic using flat arrays.
- GPU marks accepted candidates over:
  - energy index
  - `lm`
  - momentum shell indices
- CPU compacts marked candidates to preserve original deterministic ordering.
- Supports many energies at once.

**Important detail**

- Ordering is preserved by CPU-side compaction.
- This avoids atomic scrambling of basis order.

**Test file**

- `test_gpu_config_maker4_two_flavor.cu`

**Compile script**

- `compile_test_gpu_config_maker4_two_flavor.sh`

---

## GPU-safe `F2`

**Main file**

- `F2_gpu_safe_builder.cuh`

**Purpose**

- Build `F2_2plus1_mat` on the GPU.

**Main changes**

- Builds full block-diagonal two-flavor `F2`:

```text
F2 = [ F2_1   0  ]
     [  0    F2_2]
```

- One CUDA thread per matrix element.
- Flattens `plm_config` and `klm_config`.
- Ports device-side versions of needed scalar functions.
- Uses device-side real spherical harmonics up to the supported `ell`.

**Important caveat**

- CPU `F2` uses Faddeeva / `erfi` routines.
- GPU version uses a CUDA-safe approximation for the relevant special-function pieces.
- Must be validated against CPU `F2`.

**Test file**

- `test_F2_gpu_safe_builder.cu`

**Compile script**

- `compile_test_F2_gpu_safe_builder.sh`

---

## GPU-safe `G`

**Main file**

- `G_gpu_safe_builder.cuh`

**Purpose**

- Build `G_2plus1_mat` on the GPU.

**Main changes**

- Rewrites `G_2plus1_mat` structure:

```text
G = [ G11  G12 ]
    [ G21   0  ]
```

- Computes:
  - `G11`
  - `G12`
  - `G21`
  - zero `G22`
- One CUDA thread per matrix element.
- Includes the `sqrt(2)` and parity factors for off-diagonal flavor blocks.
- Ports `G_ij_lm` logic to GPU-safe form.

**Test file**

- `test_G_gpu_safe_builder.cu`

**Compile script**

- `compile_test_G_gpu_safe_builder.sh`

---

## GPU-safe `K2inv`

**Main files**

- `K2_gpu_safe_builder.cuh`
- `K2_functions_gpu_safe.cuh`

**Purpose**

- Build `K2inv_EREord2_2plus1_mat` on the GPU.

**Main changes**

- Rewrites scalar K2 functions as CUDA-safe routines.
- Builds two-flavor block-diagonal matrix:

```text
K2inv = [ K2inv_1      0      ]
        [   0      2*K2inv_2 ]
```

- Implements GPU-safe versions of:
  - `K2_inv_ERE_ang_mom`
  - `q2psq_star`
  - `cutoff_function_1`
  - `omega_func`
  - relevant scalar kinematic functions
- Avoids depending on `F2_functions_v2.h` inside the GPU K2 path.

**Important fix**

- Earlier compile failed because `K2_functions_v2.h` included `F2_functions_v2.h`, which required `Faddeeva::erfi`.
- The self-contained GPU K2 header avoids that dependency.

**Test file**

- `test_K2_functions_gpu_safe.cu`

**Compile script**

- `compile_test_K2_functions_gpu_safe.sh`

---

## GPU-safe projections / `Vsel`

**Main file**

- `projections_gpu_safe.cuh`

**Purpose**

- Build `P_irrep_projection_2plus1` and `Vsel` on GPU.

**Main changes**

- Builds full two-flavor projection matrix on GPU.
- Uses one CUDA thread per projector matrix element.
- Uses cuSOLVER eigen-decomposition to get eigenvectors with eigenvalues near one.
- Constructs:
  - `P_I`
  - `Vsel`
  - `Pproj = Vsel * Vsel.adjoint()`

**Important fix**

- `DeviceProjectionTables` owns CUDA memory and needed move constructors.
- Fixed in:

```text
projections_gpu_safe_v2_fixed_move.cuh
```

**Test file**

- `test_projections_gpu_safe.cu`

**Compile script**

- `compile_test_projections_gpu_safe.sh`

---

# Full F3 GPU pipeline versions

---

## v10 GPU matrix builder adaptive zeros

**Main file**

- `F3_gpu_full_matrix_builder_v10_adaptive_zeros.cu`

**Purpose**

- Use GPU-safe builders for the ingredient matrices while retaining the v9 adaptive zero search.

**Main changes**

- Calls GPU-safe builders:

```cpp
f4gpu::gpu_config_maker_4_single_to_cpu_vectors(...);
f2gpu::F2_2plus1_mat_gpu_safe(...);
k2gpu::K2inv_EREord2_2plus1_mat_gpu_safe(...);
ggpu::G_2plus1_mat_gpu_safe(...);
pgpu::P_irrep_projection_2plus1_and_Vsel_gpu_safe(...);
```

- Then uses the existing GPU F3 solve pipeline:
  - `H X1 = F2`
  - `F3 = F2/3 - F2 X1`
  - `F3 X2 = Vsel`
  - `Vsel^H X2`
  - determinant

**Important limitation**

- Ingredients were built on GPU, but copied back to Eigen/CPU and then sent again to GPU for the solve.
- This was not fully device-resident yet.

**Fix file**

- `F3_gpu_full_matrix_builder_v10_adaptive_zeros_v2_compilefix.cu`

**Fixes added**

- Added missing helpers:
  - `Vec3`
  - `smallest_eigenvalue`
  - `normalize_det_vector_by_max`

---

## v11: Device-resident speed-step

**Main file**

- `F3_device_resident_speed_step_v11.cu`

**Purpose**

- Keep matrices device-resident from ingredient construction through F3 determinant.

**Main changes**

- Keeps on GPU:
  - `F2`
  - `G`
  - `K2inv`
  - `H`
  - `Vsel`
  - `X1`
  - `F3`
  - `X2`
  - `F3inv_projected`
- Copies back only:
  - determinant
  - small diagnostics
- Avoids the v10 round trip:

```text
GPU -> CPU Eigen -> GPU
```

**Compile fix**

- `F3_device_resident_speed_step_v11_v2_compilefix.cu`
- Fixed most-vexing-parse issues such as:

```cpp
DevicePtr<cuDoubleComplex> dF2(size_t(elems));
```

changed to:

```cpp
DevicePtr<cuDoubleComplex> dF2(static_cast<size_t>(elems));
```

**Guarded debug version**

- `F3_device_resident_speed_step_v11_v3_guarded.cu`

**Main debug additions**

- Stage-by-stage prints.
- SIGSEGV handler.
- Synchronization after major kernels.
- Helped identify crash inside `cublasZgetrsBatched`.

**cuSOLVER LU version**

- `F3_device_resident_speed_step_v11_v4_cusolver_lu.cu`

**Main fix**

- Replaced unstable single-matrix use of:

```cpp
cublasZgetrfBatched
cublasZgetrsBatched
```

with:

```cpp
cusolverDnZgetrf
cusolverDnZgetrs
```

**Limitations**

- Energies were still processed one at a time.
- GPU parallelism existed inside each energy, not across many energies.

---

## v12: Grouped stream-parallel device-resident version

**Main file**

- `F3_device_resident_grouped_stream_v12.cu`

**Purpose**

- Process multiple energies concurrently while keeping each energy device-resident.

**Main changes**

- Prepass over a chunk of energies.
- Build configs and determine:
  - `dim1`
  - `dim2`
  - `total_dim`
- Group energies by same shape.
- Process same-shape groups using multiple host workers.
- Compile with:

```bash
--default-stream per-thread
```

- Each host worker gets its own CUDA default stream.
- Each worker owns its own cuBLAS handle.
- Each energy remains device-resident through the solve.

**Execution model**

```text
same-shape group:
    energy 0 -> stream 0
    energy 1 -> stream 1
    energy 2 -> stream 2
    ...
```

**Limitations**

- Ingredient builders were still called per energy.
- Not yet true same-shape batched kernels.

---

## v13: Grouped batched-LU experimental version

**Main file**

- `F3_device_resident_grouped_batched_lu_v13.cu`

**Purpose**

- Add same-dimension grouping and start preparing for batched LU.

**Main changes**

- CPU/OpenMP config prepass.
- Groups by:
  - `dim1`
  - `dim2`
  - `total_dim`
- Adds cuBLAS batched-LU helper for same-`n` groups.
- Keeps v12 stream-parallel fallback.
- Uses cuSOLVER fallback for small or unsafe groups.

**Important behavior**

- v13 still mostly executed the per-energy stream path.
- It did not yet truly build `F2/G/K2` as full same-shape batches.

**Observed log pattern**

```text
[v13] processing group dim1=16 dim2=12 total_dim=28 count=188
[v13 group done] i=...
```

This indicated per-energy group processing rather than true batch execution.

---

## v14: True-batch attempt

**Main file**

- `F3_device_resident_true_batch_v14.cu`

**Purpose**

- Add explicit same-shape batch ingredient kernels.

**Main changes**

- Added batch kernels:
  - `v14_build_F2_batch_same_shape_kernel`
  - `v14_build_G_batch_same_shape_kernel`
  - `v14_build_K2_batch_same_shape_kernel`
  - `v14_combine_H_batch_kernel`
  - `v14_make_ptrs_kernel`
  - `v14_build_F3_batch_kernel`
- Intended layout:

```text
F2_batch[energy_id, row, col]
G_batch[energy_id, row, col]
K2_batch[energy_id, row, col]
```

**Limitation**

- The true-batch path was compiled but not actually selected in the logs.
- Logs showed fallback to v13 stream path.

**Observed log evidence**

```text
v14_true_batch_kernels=compiled; batched_lu_same_n=available; fallback_stream_path=enabled
[v13 group done] ...
```

This meant v14 did not yet run the true batch path.

---

## v15: Force-batch diagnostics

**Main file**

- `F3_device_resident_true_batch_v15_force_batch_diagnostics.cu`

**Purpose**

- Stop silent fallback and explicitly attempt true batch for same-shape groups.

**Main changes**

- Added explicit true-batch attempt logs.
- Groups by:
  - `dim1`
  - `dim2`
  - `total_dim`
- Attempts true batch when `count >= 2`.
- Prints reason if true batch fails.
- Falls back only after printing diagnostic reason.

**Observed successful batch selection**

```text
[v15] attempting true batch for group dim1=16 dim2=12 total_dim=28 count=5
[v15 true batch] attempting group dim1=16 dim2=12 n=28 batch=5 irrep=A2
[projection device-resident] N=28 vdim=2 irrep=A2
```

**Crash found**

- `cublasZgetrsBatched` crashed:

```text
libcublas.so.13(cublasZgetrsBatched+0xcd)
```

**Important result**

- v15 proved that the true-batch branch was now being entered.
- The failure was isolated to `cublasZgetrsBatched`.

---

## v16: True batch with custom batched GETRS

**Main file**

- `F3_device_resident_true_batch_v16_custom_batched_getrs.cu`

**Purpose**

- Keep true-batch path but avoid crashing `cublasZgetrsBatched`.

**Main changes**

- Still uses:

```cpp
cublasZgetrfBatched
```

for batched LU factorization.

- Replaces:

```cpp
cublasZgetrsBatched
```

with a custom CUDA kernel for GETRS.

**Custom GETRS kernel**

For each batch matrix and RHS column:

```text
1. apply LU pivots to RHS
2. solve L y = P b
3. solve U x = y
```

**Successful test log**

```text
[v15 true batch] attempting group dim1=16 dim2=12 n=28 batch=5 irrep=A2
[projection device-resident] N=28 vdim=2 irrep=A2
[v13 batched LU] getrf/getrs n=28 nrhs=28 batch=5
[v16 custom GETRS] launching pivot+triangular solve kernel n=28 nrhs=28 batch=5
[v13 batched LU] getrf/getrs n=28 nrhs=2 batch=5
[v16 custom GETRS] launching pivot+triangular solve kernel n=28 nrhs=2 batch=5
[v15 true batch done] dim1=16 dim2=12 n=28 vdim=2 batch=5 build_group=3.4242462389999999 solve_group=0.017288906999999999 per_energy_build=0.6848492478 per_energy_solve=0.0034577813999999998
```

**Important result**

- v16 successfully runs the true-batch path.
- Batched solve is now very fast.
- For the small test:
  - `per_energy_solve ≈ 0.00346 sec`
- Remaining bottleneck:
  - batch build / projection / config setup.

---

# Current status after v16

## What works

- CPU/OpenMP config prepass.
- Same-shape grouping.
- True-batch F3 path is entered.
- Batched LU factorization works.
- Custom batched GETRS works.
- Device-resident solve path works.
- Only determinant results are copied back.
- Adaptive zero search still works on determinant outputs.

## Current bottleneck

The solve step is now fast. The main cost is:

```text
config + F2/G/K2/Vsel batch build
```

especially:

```text
Vsel projection eigensolve per energy
```

## Recommended next optimization

1. Reuse `Vsel` within same-shape groups if it is strictly independent of energy for fixed:
   - `dim1`
   - `dim2`
   - `total_dim`
   - momentum shell/order
   - irrep

2. Build `Vsel` once per group instead of once per energy.

3. Use one group-level projection instead of repeated:

```text
[projection device-resident] N=28 vdim=2 irrep=A2
```

per energy.

4. Add group-level timers:
   - config prepass time
   - config upload time
   - F2 batch build time
   - G batch build time
   - K2 batch build time
   - Vsel build/reuse time
   - H solve time
   - F3 solve time
   - determinant time

---

# Useful run commands

## v16 small debug

```bash
/usr/bin/time -v stdbuf -oL -eL \
  ./test_F3_device_resident_true_batch_v16 1 1 0 A2 A2 5 50 0.26310 0.27 y \
  2>&1 | tee run_v16_debug_110_A2.log
```

## v16 full run

```bash
/usr/bin/time -v stdbuf -oL -eL \
  ./test_F3_device_resident_true_batch_v16 1 1 0 A2 A2 2000 300 0.26310 0.36 n \
  2>&1 | tee run_v16_110_A2.log
```

## v13 comparison

```bash
/usr/bin/time -v stdbuf -oL -eL \
  ./test_F3_device_resident_grouped_batched_lu_v13 1 1 0 A2 A2 2000 300 0.26310 0.36 n \
  2>&1 | tee run_v13_110_A2.log
```

---

# Naming summary

| Version | Main idea |
|---|---|
| v9 | Inward-shape adaptive zero refinement |
| v10 | GPU-safe builders introduced but copied back through CPU |
| v11 | Device-resident single-energy F3 path |
| v12 | Same-shape grouping with stream-parallel energies |
| v13 | Grouped path with batched-LU preparation |
| v14 | Added true-batch ingredient kernels, but mostly fell back |
| v15 | Forced true-batch diagnostics, isolated `getrsBatched` crash |
| v16 | True batch with custom GETRS kernel, successful small test |

