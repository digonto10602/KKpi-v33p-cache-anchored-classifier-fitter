# KKπ Three-Body QC Code Version History

This document summarizes the code evolution used for the KKπ three-body quantization-condition workflow, starting from the Python QC3 reference code and ending with the latest cached Minuit-based Kdf,3 fitter.

---

## Python QC3 Reference Codebase

**Source package:** `QC3_release.zip`

This codebase is the Python/NumPy/SciPy reference implementation of the QC3 formalism. It provides the reference definitions for:

- spectator momentum lists,
- orbit construction,
- two-flavor `2+1` basis ordering,
- `F`, `G`, `K2^{-1}` matrix construction,
- `F3` construction,
- irrep projection,
- and `Kdf,3` matrix construction through `K3main.K3mat_2plus1(...)`.

The central Python formula is

```python
F3 = F/3 - F @ LA.inv(K2i + F + G) @ F
QC3_mat = LA.inv(F3) + K3
```

where

```python
K3 = K3main.K3mat_2plus1(
    E, L, nnP,
    K3iso,
    K3B_par,
    K3E_par,
    M12=M12,
    waves=waves
)
```

The Python `K3mat_2plus1(...)` builds the block matrix

```text
K3 = [ K11              K12 / sqrt(2) ]
     [ K21 / sqrt(2)    K22 / 2       ]
```

with flavor-1 spectator sector using `sp` waves and flavor-2 spectator sector using `s` waves.

---

## v23: Pure CPU/OpenMP/Eigen F3 Baseline

**Main code:**

```text
F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement.cpp
F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement_compilefix1.cpp
```

### Purpose

v23 was the clean CPU/OpenMP/Eigen reference version of the determinant scanner. It removed GPU/cuBLAS dependencies and became the trusted baseline for checking the matrix-building and zero-search workflow.

### Main determinant target

v23 searched zeros of the projected inverse F3 determinant:

```text
det[ Vsel† F3^{-1} Vsel ] = 0
```

In code, this was implemented without explicitly forming the full inverse:

```cpp
Eigen::PartialPivLU<Eigen::MatrixXcd> luF3(F3);
Eigen::MatrixXcd X2 = luF3.solve(Vsel);
Eigen::MatrixXcd F3inv_projected = Vsel.adjoint() * X2;
out.det = determinant_via_partial_piv_lu(F3inv_projected);
```

### Features

- Pure CPU/OpenMP matrix building.
- Eigen-based linear solves.
- No CUDA, cuBLAS, or cuSOLVER.
- Adaptive sign-flip refinement.
- Side-wise `abs(det)` classification:
  - `likely_zero`,
  - `likely_pole`,
  - `ambiguous`.
- Coarse scan plus refinement around sign flips.

### Important limitation

v23 did **not** include `Kdf,3`. It effectively used

```text
Kdf,3 = 0
```

and solved only the projected `F3^{-1}` determinant.

---

## C++ K3mat_2plus1 Header

**File:**

```text
K3_functions_2plus1.hpp
```

### Purpose

This header translated the Python QC3 `K3main.K3mat_2plus1(...)` logic into C++ for use with the existing C++ `config_maker_4` basis.

### Main idea

Unlike the Python matrix, which is momentum-major inside each flavor block, the C++ `config_maker_4` basis is effectively based on each config entry knowing its own:

```text
momentum, ell, m
```

Therefore the C++ K3 builder fills matrix elements directly from row and column config information rather than relying on a Python-style block-local ordering.

### K3 structure

The C++ version follows the same block normalization as Python:

```text
K3 = [ K11              K12 / sqrt(2) ]
     [ K21 / sqrt(2)    K22 / 2       ]
```

and supports

```text
Kiso(Delta) = K3iso0 + K3iso1 * Delta + ...
K3B_par
K3E_par
```

---

## v24: QC Determinant with Kdf,3

**Main code:**

```text
F3_cpu_openmp_v24_K3_QC_multi_signflip_side_abs_refinement.cpp
```

**Compile script:**

```text
compile_F3_cpu_openmp_v24_K3_QC_multi_signflip_side_abs_refinement.sh
```

**Test script:**

```text
run_test_F3_cpu_openmp_v24_K3_QC.sh
```

### Purpose

v24 extended v23 by adding `Kdf,3` into the quantization condition.

### Main determinant target

v24 replaced

```text
det[ Vsel† F3^{-1} Vsel ]
```

with

```text
det[ Vsel† (F3^{-1} + Kdf,3) Vsel ]
```

implemented as

```cpp
Eigen::PartialPivLU<Eigen::MatrixXcd> luF3(F3);
Eigen::MatrixXcd X2 = luF3.solve(Vsel);

Eigen::MatrixXcd F3inv_projected = Vsel.adjoint() * X2;
Eigen::MatrixXcd K3_projected    = Vsel.adjoint() * K3 * Vsel;
Eigen::MatrixXcd QC_projected    = F3inv_projected + K3_projected;

out.det = determinant_via_partial_piv_lu(QC_projected);
```

### New parameters

v24 introduced Kdf,3 parameters:

```text
K3iso0
K3iso1
K3B
K3E
```

with defaults such as

```text
K3iso0 = 200
K3iso1 = 400
K3B    = 0
K3E    = 0
```

### Output

Standalone v24 scan code wrote determinant and zero files such as

```text
det_proj_QC_K3_raw_...
det_proj_QC_K3_normalized_...
finalized_Ecm_zeros_...
```

---

## C++ Lattice Covariance Header

**File:**

```text
lattice_data_covariance_cpp.hpp
```

### Purpose

This header rewrote the Python lattice covariance workflow into C++.

It implements the equivalent of

```python
states_avg1, states_err1, nP_list1, state_no1, L_list1,
covariance_mat1, correlation_mat1 = covariance_between_states_szscl21_based(
    ensemble1,
    Lval1,
    xival1,
    energy_cutoff,
    list_of_mom,
    max_state
)
```

### Key conventions

For lattice data conversion,

```text
L = Lval * xival
P = 2π / L * |nP|
Ecm = sqrt(En^2 - P^2)
```

The same convention is used in the QC side through

```cpp
par.xi = s.xival;
par.Lbyas = s.Lval;
par.L() = par.xi * par.Lbyas;
```

Thus with

```cpp
s.Lval = 20.0;
s.xival = 3.444;
```

both data and QC use

```text
L = 68.88
```

---

## v24 Minuit Fitter

**Main code:**

```text
K3df_minuit_fit_v24.hpp
F3_cpu_openmp_v24_K3QC_core.hpp
test_K3df_minuit_fit_v24.cpp
```

### Purpose

This version wrapped the v24 QC zero finder inside a Minuit2 fit.

### Fit parameters

```text
K3iso0
K3iso1
K3B
K3E
```

### Fit workflow

For each Minuit trial point:

1. Load lattice energies and covariance using `covariance_between_states_szscl21_based(...)`.
2. Parse each momentum label into `(nnP, irrep, irrep_tag)`.
3. Run the v24 QC zero finder for each irrep/momentum.
4. Sort `likely_zero` energies.
5. Match QC levels one-to-one with lattice levels.
6. If QC levels are missing, pad missing model energies with `0.0` according to the original requested convention.
7. Compute chi-square using the covariance/correlation matrix.
8. Minuit minimizes chi-square.

### Important momentum-label map

The fitter was corrected to use the desired canonical QC representatives:

```text
"000_A1m" -> nnP = {0,0,0}, irrep = A1u, irrep_tag = A1m
"100_A2"  -> nnP = {0,0,1}, irrep = A2,  irrep_tag = A2
"110_A2"  -> nnP = {1,1,0}, irrep = A2,  irrep_tag = A2
"111_A2"  -> nnP = {1,1,1}, irrep = A2,  irrep_tag = A2
"200_A2"  -> nnP = {0,0,2}, irrep = A2,  irrep_tag = A2
```

A shell-equivalence check was added so the covariance code’s lattice convention, for example `{1,0,0}`, can still match the QC representative `{0,0,1}`.

### Minuit2 compile issue fixed

Standalone conda-forge Minuit2 uses

```cpp
hesse(fcn, min);
```

not

```cpp
min = hesse(fcn, min);
```

This API mismatch was fixed.

---

## v25: Cached Minuit Fitter

**Main code:**

```text
K3df_minuit_fit_v25_cached.hpp
F3_cpu_openmp_v25_K3QC_cached_core.hpp
test_K3df_minuit_fit_v25_cached.cpp
```

### Purpose

v25 added an in-memory cache for the K3-independent projected inverse F3 object.

### Cached object

Instead of caching the full QC matrix, v25 caches

```text
F3inv_projected = Vsel† F3^{-1} Vsel
```

plus the ingredients needed to rebuild the K3 projection:

```text
plm_config
klm_config
total_P
Vsel
En
dim1, dim2, total_dim, vdim
```

Then for a new Kdf,3 parameter guess it only recomputes

```text
K3
K3_projected = Vsel† K3 Vsel
QC_projected = F3inv_projected + K3_projected
det(QC_projected)
```

### Cache key

The cache key is based on

```text
nnP | irrep | Ecm
```

so all repeated coarse-grid energies are reusable after the first FCN evaluation. Refined energies are reused only if the exact same refined `Ecm` is visited again.

### Per-FCN evaluation printout

v25 prints diagnostics such as

```text
[K3dfFCN-v25] eval=...
chi2=...
chi2/ndof=...
nlevels=...
missing=...
K3iso0=...
K3iso1=...
K3B=...
K3E=...
cache_size=...
cache_hits=...
cache_misses=...
```

This gives live visibility into the fit and cache behavior.

### Limitation found

The final post-fit energy recalculation could create a new local cache, causing it to recompute F3i instead of reusing the FCN cache.

This was numerically okay but inefficient.

---

## v26: Shared-Cache Reuse and Automatic Fit Output Files

**Main code:**

```text
K3df_minuit_fit_v26_reuse_cache.hpp
F3_cpu_openmp_v25_K3QC_cached_core.hpp
test_K3df_minuit_fit_v26_reuse_cache.cpp
compile_K3df_minuit_fit_v26_reuse_cache.sh
```

### Purpose

v26 fixes the v25 cache lifetime issue.

### Main cache fix

v26 creates one shared cache:

```cpp
auto shared_cache = std::make_shared<F3iProjectedCache>();
```

and passes it to

```text
Minuit FCN evaluations
final best-fit model-energy solve
finite-difference model-energy covariance propagation
```

Therefore, if a projected F3i object has already been computed for a particular

```text
(nnP, irrep, Ecm)
```

then the final post-fit solve and optional finite-difference solves can reuse it.

### Output files generated

v26 automatically writes the requested files after the fit:

```text
fit_summary_v25.dat
fit_levels_v25.dat
fit_parameters_v25.dat
fit_parameter_covariance_v25.dat
fit_parameter_correlation_v25.dat
fit_qc_zeros_v25.dat
```

The filenames use

```cpp
s.output_tag = "v25";
```

so the latest implementation still writes the requested `*_v25.dat` names.

### Output-file contents

#### `fit_summary_v25.dat`

Contains global fit information:

```text
valid
chi2
ndata
npar
ndof
chi2_dof
```

#### `fit_levels_v25.dat`

Contains one row per fitted lattice level:

```text
i
mom_label
irrep
irrep_tag
nPx nPy nPz
state_no
data_Ecm
data_err
model_Ecm
model_err
residual_sigma
```

#### `fit_parameters_v25.dat`

Contains best-fit Kdf,3 parameters and errors:

```text
K3iso0 value error
K3iso1 value error
K3B    value error
K3E    value error
chi2
chi2_dof
ndata
ndof
```

#### `fit_parameter_covariance_v25.dat`

The `4 x 4` Minuit parameter covariance matrix.

#### `fit_parameter_correlation_v25.dat`

The `4 x 4` Minuit parameter correlation matrix.

#### `fit_qc_zeros_v25.dat`

All final best-fit QC zeros by momentum/irrep label:

```text
mom_label zero_index Ecm
```

---

## Current Recommended Workflow

### First small test

Use only one irrep:

```cpp
s.list_of_mom = {"000_A1m"};
s.energy_cutoff = 0.34;
s.scan_E0 = 0.255;
s.scan_E1 = 0.345;
s.coarseN = 120;
s.refineN = 30;
s.compute_model_energy_covariance = false;
```

### Production-style expansion

After the small test behaves well, increase the scan quality:

```cpp
s.energy_cutoff = 0.38;
s.scan_E1 = 0.39;
s.coarseN = 300;   // or larger
s.refineN = 50;
```

Then add all momenta:

```cpp
s.list_of_mom = {
    "000_A1m",
    "100_A2",
    "110_A2",
    "111_A2",
    "200_A2"
};
```

### Important runtime note

The cache is in-memory only. It persists during one executable run but is not written to disk. Restarting the executable starts with an empty cache.

---

## Remaining Possible Improvements

1. **Disk cache for F3i**  
   Save projected F3i objects to disk so a new executable run can reuse previous computations.

2. **Tolerant cache key**  
   Current cache reuse requires exact same `Ecm`. A future version could use rounded keys such as `round(Ecm / 1e-12)`.

3. **Missing-level penalty**  
   Instead of padding missing QC levels by `0.0` during Minuit minimization, use a controlled penalty to make the chi-square surface less discontinuous.

4. **Per-irrep scan windows**  
   Use data-driven scan windows per momentum/irrep instead of one global scan range.

5. **Cache statistics file**  
   Add a file such as

   ```text
   fit_cache_stats_v25.dat
   ```

   with cache size, hits, misses, and per-irrep reuse information.
