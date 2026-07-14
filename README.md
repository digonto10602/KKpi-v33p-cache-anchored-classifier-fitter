# KKpi v33p Cache-Anchored Classifier/Fitter Checkpoint

This is a source-only recovery checkpoint for the KKpi three-body
quantization workflow. It contains the corrected combined-raw GPU-cache reader,
cache-anchored determinant/classifier restart tools, accepted L20/100_A2
evidence, and the existing fitter startup path. Large caches and build outputs
remain external.

## Checkpoint status

`CHECKPOINT_STATUS = CACHE_ANCHORED_ACCEPTANCE_PACKAGED_READY_FOR_ALL_SECTOR_VALIDATION`

The validated reference sector is `Lbyas=20`, `irrep=100_A2`, with cache
metadata window `[0.26310, 0.36]` and `coarseN=20000`.

`digonto_v3_window` and `digonto_v4_window` both scored:

```text
TP=2 FP=0 FN=0 TN=23
accepted Ecm = 0.291556099879, 0.300441497547
```

This is sector- and cache-window-specific acceptance, not global classifier
approval. Other sectors still need user labels.

## Physics invariants

Every determinant path must preserve:

```cpp
QC = F3inv + K3df;
projected_QC = Vsel.adjoint() * QC * Vsel;
scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6.0);
det = determinant(scaled_projected_QC);
```

There is no determinant-level `Nproj` rescaling. The corrected reader uses the
format-aware combined raw `F3inv_Vsel_gpu.bin` payload and the
`variant_04_real_imag_swapped` complex convention.

## External cache requirement

Caches are not part of this repository. Set the external root to:

```text
/media/digonto/Data/F3inv_cache/
```

The canonical layout is, for example:

```text
v33e_coarse20000_sep_f2k2_L20_100_A2/
  v32zu_Lbyas20_gpu_cache/
    v32zu_Lbyas20_xi3p444_irrep100_A2_coarse20000_F3inv_Vsel_gpu.bin
    *.bin.meta.json
```

Metadata is the source of truth. The expected production window is
`Ecm_min=0.26310`, `Ecm_max=0.36`; do not substitute the nominal `0.26301`.
Validate provenance, `coarseN=20000`, row count, dimensions, and binary
existence before reading a cache.

## Build

The trusted CPU/reference tools use C++, Eigen, OpenMP, and the existing
Faddeeva source. CUDA tools additionally require `nvcc`, CUDA runtime,
cuBLAS, and cuSOLVER matching the installed driver. A minimal source build
from this checkpoint is:

```bash
cd /path/to/KKpi-v33p-cache-anchored-classifier-fitter
bash scripts/compile_v33g_all.sh
```

The script is the historical build entry point; inspect its compiler flags
before changing them. Do not copy `bin/` or `build/` from the source machine.

## Important executables and scripts

| Path | Purpose | Main inputs/options | Outputs |
|---|---|---|---|
| `source/v33h_patched_gpu_cache_oldscale_det_scan.cpp` | Corrected combined-raw reader and trusted CPU/OpenMP determinant scan | `--gpu-cache-root`, `--Lbyas`, `--irrep`, `--Emin`, `--Emax`, `--coarseN`, `--threads`, fixed K3 parameters, `--old-scaling true`, `--complex-read-convention variant_04_real_imag_swapped` | determinant CSV, summary, dimension jumps, hermiticity, worst rows |
| `scripts/run_v33p_classifier_sweep_L20_100_A2.py` | Existing CSV-only classifier restart harness | `--grid`, `--output-dir` | predictions, summary, sweep report |
| `scripts/score_v33p_classifier_against_user_labels.py` | Scores predictions against completed user labels | prediction CSV and label CSV options | score CSV/report |
| `scripts/plot_v33p_classifier_det_family_interactive.py` | Determinant-family QA plot | `--grid`, `--prediction-file`, `--candidate-file`, `--n-scale`, `--output-stem` | PNG, PDF, marker map |
| `scripts/prepare_v33p_all_sector_validation.py` | Packages finalized scan CSVs without changing classifier logic | fixed sector/cache inventory in script | per-sector grids, candidates, accepted outputs, plots, label templates |
| `bin/v33f_k3df_fitter_multiL_v33e` | Historical multi-L fitter binary, built locally | fitter config and benchmark subcommands | fit/spectrum reports |
| `scripts/run_v33n_cutoff_scan_fits.py` | Historical cutoff/fitter orchestration | cutoff configs and external caches | fitter outputs/reports |

The repository intentionally does not include compiled executables. Build them
locally after restoring the external cache path.

## Function map

- Corrected reader: `source/v33h_patched_gpu_cache_oldscale_det_scan.cpp`,
  `resolve_gpu_cache`, `validate_metadata`, `scan_gpu_cache`,
  `compute_row`, and `process_block`.
- Projection/scaling: `compute_row` constructs `QC`, `projected_QC`, and
  `scaled_projected_QC` before determinant evaluation.
- Classifier dispatch: `scripts/run_v33p_classifier_sweep_L20_100_A2.py`,
  `sign_flips`, `v3_classify`, `v3_shoulder`, and the existing mode loop.
- Accepted modes: `digonto_v3_window` and `digonto_v4_window` are retained by
  the packaging wrapper; classifier logic is unchanged.
- Scoring: `scripts/score_v33p_classifier_against_user_labels.py`.
- Plotting: `scripts/plot_v33p_classifier_det_family_interactive.py`.
- Fitter path: `source/qc_fitter_norm_refine_v2_multiL.cpp`,
  `source/K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp`,
  and `scripts/run_v33n_cutoff_scan_fits.py`.

## Logic flow

```text
external cache + metadata
        -> corrected raw reader
        -> QC / projection / fixed scaling
        -> determinant grid
        -> dimension-safe sign-change brackets
        -> existing classifier modes
        -> user labels
        -> TP/FP/FN/TN scoring
        -> accepted true zeros
        -> fitter target mapping / one-FCN test
```

Raw-sign modes may generate candidate brackets for diagnostics but are not
production classifiers. Do not use them for fitter targets.

## GPU determinant backend status

The workspace contains older CUDA/cuBLAS/cuSOLVER determinant experiments,
including device-resident LU sources. The audited batched backend consumes
preprojected v32x rows; it is not the corrected combined-raw reader path used
by this checkpoint. It therefore is not silently promoted to production.

The required next implementation is a chunked double-complex GPU determinant
backend behind the corrected reader, with grouped dimensions, LU pivot parity,
partial temporary CSVs, progress, CPU fallback, and CPU-reference parity.
See `docs/GPU_DET_BACKEND_AUDIT.md` and
`docs/GPU_VS_CPU_DET_VALIDATION_STATUS.md`. Until that gate passes, the CPU
scanner remains the trusted reference and GPU results must not drive fitter
validation.

## Minimal fitter test

The first fitter test is intentionally restricted to:

```text
Lbyas = 20, 24
irrep = 000_A1m only
Ecm cutoff = 0.335 unless an existing config overrides it
```

It must use accepted classifier zeros from completed user labels, compare
accepted-zero counts to lattice levels, load caches once outside FCN, confirm
that no 20,000-row scan occurs inside FCN, and run one FCN/chi-square
evaluation before any minimization. The plan is in
`output/v33p_minimal_fitter_000_A1m/FITTER_000_A1M_TEST_PLAN.md`.

## Restore

```bash
git clone https://github.com/digonto10602/KKpi-v33p-cache-anchored-classifier-fitter.git
cd KKpi-v33p-cache-anchored-classifier-fitter
export V33P_CACHE_ROOT=/media/digonto/Data/F3inv_cache
bash scripts/compile_v33g_all.sh
```

Then verify the accepted reference without rebuilding caches:

```bash
python3 scripts/run_v33p_classifier_sweep_L20_100_A2.py \
  --grid output/v33p_classifier_restart_L20_100_A2_E026310_0360/det_grid_L20_100_A2_corrected_E026310_0360.csv \
  --output-dir /tmp/v33p_reference_sweep
```

Resume all-sector work only from metadata windows and existing external
caches. Fill the generated user-label templates before scoring. Never run
cachegen as part of restore.

## Known limitations

- All-sector user-label validation is not complete.
- GPU determinant integration with the corrected raw reader is not yet
  validated and must not be used for fitter acceptance.
- The minimal 000_A1m fitter test is blocked until its labels/scoring are
  complete.
- Large cache binaries, build trees, and raw determinant grids are external.

This repository packages the current v33n checkpoint of the KKpi / K3df multi-`Lbyas` workflow.

Current state:

- combined raw GPU cache reader bug fixed and validated
- corrected old-scale determinant scan passed for `Lbyas=20`, `irrep=100_A2`, `Ecm=[0.34, 0.36]`, `coarseN=20000`
- CPU OpenMP subset validation passed against the corrected cache path
- corrected `fix1` interactive plot regenerated from the right CSV
- classifier restart bundle prepared, waiting only on user-labeled true zeros

The packaging goal is reproducibility, not new physics or new fitting rules.

## What this repository contains

- source code for the fitter, cache builders, classifier utilities, and plotting helpers
- configs for the v33f, v33g, v33m, and related workflows
- validation reports and checkpoint notes
- packaging docs for the v33o handoff

Large generated caches, build trees, and binary outputs are intentionally excluded.

## Critical checkpoint

The key fix was the combined raw `F3inv+Vsel` GPU cache reader. The target cache is:

```text
/media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_100_A2/v32zu_Lbyas20_gpu_cache/v32zu_Lbyas20_xi3p444_irrep100_A2_coarse20000_F3inv_Vsel_gpu.bin
```

The corrected path reconstructs the raw payload consistently and yields:

- `max |det_imag| = 0`
- `F3inv` hermiticity error at roundoff
- `projected_QC` hermiticity error at roundoff
- CPU/GPU subset agreement at sub-percent level

## Exact old scaling

Use matrix-level old scaling only:

```cpp
const double const_norm_scale = std::pow(Lbyas * xi, 6.0);
Eigen::MatrixXcd QC = F3inv + K3df;
Eigen::MatrixXcd projected_QC = Vsel.adjoint() * QC * Vsel;
Eigen::MatrixXcd scaled_projected_QC = projected_QC / const_norm_scale;
std::complex<double> det = scaled_projected_QC.determinant();
```

Do not apply determinant-level `Nproj` rescaling.

## Validation numbers

The current checkpoint is the one documented in:

- `reports/v33o_final_production_verification.md`
- `reports/v33n_final_sanity_target_parity.md`
- `reports/v33n_final_sanity_classifier_dispatch.md`
- `reports/v33n_final_sanity_parameter_mask.md`
- `reports/v33n_final_sanity_10fcn.md`

Representative values:

- `model_found = 32/32`
- `chi2 = 0.2543510436151185` for the 10-FCN smoke benchmark
- `max percent_diff_F3inv_frob ~ 3.2005e-05 %`
- `max percent_diff_QC_frob ~ 3.2005e-05 %`
- `max percent_diff_projected_QC_frob ~ 1.3901e-05 %`
- `max percent_diff_scaled_projected_QC ~ 1.3901e-05 %`
- `max stable percent_diff_det_complex ~ 6.0955e-07 %`
- `max projector_reldiff ~ 9.35e-16`

## Build and smoke test

```bash
bash scripts/compile_v33g_all.sh
python3 scripts/validate_fitter_target_counts_v33k.py
stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e configs/config_v33m_multiL_all_irreps_fastest_production.in benchmark-fcn --repeat 10 --warmup 2
```

## Main workflow commands

Run the production cutoff scan:

```bash
python3 scripts/run_v33n_cutoff_scan_fits.py
```

Generate the runtime cache validation plots:

```bash
python3 scripts/plot_v33g_runtime_qc_det_compare.py
```

Validate classifier dispatch:

```bash
python3 scripts/verify_classifier_dispatch_v33m.py
```

## Current classifier state

The classifier is not being tuned in this checkpoint. It is parked at:

- `output/v33n_classifier_restart_from_fix1_L20_100_A2/`

That bundle contains candidate sign-change brackets and segment summaries for the eight candidate roots in the corrected `fix1` CSV.

## Cache format notes

Two related cache formats exist in this repo:

1. Combined raw `F3inv+Vsel` GPU cache
   - file suffix: `_F3inv_Vsel_gpu.bin`
   - used by the corrected reader path
   - stores raw complex payloads for `F3inv` and `Vsel`

2. Runtime cache
   - file suffix: `_runtime_K3basis.bin`
   - stores projected matrices and K3 basis pieces for the hot fitter path
   - paired with `.meta.json` metadata

The reader must be format-aware. File name alone is not enough.

## Repository layout

- `source/`: fitter, cache, and classifier implementation
- `scripts/`: build, validation, plotting, and scan drivers
- `configs/`: runtime and production configs
- `docs/`: packaging notes and code-map documents
- `reports/`: validation reports and checkpoint notes
- `diagnostics/`: small verification CSVs

## Reproducibility

Large runtime caches live outside the repo and are mounted from:

```text
/media/digonto/Data/F3inv_cache
```

The repo itself carries the source, scripts, configs, docs, and validation evidence needed to recreate the checkpoint.

## Known limits

- L25 non-interacting data is not present in the current recent cache tree.
- This package does not include large binary cache artifacts.
- The classifier still needs user-labeled true zeros before any tuning step.

## See also

- `QUICKSTART.md`
- `CODE_STRUCTURE.md`
- `FUNCTION_MAP.md`
- `VALIDATION_STATUS.md`
- `GPU_CACHE_READER_FIX.md`
- `CLASSIFIER_RESTART_NOTES.md`
- `REPRODUCIBILITY.md`
