# Function Map

## Combined raw cache reader

- `source/qc_fitter_norm_refine_v2_multiL.cpp:627` - `load_gpu_coarse_cache_one`
- `source/v33g_build_runtime_k3basis_cache.cpp:186` - same coarse-cache reader logic in the runtime builder
- Purpose: load the combined raw `F3inv+Vsel` GPU cache, reconstruct `Eigen::MatrixXcd` objects, and attach them to the fitter data path

## Runtime cache writer / reader

- `source/v33g_runtime_k3basis_cache.hpp:1` - runtime cache metadata and compact matrix serialization
- Purpose: store projected matrices and K3 basis pieces with explicit `.meta.json` metadata

## Isotropic diagnostics

- `source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp:30` - `scaled_signed_logdet`
- `source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp:52` - `isotropic_projected_vector`
- `source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp:68` - `main`
- Purpose: read the combined raw cache and write scalar diagnostics for isotropic checks

## Classifier core

- `source/digonto_classifier_v3.hpp:36` - `sign`
- `source/digonto_classifier_v3.hpp:39` - `linear_zero`
- `source/digonto_classifier_v3.hpp:60` - `classify_shoulder`
- `source/digonto_classifier_v3.hpp:122` - `classify_one_flip`
- `source/digonto_classifier_v3.hpp:184` - `classify_series`
- Purpose: detect sign flips, classify shoulders, and produce candidate true-zero or pole labels

## Production fitter orchestration

- `scripts/run_v33n_cutoff_scan_fits.py:25` - repository paths and benchmark constants
- `scripts/run_v33n_cutoff_scan_fits.py:118` - `ctag`
- `scripts/run_v33n_cutoff_scan_fits.py:122` - `run_cmd`
- `scripts/run_v33n_cutoff_scan_fits.py:154` - `base_config_text`
- `scripts/run_v33n_cutoff_scan_fits.py:217` - `parse_benchmark_output`
- Purpose: compile, validate, benchmark, fit, plot, and report the v33n cutoff scan

## Dispatch and speed checks

- `scripts/verify_classifier_dispatch_v33m.py:36` - `make_temp_config`
- `scripts/verify_classifier_dispatch_v33m.py:46` - `run_mode`
- `scripts/verify_classifier_dispatch_v33m.py:86` - `write_outputs`
- `scripts/verify_classifier_dispatch_v33m.py:129` - `main`
- `scripts/run_v33m_classifier_speed_search.py:122` - `parse_summary`
- Purpose: confirm classifier mode routing and collect timing for FCN evaluation modes
