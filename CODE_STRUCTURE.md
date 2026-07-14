# Code Structure

## Top level

- `source/`: C++ / CUDA implementation
- `scripts/`: build, validation, scan, and plotting entrypoints
- `configs/`: config files for v31 through v33 workflows
- `docs/`: packaging notes and code maps
- `reports/`: validation and checkpoint reports
- `diagnostics/`: small CSV evidence files
- `LICENSE`: repo license
- `README.md`: top-level package summary
- `MANIFEST.md` and `PACKAGING_MANIFEST.md`: inclusion and exclusion rules

## Source clusters

- `source/qc_fitter_norm_refine_v2_multiL.cpp`: main multi-L fitter and coarse-cache reader
- `source/v33g_build_runtime_k3basis_cache.cpp`: runtime cache builder
- `source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp`: scalar diagnostics from the combined raw cache
- `source/digonto_classifier_v3.hpp`: classifier core
- `source/F3_cpu_openmp_v23_pure_cpu_multi_signflip_side_abs_refinement.cpp`: CPU OpenMP reference path
- `source/F3_gpu_omp_pipeline_v2_eigenbased.cpp`: GPU / Eigen pipeline reference

## Script clusters

- `scripts/compile_v33g_all.sh`: build all package binaries
- `scripts/validate_fitter_target_counts_v33k.py`: target count parity validation
- `scripts/verify_classifier_dispatch_v33m.py`: classifier dispatch audit
- `scripts/run_v33n_cutoff_scan_fits.py`: production cutoff scan
- `scripts/plot_v33g_runtime_qc_det_compare.py`: runtime cache vs fitter comparison plot
- `scripts/plot_oldscale_det_coarse20000_interactive.py`: checkpoint plot used for Gate 7

## Report clusters

- `reports/v33n_final_sanity_target_parity.md`: target count parity
- `reports/v33n_final_sanity_classifier_dispatch.md`: dispatch behavior
- `reports/v33n_final_sanity_parameter_mask.md`: parameter mask
- `reports/v33n_final_sanity_10fcn.md`: benchmark smoke test
- `reports/v33o_final_production_verification.md`: summary of the packaged checkpoint
