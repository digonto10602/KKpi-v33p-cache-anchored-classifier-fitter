# V33P scatter-parameter audit

## Guard result

- Status: **PASS**
- Active production configs checked: `31`
- Failing active values: `0`
- `xival` is accepted as the existing config spelling of required `xi`.

## Active production config allowlist

- `configs/config_v33f_QC_spectrum_generator_v33e_v3.in`
- `configs/config_v33f_multiL_000A1m_5levels_v33e_v3.in`
- `configs/config_v33f_multiL_all_irreps_v33e_v3.in`
- `configs/config_v33g_build_runtime_cache.in`
- `configs/config_v33g_build_runtime_cache_000A1m.in`
- `configs/config_v33g_multiL_000A1m_runtime_smoke.in`
- `configs/config_v33g_multiL_all_irreps_runtime.in`
- `configs/config_v33g_validate_runtime_vs_v33f.in`
- `configs/config_v33g_validate_runtime_vs_v33f_000A1m.in`
- `configs/config_v33m_multiL_all_irreps_fastest_production.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p315.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p325.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p335.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p345.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p355.in`
- `configs/v33n_cutoff_scan/config_v33n_Ecmcut_0p365.in`
- `output/v33p_gpu_classifier_all_irrep/fitter/fitter_classifier_accepted_sectors.in`
- `output/v33p_gpu_classifier_all_irrep/fitter/fitter_current_accepted_sectors.in`
- `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_chi2_audit.in`
- `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_controlled_minimize.in`
- `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_hot_windows.in`
- `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_reduced_minimize.in`
- `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_short_minimize.in`
- `output_v33f/runtime_configs/config_v33f_multiL_000A1m_5levels_v33e_v3.pipeline1.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_000A1m_5levels_v33e_v3.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_000A1m_5levels_v33e_v3.v33gbuild.force.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_000A1m_5levels_v33e_v3.v33gbuild.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_all_irreps_v33e_v3.pipeline1.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_all_irreps_v33e_v3.pipeline1_omp1.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_all_irreps_v33e_v3.runtime.in`
- `output_v33f/runtime_configs/config_v33f_multiL_all_irreps_v33e_v3.v33gbuild.runtime.in`

## Wrong values found

- Legacy comparison configs under `configs/config_v31z*` and `configs/config_v32*` contain intentionally non-production scattering values, including `scatter1_10 = 0.10` and `1.00`, and small comparison s-wave values such as `0.01` or `0.1`.
- These files are not in the active allowlist and were not modified.
- No wrong value was found in an active production config.

## Active failures

- None.
