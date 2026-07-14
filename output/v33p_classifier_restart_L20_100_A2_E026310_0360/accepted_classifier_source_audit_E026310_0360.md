# Accepted classifier source audit

## Existing implementation used

- Sweep harness: `scripts/run_v33p_classifier_sweep_L20_100_A2.py`
- Accepted modes: `digonto_v3_window`, `digonto_v4_window`
- Plot harness: `scripts/plot_v33p_classifier_det_family_interactive.py`
- Determinant reader: `bin/v33h_patched_gpu_cache_oldscale_det_scan`
- Source reader: `source/v33h_patched_gpu_cache_oldscale_det_scan.cpp`

The accepted predictions were selected from the existing sweep output. No
classifier source file or classifier logic was modified for this package.
The reader uses the corrected combined-raw format and
`variant_04_real_imag_swapped` convention.

## Scope

Acceptance is sector- and cache-window-specific. It is not a global classifier
approval. All other Lbyas/irrep sectors require independent metadata-anchored
validation and user labels.
