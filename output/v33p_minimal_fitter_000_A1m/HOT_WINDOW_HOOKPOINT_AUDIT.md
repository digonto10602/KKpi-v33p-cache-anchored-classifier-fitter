# Hot-window FCN hook-point audit

## Sources inspected

- `source/qc_fitter_norm_refine_v2_multiL.cpp`: `MultiConfig`, config parsing,
  `MultiLFCN` constructor, `MultiLFCN::operator()`, `MultiLFCN::model_for`,
  raw cache loading, and projected-basis setup.
- `source/qc_fitter_norm_refine_v2.cpp`: `find_QC_zeros_refined`,
  `find_QC_zeros_v3_from_coarse_grid`, `merge_QC_zeros_v4`, and
  `eval_entry_QC`.
- `source/K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp`:
  `assemble_QC` and `precompute_projected_k3_basis`.

## Hook points

The previous `MultiLFCN::model_for` called `find_QC_zeros_refined` once per
sector. That function evaluates every `ic.grid` row, causing a 20,000-row
determinant/classifier scan inside every FCN.

`MultiLFCN::model_for` now branches before that legacy call when
`root_search_mode=accepted_windows`. Accepted windows load once in the
constructor. The FCN evaluates only local rows through `eval_entry_QC`, then
uses the existing v3/v4 classifier dispatch. A missed 101-row window expands
once to its configured 501-row maximum. `fallback_full_scan=0` is enforced.

The determinant path remains the existing `assemble_QC`/`eval_entry_QC` path,
with current K3df parameters. The existing linear projected K3df basis is
precomputed only for configured maximum-window rows; no full-grid K3df basis
is built by the hot-window path.

## Risk assessment

The change is low-risk architecturally because the raw reader, matrix
assembly, scaling, determinant, and classifier implementations are reused
unchanged. The one-FCN test still found one model-root mismatch at the fixed
starting parameters; see the hot-window one-FCN report.
