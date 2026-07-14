# Hot-window fitter runtime audit

CHECKPOINT_STATUS = `MINIMAL_FITTER_000_A1M_ONE_FCN_HOT_WINDOWS_PASS`

- Cache loading: once in the `MultiLFCN` constructor.
- Accepted windows: once in the constructor through `load_accepted_windows`.
- FCN full-scan call: bypassed when `root_search_mode=accepted_windows`.
- Full 20,000-row scan inside FCN: disabled and not reachable through fallback.
- Cache reload inside FCN: none.
- Plotting inside FCN: none.
- CSV/report writing inside FCN: none for `fcn-once`.
- `fallback_full_scan`: configured and enforced as `0`.
- Determinant backend: `auto` resolves to `cpu_openmp` in this pass.
- Local rows: four initial windows of 101 rows, 404 total; maximum expanded
  range is four windows of 501 rows, 2004 total. The corrected one-FCN run
  evaluated 404 rows total: 101 for L20 and 303 for L24, with no expansions.
- Legacy full scan: retained only for explicit non-`accepted_windows` mode.
- GPU full-grid scanner: not implemented or connected.

The FCN recomputed determinants for the current K3df parameters only on local
window rows and assigned the resulting local roots to the matched lattice
levels. The targeted L20 reader-convention fix restored the accepted root;
the fixed-parameter one-FCN result found all 4 of 4 roots with finite chi2.
