# Hot-window FCN checkpoint

`CHECKPOINT_STATUS = MINIMAL_FITTER_000_A1M_ONE_FCN_HOT_WINDOWS_PASS`

The accepted-window FCN path passes the fixed-parameter one-FCN test for
`L20/000_A1m` and `L24/000_A1m`:

- roots: `1/1`, `3/3`, total `4/4`
- rows evaluated: `404`
- chi2: `1.5125961842376174e-05`
- FCN time: `0.000736493 s`
- cache/window preparation: `350.256811719 s`
- peak RSS: `7,653,700 kB`
- full scan/fallback: `0/0`

## Correctness fix

The hot raw-cache path applied one variant-04 real/imag swap while the
trusted v33h determinant path applies two effective swaps. A localized second
swap was added in `load_gpu_coarse_cache_one` in
`source/qc_fitter_norm_refine_v2_multiL.cpp`. The trusted L20 bracket was
restored; physics formulas and classifier logic were unchanged.

## FCN safety

`root_search_mode=accepted_windows`, `det_backend=auto` (CPU/OpenMP), and
`fallback_full_scan=0` are required. The FCN evaluates only accepted local
windows; it does not reload caches, write reports, or scan the 20,000-row
determinant grid.

## Next step

Run only the bounded short `000_A1m` minimization. Do not start production
minimization, all-sector validation, cachegen, or the GPU scanner from this
checkpoint.
