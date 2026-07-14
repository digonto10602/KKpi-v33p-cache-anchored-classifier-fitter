# 000_A1m hot-window one-FCN attempt 2

## Result

`CHECKPOINT_STATUS = MINIMAL_FITTER_000_A1M_ONE_FCN_HOT_WINDOWS_PASS`

The targeted L20 reader-convention fix restored the accepted root. The run
found all four expected model roots and returned a finite chi-square.

## Run

- Command: `bash output/v33p_minimal_fitter_000_A1m/run_one_fcn_hot_windows.sh > output/v33p_minimal_fitter_000_A1m/one_fcn_hot_windows_attempt2.log 2>&1`
- Config: `output/v33p_minimal_fitter_000_A1m/fitter_000_A1m_hot_windows.in`
- Sectors: `L20/000_A1m`, `L24/000_A1m`
- Root mode: `accepted_windows`
- Determinant backend: `auto -> cpu_openmp`
- `fallback_full_scan=0`
- K3df: `73735.840894011912, -972421.14060757787, 347174.05548116949, -1226756.7068845264`
- Ecm cutoff: `0.335`

## Measurements

| sector | roots found | rows evaluated | expansions | fallback/full scan |
|---|---:|---:|---:|---|
| L20/000_A1m | 1/1 | 101 | 0 | 0 / 0 |
| L24/000_A1m | 3/3 | 303 | 0 | 0 / 0 |
| total | 4/4 | 404 | 0 | 0 / 0 |

- chi2: `1.5125961842376174e-05`
- model_found: `4/4`
- FCN time: `0.000736493 s`
- cache load and window preparation: `350.256811719 s`
- wall time: `6:10.17`
- peak RSS: `7,653,700 kB` (~7.30 GiB)

## Correctness and safety

- No full 20,000-row determinant scan occurred inside FCN.
- No cache reload, plotting, or report/CSV writing occurred inside FCN.
- No window expanded and no fallback occurred.
- The L20 audit confirmed the original bracket rows `1284-1285` remain in
  the initial window and that the local determinant reproduces the trusted
  sign flip.
- No cachegen, minimization, GPU scanner, or cache binary copying was run.
- Physics formulas and classifier logic were unchanged.

The hot-window path is safe for a short 000_A1m-only minimization after user
approval. Cold-start I/O remains the dominant cost and was not optimized in
this pass.
