# Short 000_A1m minimization report

`CHECKPOINT_STATUS = MINIMAL_FITTER_000_A1M_SHORT_MINIMIZATION_PARTIAL`

## Run

- Command: `bash output/v33p_minimal_fitter_000_A1m/run_short_minimize_hot_windows_000_A1m.sh > output/v33p_minimal_fitter_000_A1m/short_minimize_hot_windows_000_A1m.log 2>&1`
- Sectors: `L20/000_A1m`, `L24/000_A1m`
- Ecm cutoff: `0.335`
- Root mode: `accepted_windows`
- Determinant backend: `auto -> cpu_openmp`
- `fallback_full_scan=0`
- Hard FCN-call guard: `25`
- Existing `migrad(25)` control was insufficient by itself; an explicit
  `max_fcn_evals` guard was added after the first run made 152 calls.

## Parameters and chi-square

Starting parameters:

```text
K3iso0 = 73735.840894011912
K3iso1 = -972421.14060757787
K3B    = 347174.05548116949
K3E    = -1226756.7068845264
```

- starting chi2: `1.5125961842376174e-05`
- last evaluated chi2 at FCN 25: `1.5081201458745636e-05`
- last evaluated parameters: `(73738.66934246452, -972421.14060757787,
  349999.67978073552, -1226756.7068845264)`
- final fitted parameters: **not produced**; the guard stopped the run before
  Minuit returned a valid final minimum.

## Evaluation audit

- completed FCN evaluations: `25`
- attempted FCN 26: stopped immediately by the explicit guard
- model found on every completed evaluation: `4/4`
- L20 roots on every completed evaluation: `1/1`
- L24 roots on every completed evaluation: `3/3`
- rows per FCN: `404` (`101` L20 + `303` L24)
- window expansions: `0`
- fallback/full scan: `0/0`
- cache reload inside FCN: none
- cache load/window preparation: `327.251721401 s`
- peak RSS: `7,654,636 kB`
- wall time: `5:46.58`
- process exit: `1` due to the intentional FCN-call guard

The short minimizer did not complete normally, so this is a bounded partial
optimization result rather than a production fit. The numerical path itself
remained sane: chi2 decreased modestly and all four roots remained present.

Per-FCN timing was not emitted by the final executable log despite the
one-FCN baseline measurement of `0.000736493 s`; the hot-window work remained
404 rows per call. No full determinant scan, cachegen, cache regeneration,
GPU scanner, or minimization beyond the 25-call bound was run.
