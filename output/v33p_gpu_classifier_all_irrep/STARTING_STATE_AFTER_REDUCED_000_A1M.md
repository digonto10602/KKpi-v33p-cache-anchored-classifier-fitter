# Starting state after reduced 000_A1m fit

## Status

`COVINV_SENSITIVITY_COMPLETE_AND_000_A1M_REDUCED_FIT_PASS`

The validated 000_A1m hot-window path uses external raw v33e/v32zu caches,
accepted windows, `det_backend=auto -> cpu_openmp`, and
`fallback_full_scan=0`.

## Reduced fit

- Sectors: L20/000_A1m and L24/000_A1m
- Floating: K3iso0, K3iso1, K3B
- Fixed: K3E `-1226756.7068845264`
- Final chi2: `1.4818722402508267e-05`
- Minuit valid: `1`
- FCNs: `137`
- Model found: `4/4` on every FCN
- Rows per FCN: `404`
- Fallback/full scans: `0/0`

## Source checkpoint

Local source changes from this phase are in:

- `source/qc_fitter_norm_refine_v2_multiL.cpp`

They add diagnostic parameter sensitivity and a config-driven floating/fixed
parameter mask. Physics formulas and classifier logic are unchanged.

The safe checkpoint commit and public branch are recorded in the final handoff
after the selected small files are committed.
