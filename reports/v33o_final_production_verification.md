# v33o final production verification

## Outcome

- compile: PASS
- target parity: PASS
- classifier dispatch: PASS
- parameter mask: PASS
- 10-FCN benchmark: PASS
- cutoff scan: PASS
- QC spectra: PASS
- spectrum plots: PASS
- determinant plots: PASS
- non-interacting L25: PARTIAL, not available in the current recent cache

## Evidence

- compile: `bash scripts/compile_v33g_all.sh`
- target parity: `reports/v33n_final_sanity_target_parity.md`
- classifier dispatch: `reports/v33n_final_sanity_classifier_dispatch.md`
- parameter mask: `reports/v33n_final_sanity_parameter_mask.md`
- benchmark: `reports/v33n_final_sanity_10fcn.md`
- cutoff scan summary: `reports/v33n_cutoff_scan/cutoff_scan_summary.md`
- verification CSV: `diagnostics/v33o/final_verification_summary.csv`

## Notes

- The v33n cutoff scan converged for all six cutoffs: 0.315, 0.325, 0.335, 0.345, 0.355, 0.365.
- The first three cutoff fits still report a Minuit initial-matrix warning, but they converge.
- Timing fields are backfilled from existing logs and artifact modification times; the original cutoff summary had `fit_wall_time_sec = 0.0`.

## Tabular summary

| check | status | details |
| --- | --- | --- |
| compile | PASS | bash scripts/compile_v33g_all.sh |
| target_parity | PASS | reports/v33n_final_sanity_target_parity.md |
| classifier_dispatch | PASS | reports/v33n_final_sanity_classifier_dispatch.md |
| parameter_mask | PASS | reports/v33n_final_sanity_parameter_mask.md |
| benchmark_fcn_10 | PASS | avg_total_sec=1.4074465948; model_found=32/32 |
| cutoff_scan_fits | PASS | 0.315, 0.325, 0.335, 0.345, 0.355, 0.365 converged |
| qc_spectrum_files_exist | PASS | 6/6 cutoff QC CSV files present |
| spectrum_plots_exist | PASS | 6/6 cutoff spectrum PNG/PDF bundles present |
| determinant_plots_exist | PASS | 60 PNG plots present |
| noninteracting_L25 | PARTIAL | recent cache stops at L24; L25 unavailable |
| timing_backfill | PASS | fit and spectrum-check wall times backfilled from logs |
