# v33o cutoff timing fix report

## Root cause

The cutoff scan used `/usr/bin/time -v`, but the parser matched the `Elapsed` line without trimming leading whitespace and then split on the first colon in the label itself. The log lines are tab-indented, so the wall time was parsed as zero.

## Fix

- Updated `scripts/run_v33n_cutoff_scan_fits.py` to strip each line before matching the elapsed-time field.
- Updated the parser to capture the wall-time value with a regex instead of splitting the label on the first colon.
- Backfilled the existing cutoff summary from the saved logs instead of rerunning the long production fits.

## Timing source

- fit wall time: reconstructed from `logs/v33n_cutoff_scan/Ecmcut_*_fit.log`
- spectrum-check wall time: reconstructed from `logs/v33n_cutoff_scan/Ecmcut_*_spectrum_only_*.log`
- plot wall time: approximated from artifact mtimes between fit-log completion and cutoff report emission
- total bundle wall time: reconstructed sum of fit, spectrum-check, and post-fit plotting/report time

## Result

- The historical v33n cutoff timing data is now nonzero in `diagnostics/v33o/cutoff_timing_summary.csv`.
- Future runs will parse wall time correctly.
