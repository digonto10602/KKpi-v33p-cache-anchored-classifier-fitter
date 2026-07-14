# v33o non-interacting L19-L25 verification

## Result

PARTIAL

## Findings

- The current recent non-interacting cache tree contains L19 through L24 files.
- No L25 non-interacting rows were present in the current cache root.
- The v33n cutoff plots therefore use the available cache rows only; there is no source data here to backfill L25.

## Evidence

- cache root: `output_recent_nonint_plotter/recent_nonint_L16to24_cache`
- verification CSV: `diagnostics/v33o/noninteracting_L19_L25_summary.csv`

## Consequence

The L25 risk is real and remains documented. The package should not claim full L19-L25 coverage until a new non-interacting source for L25 is generated.
