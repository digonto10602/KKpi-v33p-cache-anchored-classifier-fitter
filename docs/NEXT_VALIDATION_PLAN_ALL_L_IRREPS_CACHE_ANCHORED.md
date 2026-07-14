# Next validation plan: all L/irreps, cache anchored

## Scope

- Lbyas: `20`, `24`
- Irreps: `000_A1m`, `100_A2`, `110_A2`, `111_A2`, `200_A2`
- Ecm: `[actual cache Ecm_min, 0.36]` independently per sector
- Classifiers: `digonto_v3_window`, `digonto_v4_window` only

## Procedure for each sector

1. Load existing cache metadata first.
2. Use metadata `Ecm_min` and `Ecm_max` as the source of truth.
3. If the cache is missing, stop and report the missing path.
4. If the cache does not cover `0.36`, shrink to the available range and report it.
5. Do not run cachegen and do not copy cache binaries.
6. Use the corrected format-aware combined raw GPU cache reader only.
7. Use raw-sign methods only as candidate generators, never as production classifiers.
8. Produce a determinant grid, candidate brackets, accepted-mode predictions,
   QA plot, and user-label template.
9. Stop for user labels before scoring that sector.
10. Accept a sector only if user-labeled `FP=0` and `FN=0`.
11. If any sector fails, stop and request approval before changing classifier logic.

## Required per-sector outputs

- Cache-window audit
- Metadata-anchored determinant grid
- Candidate bracket CSV
- `digonto_v3_window` / `digonto_v4_window` predictions
- PNG/PDF QA plot
- Blank user-label template
- Sector validation report
