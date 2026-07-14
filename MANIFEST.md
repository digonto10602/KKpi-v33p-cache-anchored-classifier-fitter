# KKpi v33o production fitter manifest

## Included

- `source/`
- `scripts/`
- `configs/`
- `docs/`
- selected `reports/` from the v33m and v33n workflows
- v33o verification reports and diagnostics in the repo root

## Excluded

- `bin/`
- `build/`
- `cache/`
- `output*/`
- `plots/`
- large generated binaries and fit outputs

## Notes

Runtime caches must live outside the packaged tree and be matched by metadata before reuse.
