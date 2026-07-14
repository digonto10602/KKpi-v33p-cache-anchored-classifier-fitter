# Packaging Manifest

## Included

- `source/`
- `scripts/`
- `configs/`
- `docs/`
- `reports/`
- `diagnostics/`
- `README.md`
- `QUICKSTART.md`
- `CODE_STRUCTURE.md`
- `FUNCTION_MAP.md`
- `VALIDATION_STATUS.md`
- `GPU_CACHE_READER_FIX.md`
- `CLASSIFIER_RESTART_NOTES.md`
- `REPRODUCIBILITY.md`
- `MANIFEST.md`
- `VERSION`
- `.gitignore`

## Excluded

- `bin/`
- `build/`
- `cache/`
- `output*/`
- `plots/`
- large binary cache files
- object files
- local editor and Python cache artifacts

## Notes

- The repo should remain small enough to clone without the machine-local GPU cache.
- Validation evidence is kept as small text/CSV/markdown artifacts.
- The package is meant to be reproducible from source plus the documented external cache path.
