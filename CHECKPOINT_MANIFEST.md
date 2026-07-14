# v33p checkpoint manifest

## Included

- source-controlled `source/`, `scripts/`, `configs/`, `docs/`, `tests/`,
  `reports/`, and `diagnostics/` material from the v33o source package
- `AGENTS.md` and `KKpi_CODEX_CODEBASE_GUIDE.md`
- corrected v33h reader source
- classifier sweep, scoring, plotting, and all-sector packaging scripts
- accepted L20/100_A2 decision, cache-window, source-audit, candidate, and
  true-zero artifacts
- current build and restore documentation

## Excluded

- `bin/`, `build/`, `.codex_build/`
- external `/media/digonto/Data/F3inv_cache/`
- cache binaries and generated raw determinant grids
- object, shared-library, archive, NumPy, HDF5, ROOT, and temporary data files
- temporary logs, Python bytecode, and massive generated output directories

## Recovery invariant

This package does not contain physics data needed to run independently. Restore
the external cache root and verify metadata before reading any cache.
