# Projected-basis hot-window checkpoint push

## Scope

This checkpoint records the L20/111_A2 label score, accepted windows for five
classifier-passing sectors, hot-window projected-basis binary/JSON caches, the
L20/200_A2 validation preparation, and the five-sector one-FCN pass.

## Validation recorded

- L20/111_A2: v3/v4 `TP=5 FP=0 FN=0 TN=25`; two accepted zeros and two lattice
  levels under the cutoff.
- Five-sector one-FCN: 12/12 roots, chi2 `0.0002394162713767611`, 2214 rows,
  fallback/full scan `0/0`.
- Projected-basis matrices agree with direct assembly to at most
  `3.410605131648481e-13` over the saved hot-window rows; all saved raw
  determinant signs agree.

## Safe files included

Source/scripts, accepted-window manifests, scoring reports, projected-basis
hot-window `.bin`/`.json`/validation artifacts, compact L20/200_A2 candidate
and classifier outputs, the fitter config/wrapper, and this report.

## Excluded

External cache binaries, original raw scans, full 20,000-row determinant grids,
temporary chunk directories, build artifacts, and massive logs are excluded.
L24/100_A2 remains excluded from fitter windows.

Commit: `b6825ae19e1c622512d0c2214eb76320784927dc`.
Branch target: `hot-window-fcn-pass`; push succeeded.
