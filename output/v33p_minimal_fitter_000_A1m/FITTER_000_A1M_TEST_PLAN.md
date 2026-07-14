# Minimal fitter test plan: 000_A1m only

Status: **MINIMAL_FITTER_000_A1M_ONE_FCN_HOT_WINDOWS_PASS**.

## Scope

- Lbyas: `20`, `24`
- irrep: `000_A1m` only
- cache window: metadata `[0.26310, 0.36]`
- cutoff: `Ecm_cutoff = 0.335` unless explicitly overridden
- classifiers: `digonto_v3_window`, `digonto_v4_window` only

The completed CPU determinant grids are already available in
`output/v33p_fasttrack_cpu_existing_000_A1m/`. Do not rescan them.

## Gate

The label and scoring gate is complete for both files:

- `input/v33p_user_truezero_labels_L20_000_A1m_E026310_0360.csv`
- `input/v33p_user_truezero_labels_L24_000_A1m_E026310_0360.csv`

Both accepted classifiers satisfy the project acceptance rule in both sectors:
`TP=3`, `FP=0`, `FN=0`, and the level/zero counts match. However, the runtime
audit found that the current FCN scans all 20,000 cached rows for each sector
on every evaluation. `accepted_windows` now removes that work from the FCN
without changing determinant or classifier logic. The first hot-window run
found only 3/4 model roots because the raw-reader convention differed from the
trusted v33h determinant scan. A localized second real/imag swap in the raw
cache loader restored the L20 bracket and the corrected one-FCN run found 4/4.

See `FITTER_RUNTIME_PATH_AUDIT.md`, `FITTER_RUNTIME_PATH_AUDIT_HOT_WINDOWS.md`,
`LEVEL_ZERO_MATCH_REPORT.md`, and
`FITTER_000_A1M_ONE_FCN_HOT_WINDOWS_REPORT.md`.

## First run

1. Count accepted true zeros below `0.335` for each L.
2. Count lattice levels selected under the same cutoff.
3. Confirm the k-th lattice level maps to the k-th accepted zero per L.
4. Run one FCN/chi-square evaluation only.
5. Record cold-start and one-FCN timings.
6. Audit that cache loading occurs outside the FCN.
7. Audit that no 20,000-row determinant scan occurs inside the FCN.
8. Obtain approval before starting a short minimization.

## Hot-window implementation notes

`root_search_mode=accepted_windows` loads accepted true-zero windows once,
recomputes determinants with current K3df parameters only within those rows,
and applies the existing `digonto_v3_window` or `digonto_v4_window` logic.
`fixed_accepted_zeros` was not used; that debug mode would not be valid for
minimization because its roots would not move with K3df parameters. Full-grid
FCN scans remain forbidden. A future GPU backend should implement the same
determinant interface after CPU local-window correctness is established.

The one-FCN path is now logically safe for a short 000_A1m-only minimization,
but no minimization has been run. Do not start it without approval.

Physics formulas and classifier logic remain unchanged.
