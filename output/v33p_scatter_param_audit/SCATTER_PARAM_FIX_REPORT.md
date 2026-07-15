# V33P scatter-parameter fix report

Status: **NO ACTIVE PRODUCTION FILE REQUIRED A VALUE CHANGE**

## Active path

All active v33p fitter inputs already contain:

```text
xival = 3.444
atmK = 0.09698
atmpi = 0.06906
eta_1 = 1.0
eta_2 = 0.5
alpha = 0.5
epsilon_h = 0.0
max_shell_num = 20
tolerance = 1e-12
parity = -1
eig_tol = 0.05
norm_tol = 1e-12
proj_tol = 1e-10
waves_vec_1 = 0 1
waves_vec_2 = 0
scatter1_00 = 4.04
scatter1_10 = -43.2
scatter2_00 = 4.12
```

`xival` is the existing config spelling of the required `xi` value.

## Files changed

- No active production config scatter value was changed.
- Added `scripts/check_v33p_production_scatter_params.py`.
- Added the audit CSV and markdown reports in this directory.

The one-FCN smoke path also received a non-physics schema compatibility fix: the existing L20/100_A2 accepted-zero CSV was preserved, a normalized companion with the hot-window loader's required columns was added, and the two active v33p fitter configs were pointed to that companion. This did not change any scatter value, physics formula, classifier rule, cache, or determinant grid.

## Files intentionally not changed

The v31z/v32/v32b comparison configs remain unchanged because they are archival/test workflows, explicitly named as scatter comparisons, and are not referenced by the current v33p production path. Changing them would destroy their historical comparison meaning and would not fix the active path.
