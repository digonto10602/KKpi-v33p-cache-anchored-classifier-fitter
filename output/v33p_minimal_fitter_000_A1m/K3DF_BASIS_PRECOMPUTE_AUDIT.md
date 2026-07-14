# K3df basis precompute audit

Status: **local-window precompute implemented using the existing basis path**.

`source/K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp` already
provides `precompute_projected_k3_basis` and `assemble_QC`. The raw-cache load
path calls the precompute routine for successful cache entries. `assemble_QC`
then forms the current-parameter matrix as:

```text
F3inv_proj + K3iso0*K3_proj_basis[0]
             + K3iso1*K3_proj_basis[1]
             + K3B*K3_proj_basis[2]
             + K3E*K3_proj_basis[3]
```

The hot-window path reuses this exact function, so the K3df combination is
updated for every FCN while the basis is not rebuilt. Physics formulas and
scaling are unchanged. The raw reader no longer builds this basis for every
20,000-row cache entry; the constructor precomputes it only for maximum-window
rows.

For this test, the initial window set contains 404 rows across four windows;
the maximum configured set is 2004 rows. The one-FCN execution evaluated 1406
rows after two local expansions (602 for L20 and 804 for L24). A precise
memory estimate depends on per-row projected dimensions, but the new path does
not allocate a second full-grid basis representation.

Recommendation: keep this existing basis path for the one-FCN test. Optimize
storage or GPU batching only after hot-window correctness is validated.
