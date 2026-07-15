# One-FCN after scatter audit

Status: **PASS**

## Command

```bash
output/v33p_scatter_param_audit/run_one_fcn_after_scatter_fix.sh > output/v33p_scatter_param_audit/one_fcn_after_scatter_fix.log 2>&1
```

The wrapper used `output/v33p_gpu_classifier_all_irrep/fitter/fitter_classifier_accepted_sectors.in` and `fcn-once`.

## Result

- Sectors: `L20/000_A1m`, `L20/100_A2`, `L20/110_A2`, `L24/000_A1m`.
- Levels: `10` total.
- Model roots: `10/10`.
- Chi-square: `0.0002078244109677272`.
- FCN time: `0.027263653 s`.
- Cache load/window preparation: `741.317397861 s`.
- Rows evaluated inside FCN: `2012`.
- Window expansions: `2`.
- Fallback/full scan: `0/0`.
- Peak RSS: `13,662,732 kB`.
- Exit status: `0`.

## Scatter/config audit relationship

The active config passed the production scatter guard. The only additional correction required to reach FCN was a schema-only companion for the existing L20/100_A2 accepted-zero CSV: the loader requires `zero_estimate` and `inside_Ecm_cutoff`. No physics formula, scatter value, classifier logic, cache, or determinant grid was changed.

No minimization, cachegen, cache regeneration, or new classifier validation was run.
