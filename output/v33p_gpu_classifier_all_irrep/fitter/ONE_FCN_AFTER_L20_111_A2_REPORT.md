# One-FCN after L20/111_A2

## Result

`CHECKPOINT_STATUS = L20_111_A2_ACCEPTED_ONE_FCN_PASS`

The accepted-window FCN completed for five sectors at the fixed starting
K3df parameters. It found all 12 requested roots and returned finite chi2.

| item | result |
|---|---:|
| sectors | L20/000_A1m, L20/100_A2, L20/110_A2, L20/111_A2, L24/000_A1m |
| excluded | L24/100_A2 (classifier FP=1 at bracket 36) |
| lattice/model levels | 12 / 12 |
| chi2 | 0.0002394162713767611 |
| rows evaluated in FCN | 2214 |
| determinant backend | auto -> cpu_openmp |
| window expansions | 2 |
| fallback/full scan | 0 / 0 |
| FCN time | 0.029772724 s (internal); 0.03801349 s wrapper output |
| cache/window preparation | 944.73168146 s |
| peak RSS | 16,880,084 kB |

Per-sector roots were 1/1, 2/2, 4/4, 2/2, and 3/3 in the order listed above.
No minimization was run. The FCN used the corrected cache-reader path and
accepted hot windows only; no 20,000-row determinant scan occurred inside the
FCN and `fallback_full_scan=0`.

## Command

```bash
./output/v33p_gpu_classifier_all_irrep/fitter/run_one_fcn_after_L20_111_A2.sh
```

The wrapper invoked:

```bash
/usr/bin/time -v stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e \
  output/v33p_gpu_classifier_all_irrep/fitter/fitter_after_L20_111_A2.in fcn-once
```

## Physics/classifier guard

The run used the existing QC, projection, `pow(Lbyas*xi,6)` scaling, and raw
determinant definitions. Classifier logic was unchanged. L24/100_A2 remains
excluded and no cachegen, cache regeneration, minimization, or GPU scanner was
run.
