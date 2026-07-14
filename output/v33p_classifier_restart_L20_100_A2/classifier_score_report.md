# V33P classifier scores

Labels loaded: 8

| method | TP | FP | FN | TN | score | acceptable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| algo_v6_branch_count_final_select | 0 | 0 | 0 | 8 | 0.0 | YES |
| algo_v7_eigenbranch | 0 | 0 | 0 | 8 | 0.0 | YES |
| algo_v7_hybrid_det_eigenbranch | 0 | 0 | 0 | 8 | 0.0 | YES |
| digonto_v3_window | 0 | 0 | 0 | 8 | 0.0 | YES |
| digonto_v4_window | 0 | 0 | 0 | 8 | 0.0 | YES |
| raw_sign_clustered | 0 | 8 | 0 | 0 | -8.0 | NO |
| raw_sign_only | 0 | 8 | 0 | 0 | -8.0 | NO |

## False positives by bracket

- `raw_sign_only`: bracket IDs `1,2,3,4,5,6,7,8`
- `raw_sign_clustered`: bracket IDs `1,2,3,4,5,6,7,8`
- all other modes: none

## Negative-control interpretation

The classifiers with zero false positives survive this negative-control test:
`algo_v6_branch_count_final_select`, `algo_v7_eigenbranch`,
`algo_v7_hybrid_det_eigenbranch`, `digonto_v3_window`, and
`digonto_v4_window`. This is only a false-positive diagnostic for the
0.34–0.36 window where all eight user labels are false; it does not globally
accept any classifier or establish true-zero recall.
