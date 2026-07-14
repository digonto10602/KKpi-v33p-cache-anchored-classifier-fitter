# v33m Production Classifier Benchmark and Fit Full Report

## Executive summary

Chosen classifier: `raw_sign_only`

## Target parity

PASS

## Dispatch audit

PASS

## Correctness comparison

| mode | eligible | model_found | chi2 | assignment_equivalent_to_reference | status |
| --- | --- | --- | ---: | --- | --- |
| raw_sign_only | YES | 32 | 0.2543510436151185 | YES | PASS |
| raw_sign_clustered | YES | 32 | 0.08475390020682615 | NO | COVERAGE_ONLY_NOT_EQUIVALENT |
| algo_v6_branch_count_final_select | YES | 32 | 0.08475390020682615 | NO | COVERAGE_ONLY_NOT_EQUIVALENT |
| algo_v7_eigenbranch | YES | 32 | 0.2543510436151185 | YES | PASS |
| algo_v7_hybrid_det_eigenbranch | YES | 32 | 0.2543510436151185 | YES | PASS |
| digonto_v3_window | YES | 32 | 0.2543510436151185 | YES | PASS |
| digonto_v4_window | YES | 32 | 0.08475390020682615 | NO | COVERAGE_ONLY_NOT_EQUIVALENT |

## 10-FCN benchmark

All seven benchmarked modes reached `32/32` coverage. The three coverage-only modes are marked `NO` in the assignment-equivalence column because they produce a different root assignment / chi2 than the reference `raw_sign_only` path.

| mode | avg_total_sec | median_total_sec | min_total_sec | max_total_sec | assignment_equivalent |
| --- | ---: | ---: | ---: | ---: | --- |
| raw_sign_only | 1.3727941782 | 1.377119312 | 1.320426902 | 1.421347011 | YES |
| algo_v7_hybrid_det_eigenbranch | 1.3810405386 | 1.3873218965 | 1.347030847 | 1.412480806 | YES |
| algo_v7_eigenbranch | 1.3831377883000002 | 1.3891617625000001 | 1.355646395 | 1.406073163 | YES |
| raw_sign_clustered | 1.3952508624 | 1.3920239955 | 1.323797044 | 1.465788268 | NO |
| digonto_v3_window | 1.397978053 | 1.394144606 | 1.36268128 | 1.429787946 | YES |
| digonto_v4_window | 1.4193617075 | 1.42017048 | 1.35423242 | 1.50558759 | NO |
| algo_v6_branch_count_final_select | 1.4297087228999998 | 1.4433650165 | 1.35416321 | 1.476740523 | NO |

## Full fit

completed: YES
final chi2: 0.25174650417937361
model_found: 32/32
full fit report: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/reports/v33m_full_fit_report.md
production config: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/configs/config_v33m_multiL_all_irreps_fastest_production.in
