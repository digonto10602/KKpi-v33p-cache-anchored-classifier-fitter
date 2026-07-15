# V33P scatter-parameter hard stop

Status: **AUDIT COMPLETE; HARD STOP CLEARED FOR THE LIMITED VERIFIED TEST ONLY**

The unsafe continuation was held while the active production configuration path was audited. No fitter, minimization, new classifier validation, or bracket-review conclusion was run during the audit.

## Findings

- Wrong comparison values are present in archival configs including:
  - `configs/config_v31zw_110A2_waves01_scatter1p010.in`: `scatter1_10 = 0.10`, `scatter1_00 = 0.01`, `scatter2_00 = 0.01`.
  - `configs/config_v31zw_110A2_waves01_scatter1p100.in`: `scatter1_10 = 1.00`, `scatter1_00 = 0.01`, `scatter2_00 = 0.01`.
  - matching v32/v32b comparison configs and older v31z scatter-comparison configs contain analogous non-production values.
- The old `run_v31zw_110A2_compare_waves01_p010_vs_p100_digonto_classifier_v1.sh` explicitly references those comparison configs.
- No current v33p fitter script, v33p generated fitter input, v33f/v33g/v33m/v33n production config, or current v33p validation command references those old comparison configs.
- The existing v33p one-FCN attempts used `output/v33p_gpu_classifier_all_irrep/fitter/fitter_classifier_accepted_sectors.in`, whose audited values are production-correct; both attempts stopped before FCN evaluation by signal 9.

## Guard result

`scripts/check_v33p_production_scatter_params.py` checked 31 active production configs and found zero active failures. The full table is in `SCATTER_PARAM_AUDIT_TABLE.csv`; the executable result is in `SCATTER_PARAM_GUARD_RESULT.md`.

## Invariants held

No cachegen, cache regeneration, determinant scan, minimization, classifier-logic change, or physics-formula change was performed while resolving this hard stop.
