# v33o Minuit warning summary

## Status by cutoff

| Ecm_cutoff | status | minuit_status | warnings |
| --- | --- | --- | --- |
| 0.315 | CONVERGED | CONVERGED_WITH_MINUIT_WARNING | [Error] VariableMetricBuilder Initial matrix not pos.def. |
| 0.325 | CONVERGED | CONVERGED_WITH_MINUIT_WARNING | [Error] VariableMetricBuilder Initial matrix not pos.def. |
| 0.335 | CONVERGED | CONVERGED_WITH_MINUIT_WARNING | [Error] VariableMetricBuilder Initial matrix not pos.def. |
| 0.345 | CONVERGED | CONVERGED | none |
| 0.355 | CONVERGED | CONVERGED | none |
| 0.365 | CONVERGED | CONVERGED | none |

## Interpretation

- 0.315, 0.325, 0.335: converged with `VariableMetricBuilder Initial matrix not pos.def.` warning.
- 0.345, 0.355, 0.365: converged without that warning.
- The warning is documented, not treated as a hard failure, because the fits converged and wrote final summaries.
