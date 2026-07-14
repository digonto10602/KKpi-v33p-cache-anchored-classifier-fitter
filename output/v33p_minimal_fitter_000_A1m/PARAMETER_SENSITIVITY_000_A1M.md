# 000_A1m parameter sensitivity

## Method

The diagnostic reused the existing accepted-window `model_for()` evaluator
after one cache load. It evaluated central, minus-step, and plus-step points
at both the starting point and the best previously logged point. Steps were:

```text
K3iso0: 1000
K3iso1: 10000
K3B:    1000
K3E:    10000
```

All 24 perturbed/central evaluations had model_found `4/4`, evaluated 404
rows, used zero window expansions, and had zero fallback/full scans.

## Summary

| parameter | start max | best max | assessment |
|---|---:|---:|---|
| K3iso0 | `2.46390e-09` Ecm/parameter | `2.43154e-09` | active/moderate |
| K3iso1 | `1.42947e-09` Ecm/parameter | `1.41033e-09` | active; strongest chi2 response |
| K3B | `6.40042e-10` Ecm/parameter | `6.08004e-10` | active/moderate |
| K3E | `4.12690e-10` Ecm/parameter | `4.11118e-10` | weakest/approximately flat relative to others |

The largest chi-square response was from K3iso1. At the starting point, the
plus/minus chi-square values were:

```text
K3iso0: 1.5146614190181035e-05, 1.5105260405290964e-05
K3iso1: 1.5284968448981461e-05, 1.4973792009619562e-05
K3B:    1.5141991608990335e-05, 1.5109782538317566e-05
K3E:    1.5105137325914844e-05, 1.5147824339824970e-05
```

At the best logged point, the same ordering remains: K3iso1 is most active,
K3E is weakest, and K3iso0/K3B are intermediate. Full row-level values,
roots, chi-square changes, and finite differences are in
`PARAMETER_SENSITIVITY_000_A1M.csv`.

No minimization was performed as part of this diagnostic.
