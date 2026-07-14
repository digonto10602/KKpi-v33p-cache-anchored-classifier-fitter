# Accepted classifier decision: L20 100_A2

## Decision

`digonto_v3_window` and `digonto_v4_window` are accepted for this labeled
sector.

- Lbyas: `20`
- Irrep: `100_A2`
- Cache-anchored Ecm range: `[0.2631, 0.36]`
- TP/FP/FN/TN: `2/0/0/23`
- Precision/recall/F1: `1.0/1.0/1.0`
- Accepted true-zero brackets: `7` and `8`

Acceptance is limited to `Lbyas=20`, `irrep=100_A2`, cache-anchored range
`[0.2631,0.36]`. Global production readiness still requires validation over
all L/irreps.

## Accepted energies

| bracket_id | E_left | E_right | E_zero_linear |
|---:|---:|---:|---:|
| 7 | 0.291551262563 | 0.291556107805 | 0.291556099879 |
| 8 | 0.300437436872 | 0.300442282114 | 0.300441497547 |

## Basis for decision

- User labels were complete for all 25 candidates.
- Both accepted modes predicted only brackets 7 and 8.
- No classifier logic, physics formula, cache binary, or corrected fix1 output
  was changed in this packaging step.
