# GPU classifier validation summary

Status: `L20_111_A2_LABELS_SCORED_PASS`.

| sector | rows | candidates | v3/v4 predictions | score status | fitter status |
|---|---:|---:|---:|---|---|
| L20/000_A1m | existing accepted | existing | existing | PASS | available |
| L24/000_A1m | existing accepted | existing | existing | PASS | available |
| L20/100_A2 | existing accepted | existing | existing | PASS | available |
| L24/100_A2 | 20000 | 52 | 10 / 10 true predictions | FAIL: TP=9 FP=1 FN=0 TN=42 | manual windows; one-FCN startup blocked |
| L20/110_A2 | 20000 | 36 | 4 / 4 true predictions | PASS: TP=4 FP=0 FN=0 TN=32 | accepted window ready |
| L20/111_A2 | 20000 | 30 | 5 / 5 true predictions | PASS: TP=5 FP=0 FN=0 TN=25 | accepted window ready (2/2 count match) |

L24/100_A2 false-positive bracket: `bracket_id=36` (both v3 and v4).

L20/110_A2 accepted zero estimates under `Ecm_cutoff=0.335`:
`0.30679110096539869`, `0.31583037429108624`,
`0.31915497052908925`, `0.3287180485505124`.

All new grids reused completed corrected raw determinant outputs. No cachegen,
cache regeneration, or determinant scan was run in this task. Raw determinant
remains the classifier quantity; logabs/logdet were not used.

Remaining label work: L20/200_A2 and the later incomplete sectors; L24/100_A2 remains excluded after its FP=1 classifier result.
Current fitter status: the previous four-sector run passed 10/10 roots; the next one-FCN run will add L20/111_A2.
