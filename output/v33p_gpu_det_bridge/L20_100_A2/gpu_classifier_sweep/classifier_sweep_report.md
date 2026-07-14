# V33P classifier sweep

Status: PASS for the corrected determinant-grid restart harness.

- Grid: corrected `fix1` CSV; dimension-jump crossings excluded.
- In-segment sign-change candidates: 8.
- Modes: all seven discovered dispatch modes.
- Matrix/cache modes reuse only matching v33l corrected-cache diagnostic rows; no cache binary was copied or rebuilt.
- No classifier logic was changed.

Outputs: `classifier_predictions.csv`, `classifier_summary.csv`.
