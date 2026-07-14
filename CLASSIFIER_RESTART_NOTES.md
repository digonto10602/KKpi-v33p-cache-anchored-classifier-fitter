# Classifier Restart Notes

## Current state

The classifier is parked after the corrected `fix1` plot and is waiting for user-labeled true zeros.

## Input used

- CSV: `output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_oldscale_det_coarse20000_E034_036.csv`
- Plot PNG: `output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_corrected_oldscale_det_interactive.png`
- Plot PDF: `output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_corrected_oldscale_det_interactive.pdf`

## Candidate brackets

Eight sign-change brackets were found in the constant-dimension segment:

1. `0.340570578529 -> 0.340575423771`
2. `0.342368163408 -> 0.342373008650`
3. `0.346864548227 -> 0.346869393470`
4. `0.347780299015 -> 0.347785144257`
5. `0.350701980099 -> 0.350706825341`
6. `0.354583019151 -> 0.354587864393`
7. `0.356865128256 -> 0.356869973499`
8. `0.357407795390 -> 0.357412640632`

## Dimension jump

- row `22 -> 23`
- `Nfull/Nproj 104/29 -> 108/30`
- `Ecm 0.3401102805140257 -> 0.3401151257562878`

## Rule to keep

Do not classify a sign change across a dimension jump as a normal zero bracket.

## Files to inspect next

- `output/v33n_classifier_restart_from_fix1_L20_100_A2/candidate_sign_change_brackets.csv`
- `output/v33n_classifier_restart_from_fix1_L20_100_A2/determinant_segment_summary.csv`
- `output/v33n_classifier_restart_from_fix1_L20_100_A2/classifier_restart_context.md`
