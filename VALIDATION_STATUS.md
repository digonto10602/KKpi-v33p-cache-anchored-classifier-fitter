# Validation Status

## Current checkpoint

Status: `PASS_READY_FOR_USER_TRUE_ZERO_LABELING`

## Passed validations

| check | result | evidence |
| --- | --- | --- |
| combined raw cache reader identity | PASS | v33l / v33n handoff reports |
| corrected old-scale determinant scan | PASS | `output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/` |
| CPU OpenMP subset validation | PASS | v33m handoff and summary numbers below |
| corrected fix1 plot | PASS | `output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_corrected_oldscale_det_interactive.png` |
| target parity | PASS | `reports/v33n_final_sanity_target_parity.md` |
| classifier dispatch | PASS | `reports/v33n_final_sanity_classifier_dispatch.md` |
| parameter mask | PASS | `reports/v33n_final_sanity_parameter_mask.md` |
| 10-FCN smoke benchmark | PASS | `reports/v33n_final_sanity_10fcn.md` |

## Key numbers

### Identity and cache reader

- `current/manual F3inv max_abs_diff = 0`
- `current/manual Vsel max_abs_diff = 0`
- `max |det_imag| = 0`

### CPU subset validation

- `max percent_diff_F3inv_frob ~ 3.2005e-05 %`
- `max percent_diff_QC_frob ~ 3.2005e-05 %`
- `max percent_diff_projected_QC_frob ~ 1.3901e-05 %`
- `max percent_diff_scaled_projected_QC ~ 1.3901e-05 %`
- `max stable percent_diff_det_complex ~ 6.0955e-07 %`
- `max projector_reldiff ~ 9.35e-16`

### Gate 7 plot

- rows plotted: `4128`
- raw sign counts: `+1473 / -2655 / 0`
- near-zero threshold: `1e-90`
- near-zero sign counts: `+1464 / -2655 / 9`
- dimension jumps: `1`
- in-segment sign-change brackets: `8`

### Production benchmark

- `model_found = 32/32`
- `chi2 = 0.2543510436151185`
- `classifier_mode = raw_sign_only`
- `repeat = 10`

## Residual risk

- The classifier still needs user-labeled true zeros before algorithm tuning can continue.
- L25 non-interacting coverage is still partial.
