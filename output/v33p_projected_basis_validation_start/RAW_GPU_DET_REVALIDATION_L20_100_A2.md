# Raw GPU determinant revalidation: L20/100_A2

Status: `RAW_GPU_DET_REVALIDATION_PASS`.

Inputs:

- CPU reference: `output/v33p_classifier_restart_L20_100_A2_E026310_0360/det_grid_L20_100_A2_corrected_E026310_0360.csv`
- GPU bridge output: `output/v33p_gpu_det_bridge/L20_100_A2/gpu_det_grid_L20_100_A2_E026310_0360.csv`
- Bridge: `bin/v33p_cpuassembled_gpu_det_scan`

Results over 20,000 rows:

- `det_real` maximum absolute difference: `1.1892064646764559e-17`.
- `det_imag` maximum absolute difference: `0`.
- Relative-difference denominator: `max(abs(det_cpu), 1e-300)`.
- Maximum relative difference: `7.162738729525389e-05`.
- Median relative difference: `1.703514543334017e-14`.
- Validation-only maximum `log(abs(det))` difference: `7.162482219769117e-05`.
- Rows with `abs(det_cpu) < 1e-30`: `17,517`.
- Sign agreement: `20,000/20,000`.
- Same-dimension sign-change brackets: `25/25` identical.
- `digonto_v3_window` identity: PASS.
- `digonto_v4_window` identity: PASS.
- Accepted-zero identity: PASS; `0.29155609987916675` and `0.30044149754668775` for both modes.
- GPU fallback count: `0`.

Conclusion: tiny determinants require absolute/sign/bracket checks rather than
relative error alone. Those checks pass, so the raw determinant remains the
primary trusted quantity. `logabs` was used only as a diagnostic.
