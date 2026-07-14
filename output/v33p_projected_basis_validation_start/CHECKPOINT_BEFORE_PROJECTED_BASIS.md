# Checkpoint before projected-basis validation

Status: `GPU_DET_BACKEND_REFERENCE_VALIDATION_PASS`.

- Existing public branch: `hot-window-fcn-pass`.
- Current raw GPU bridge: `bin/v33p_cpuassembled_gpu_det_scan`.
- L20/100_A2 raw determinant validation remains 20,000-row PASS with zero fallback and identical sign-change/classifier identities.
- Lightweight checks passed: bridge `--help`; generated GPU grid has exactly 20,000 data rows.
- The parent worktree contains unrelated user files and generated outputs; they were not staged.
- This checkpoint contains only the bridge source, compact reports/diagnostics, and validation scripts. Cache binaries, full determinant grids, logs, and build artifacts are excluded.

The next phase is the bounded projected-basis reconstruction prototype.
