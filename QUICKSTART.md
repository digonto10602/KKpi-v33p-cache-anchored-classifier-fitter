# Quickstart

```bash
cd /path/to/kkpi-v33o-production-fitter
bash scripts/compile_v33g_all.sh
python3 scripts/validate_fitter_target_counts_v33k.py
stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e configs/config_v33m_multiL_all_irreps_fastest_production.in benchmark-fcn --repeat 10 --warmup 2
python3 scripts/run_v33n_cutoff_scan_fits.py
```

For the checkpoint-specific plot restart, use:

```bash
python3 -u scripts/plot_oldscale_det_coarse20000_interactive.py \
  --csv output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_oldscale_det_coarse20000_E034_036.csv \
  --title "L20 100_A2 old-scale det(scaled projected QC), corrected GPU reader, coarseN=20000" \
  --Emin 0.34 \
  --Emax 0.36 \
  --mark-dimension-jumps \
  --mark-sign-changes \
  --show-imag \
  --save-prefix output/v33m_resume_gates5_7_before_classifier/oldscale_det_scan_L20_100_A2_fix1/L20_100_A2_corrected_oldscale_det_interactive
```

## Expected checkpoint result

- `max |det_imag| = 0`
- raw sign counts `+1473 / -2655 / 0`
- dimension jump count `1`
- sign-change bracket count `8`
