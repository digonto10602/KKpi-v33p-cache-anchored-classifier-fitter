# Reproducibility

## Environment

- package root: `packaging/KKpi_v33o_production_fitter`
- external cache root: `/media/digonto/Data/F3inv_cache`
- target block: `Lbyas=20`, `irrep=100_A2`, `Ecm=[0.34, 0.36]`, `coarseN=20000`, `xi=3.444`

## Required source inputs

- source tree in this repo
- configs in `configs/`
- validation reports in `reports/`
- small diagnostics in `diagnostics/`

## Not included

- `bin/`
- `build/`
- `cache/`
- `output*/`
- `plots/`
- large `.bin` cache files

## Recreate the package checkpoint

1. Clone the GitHub repo.
2. Mount or copy the external cache root to `/media/digonto/Data/F3inv_cache`, or update the scripts to point at the equivalent location.
3. Build the fitter:

```bash
bash scripts/compile_v33g_all.sh
```

4. Verify target parity and dispatch:

```bash
python3 scripts/validate_fitter_target_counts_v33k.py
python3 scripts/verify_classifier_dispatch_v33m.py
```

5. Run the benchmark smoke test:

```bash
stdbuf -oL -eL bin/v33f_k3df_fitter_multiL_v33e configs/config_v33m_multiL_all_irreps_fastest_production.in benchmark-fcn --repeat 10 --warmup 2
```

6. Recreate the corrected plot if needed:

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

## Validation record

The packaged checkpoint is the one that produced the reports in `reports/` and the fix1 artifacts in `output/v33m_resume_gates5_7_before_classifier/`.
