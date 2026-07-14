# v33n Production Sanity Cutoff Scan Full Report

## Goal

Run the v33n production sanity cutoff scan, validate the final pre-production setup, and write a restart-ready report for the converged cutoff set.

## Executive summary

Final sanity: PASS
Target parity: PASS
Classifier dispatch: PASS
Parameter mask: PASS
Warm 10-FCN benchmark avg: 1.4042848578000002

## Files changed

- `source/qc_fitter_norm_refine_v2_multiL.cpp`
- `scripts/run_v33n_cutoff_scan_fits.py`
- `reports/v33n_final_sanity_target_parity.md`
- `reports/v33n_final_sanity_classifier_dispatch.md`
- `reports/v33n_final_sanity_parameter_mask.md`
- `reports/v33n_final_sanity_10fcn.md`
- `reports/v33n_cutoff_scan/*.md`
- `reports/v33n_production_sanity_cutoff_scan_full_report.md`
- `START_HERE_v33n_production_sanity_cutoff_scan_handoff.md`

## Commands run

```bash
bash scripts/compile_v33g_all.sh
python3 -m py_compile scripts/run_v33n_cutoff_scan_fits.py
python3 scripts/run_v33n_cutoff_scan_fits.py
```

Follow-up inspection commands were used to confirm the cutoff outputs and final reports on disk:

```bash
ps -ef | grep -E 'v33f_k3df_fitter_multiL_v33e|run_v33n_cutoff_scan_fits.py'
find output_v33n/cutoff_scan -maxdepth 2 -type f
find plots/v33n_cutoff_scan -maxdepth 2 -type f
find reports/v33n_cutoff_scan -maxdepth 1 -type f
```

## Final pre-production sanity results

- target parity report: `reports/v33n_final_sanity_target_parity.md`
- classifier dispatch report: `reports/v33n_final_sanity_classifier_dispatch.md`
- parameter mask report: `reports/v33n_final_sanity_parameter_mask.md`
- 10-FCN report: `reports/v33n_final_sanity_10fcn.md`

## Target-loader parity result

PASS

## Classifier dispatch result

PASS

## Active/free parameter mask

PASS, all four K3df parameters are floating in the production config.

## Warm 10-FCN timing confirmation

avg_total_sec: 1.4042848578000002
median_total_sec: 1.411211705
model_found: 32/32

## Cutoff list attempted

0.315, 0.325, 0.335, 0.345, 0.355, 0.365

## Target availability by cutoff

- 0.315: 18 targets, status=CONVERGED
- 0.325: 24 targets, status=CONVERGED
- 0.335: 32 targets, status=CONVERGED
- 0.345: 36 targets, status=CONVERGED
- 0.355: 39 targets, status=CONVERGED
- 0.365: 40 targets, status=CONVERGED

## Fit status by cutoff

- 0.315: CONVERGED, seed=v33m_initial, final chi2=0.059420738952681296
- 0.325: CONVERGED, seed=v33m_initial, final chi2=0.06908119832574422
- 0.335: CONVERGED, seed=v33m_initial, final chi2=0.2517465041793736
- 0.345: CONVERGED, seed=v33m_initial, final chi2=0.6194461995213746
- 0.355: CONVERGED, seed=v33m_initial, final chi2=0.6631421334634692
- 0.365: CONVERGED, seed=v33m_initial, final chi2=1.605769267817991

## Cutoff summary

- `reports/v33n_cutoff_scan/cutoff_scan_summary.md`
- `diagnostics/v33n/cutoff_scan/cutoff_scan_summary.csv`
- `diagnostics/v33n/cutoff_scan/cutoff_scan_summary.json`

## Per-cutoff outputs

Each converged cutoff wrote the following bundles:

- `output_v33n/cutoff_scan/Ecmcut_0p315/`
- `output_v33n/cutoff_scan/Ecmcut_0p325/`
- `output_v33n/cutoff_scan/Ecmcut_0p335/`
- `output_v33n/cutoff_scan/Ecmcut_0p345/`
- `output_v33n/cutoff_scan/Ecmcut_0p355/`
- `output_v33n/cutoff_scan/Ecmcut_0p365/`

Each bundle includes:

- `qc_spectrum_converged_K3df.csv`
- `qc_spectrum_converged_K3df.dat`
- `qc_spectrum_converged_K3df.json`
- `lattice_spectrum_used.csv`
- `lattice_spectrum_used.json`
- `noninteracting_levels_L19_to_L25.csv`
- `fit_run/v33n_Ecmcut_*_fit_summary_allL.dat`
- `spectrum_check/v33n_Ecmcut_*_fit_summary_allL.dat`
- `plots/v33n_cutoff_scan/Ecmcut_*/spectrum_Ecm_vs_Lbyas_all_irreps.png`
- `plots/v33n_cutoff_scan/Ecmcut_*/spectrum_Ecm_vs_Lbyas_all_irreps.pdf`
- determinant plots under `plots/v33n_cutoff_scan/Ecmcut_*/`

## Converged K3df parameters by cutoff

- 0.315: K3iso0=60236.19647511613, K3iso1=-1035173.9140440958, K3B=253921.14757953485, K3E=-961680.9171168563
- 0.325: K3iso0=60136.14628532966, K3iso1=-1035173.9140440958, K3B=253921.14757953485, K3E=-961680.9171168563
- 0.335: K3iso0=60243.849867119694, K3iso1=-1035173.9140440958, K3B=253921.14757953485, K3E=-961680.9171168563
- 0.345: K3iso0=29067.159771230767, K3iso1=-1007464.2949725426, K3B=260656.195579606, K3E=-917434.4968605285
- 0.355: K3iso0=28317.15899030968, K3iso1=-1035173.9103064314, K3B=253921.15131713764, K3E=-961643.0506003336
- 0.365: K3iso0=28316.860050936837, K3iso1=-1035173.9137547816, K3B=253921.14786884867, K3E=-961670.1295368643

## QC spectrum output files by cutoff

- 0.315: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p315/qc_spectrum_converged_K3df.csv
- 0.325: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p325/qc_spectrum_converged_K3df.csv
- 0.335: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p335/qc_spectrum_converged_K3df.csv
- 0.345: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p345/qc_spectrum_converged_K3df.csv
- 0.355: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p355/qc_spectrum_converged_K3df.csv
- 0.365: /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/output_v33n/cutoff_scan/Ecmcut_0p365/qc_spectrum_converged_K3df.csv

## Spectrum plot index

- /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/plots/v33n_cutoff_scan

## Determinant plot index

- /home/digonto/Codes/Practical_Lattice_v2/3body_quantization_v3/main_fitter/v33f_k3df_v33e_cache_classifier_v3_package/plots/v33n_cutoff_scan


## Parameter-vs-cutoff and chi2-vs-cutoff discussion

The cutoff scan keeps the same raw-sign classifier and shows how the fitted K3df parameters move with the energy window. The most stable cutoff range is the one where the fit converges without changing the selected target count.

## Minuit warnings and convergence notes

- 0.315: [Error] VariableMetricBuilder Initial matrix not pos.def.
- 0.325: [Error] VariableMetricBuilder Initial matrix not pos.def.
- 0.335: [Error] VariableMetricBuilder Initial matrix not pos.def.
- 0.345: none
- 0.355: none
- 0.365: none

## Exact commands to reproduce

```bash
python3 scripts/run_v33n_cutoff_scan_fits.py
```

## Remaining risks

- L25 non-interacting cache rows are not present in the current recent non-interacting cache.
- Minuit warnings may still appear even when the fit converges.
- `fit_wall_time_sec` is recorded as `0.0` in the cutoff summary, so that field should not be used as a performance metric.

## Recommended next command

python3 scripts/run_v33n_cutoff_scan_fits.py
