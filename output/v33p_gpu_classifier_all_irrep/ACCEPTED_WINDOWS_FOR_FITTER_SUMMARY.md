# Accepted windows for fitter

Status: L20/110_A2 is accepted and count-matched. L24/100_A2 is excluded because both accepted modes have FP=1.

## L20/110_A2

- Accepted modes: `digonto_v3_window`, `digonto_v4_window`
- Accepted true zeros under Ecm_cutoff=0.335: 4
- Lattice levels under cutoff: 4
- Count match: **yes**
- Lattice Ecm: 0.3073332928312638, 0.31643789359657837, 0.31813597720593867, 0.3291548445548671
- Zero estimates: 0.30679110096539869, 0.31583037429108624, 0.31915497052908925, 0.3287180485505124
- Window configuration: initial half-width 50 rows; maximum half-width 250 rows; bracket rows included.
- Window file: `output/v33p_gpu_classifier_all_irrep/accepted_windows/L20_110_A2/accepted_windows.csv`
- Fitter accepted-zero source: `output/v33p_gpu_classifier_all_irrep/sectors/L20_110_A2/accepted_truezeros_L20_110_A2_E026310_0360.csv`

## Excluded sector

- L24/100_A2: not included; score was TP=9, FP=1, FN=0, TN=42 for both accepted modes.
