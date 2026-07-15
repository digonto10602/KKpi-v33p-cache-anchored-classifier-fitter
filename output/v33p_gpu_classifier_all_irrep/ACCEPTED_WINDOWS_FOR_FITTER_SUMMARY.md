# Accepted windows for fitter

Status: L20/110_A2 and L20/111_A2 are classifier-accepted and count-matched. L24/100_A2 has a count-matched manual window set but remains a classifier FAIL because both accepted modes have FP=1.

The accepted-sector one-FCN was attempted and was killed by signal 9 during
fitter startup/cache preparation before any FCN evaluation.

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

- L24/100_A2: manual windows are available below, but classifier score remains TP=9, FP=1, FN=0, TN=42 for both accepted modes.

## L24/100_A2 manual windows

- User-true zeros under cutoff: 2
- Lattice levels under cutoff: 2
- Count match: **yes**
- Window file: `output/v33p_gpu_classifier_all_irrep/accepted_windows/L24_100_A2/accepted_windows.csv`
- Provenance: `MANUAL_LABEL_ACCEPTED`; bracket 36 remains excluded and the user label was not edited.

## L20/111_A2

- Accepted modes: `digonto_v3_window`, `digonto_v4_window`
- Score: TP=5, FP=0, FN=0, TN=25 for both modes
- Accepted true zeros under Ecm_cutoff=0.335: 2
- Lattice levels under cutoff: 2
- Count match: **yes**
- Lattice Ecm: 0.32035801348805054, 0.33413173149593406
- Zero estimates: 0.31942184564777087, 0.33433216652672620
- Window configuration: initial half-width 50 rows; maximum half-width 250 rows; bracket rows included.
- Window file: `output/v33p_gpu_classifier_all_irrep/accepted_windows/L20_111_A2/accepted_windows.csv`
- Fitter accepted-zero source: `output/v33p_gpu_classifier_all_irrep/sectors/L20_111_A2/accepted_truezeros_L20_111_A2_E026310_0360.csv`
