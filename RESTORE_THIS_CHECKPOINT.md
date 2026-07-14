# Restore this checkpoint

1. Clone the public repository and enter it.
2. Confirm `/media/digonto/Data/F3inv_cache/` is mounted or set the equivalent
   cache-root option; do not copy cache binaries into the repository.
3. Build with `bash scripts/compile_v33g_all.sh` after checking local CUDA,
   Eigen, OpenMP, cuBLAS, and cuSOLVER paths.
4. Re-run the CSV-only accepted reference sweep using the lightweight grid and
   reports under `output/v33p_classifier_restart_L20_100_A2_E026310_0360/`.
5. Resume sector work from actual `*.bin.meta.json` windows, expected
   `[0.26310,0.36]`, and the corrected reader.
6. Fill user-label templates before scoring any new sector.
7. Do not run cachegen, copy raw cache binaries, alter classifier logic, or
   change the determinant formulas.

The accepted reference is L20/100_A2, with both accepted modes selecting only
brackets 7 and 8 and scores `TP=2 FP=0 FN=0 TN=23`.
