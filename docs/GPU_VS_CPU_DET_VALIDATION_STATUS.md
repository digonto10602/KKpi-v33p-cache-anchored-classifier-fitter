# GPU versus CPU determinant validation

Status: **NOT PASSED / NOT USED**.

The accepted CPU reference is:

```text
output/v33p_classifier_restart_L20_100_A2_E026310_0360/det_grid_L20_100_A2_corrected_E026310_0360.csv
```

The existing CUDA/cuBLAS/cuSOLVER experiments were audited but do not consume
the corrected combined raw reader and therefore were not compared as if they
were equivalent. No GPU determinant CSV was generated, no GPU result was used
for classifier acceptance, and no fitter validation was run from the unsafe
backend.

Required gate before use:

- equal row count and Ecm grid
- equal dimensions and dimension jumps
- justified double-precision determinant differences
- equal determinant imaginary behavior
- identical sign-change brackets
- identical accepted classifier predictions
- runtime, VRAM, and CPU-fallback accounting
