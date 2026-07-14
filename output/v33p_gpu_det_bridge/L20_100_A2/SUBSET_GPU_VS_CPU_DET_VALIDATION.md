# CPU-assembled corrected-reader GPU determinant bridge

Status: SUBSET RUN

- cache: `/media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_100_A2/v32zu_Lbyas20_gpu_cache/v32zu_Lbyas20_xi3p444_irrep100_A2_coarse20000_F3inv_Vsel_gpu.bin`
- sector: Lbyas=20, irrep=100_A2
- reader: corrected `v33h` raw reader, `variant_04_real_imag_swapped`
- assembly: reused `compute_row` with `QC = F3inv + K3df`, `Vsel.adjoint() * QC * Vsel`, and division by `pow(Lbyas * xi, 6)`
- GPU LU API: `cublasZgetrfBatched`; determinant is pivot parity times the LU diagonal product
- rows processed: 230
- CPU fallback rows: 0
- max CPU-assembled vs trusted CSV determinant difference: 0
- max GPU vs CPU-assembled determinant difference: 1.1892064646764559e-17
- max reported GPU allocation bytes: 133504
- CPU assembly seconds: 301.85738926200003
- GPU determinant seconds: 0.14023581399999999

No cachegen, cache regeneration, cache copying, classifier change, or physics-formula change was performed.
