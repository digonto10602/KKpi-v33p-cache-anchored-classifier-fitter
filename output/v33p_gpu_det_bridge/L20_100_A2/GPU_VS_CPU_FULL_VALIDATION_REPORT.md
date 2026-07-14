# CPU-assembled corrected-reader GPU determinant bridge

Status: FULL BRIDGE RUN

- cache: `/media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_100_A2/v32zu_Lbyas20_gpu_cache/v32zu_Lbyas20_xi3p444_irrep100_A2_coarse20000_F3inv_Vsel_gpu.bin`
- sector: Lbyas=20, irrep=100_A2
- reader: corrected `v33h` raw reader, `variant_04_real_imag_swapped`
- assembly: reused `compute_row` with `QC = F3inv + K3df`, `Vsel.adjoint() * QC * Vsel`, and division by `pow(Lbyas * xi, 6)`
- GPU LU API: `cublasZgetrfBatched`; determinant is pivot parity times the LU diagonal product
- rows processed: 20000
- CPU fallback rows: 0
- max CPU-assembled vs trusted CSV determinant difference: 0
- max GPU vs CPU-assembled determinant difference: 1.1892064646764559e-17
- max reported GPU allocation bytes: 7440384
- CPU assembly seconds: 4641.6017273859998
- GPU determinant seconds: 0.73331562600000033

No cachegen, cache regeneration, cache copying, classifier change, or physics-formula change was performed.

## Full-grid identity checks

- CPU reference rows: `20000`; GPU bridge rows: `20000`.
- Ecm grid: exact match; maximum absolute Ecm difference `0`.
- Nfull/Nproj sequence: exact match.
- Dimension jumps: exact match; `11` jumps.
- Same-dimension sign-change brackets: exact match; `25` brackets.
- Maximum absolute determinant difference: `1.1892064646764559e-17`.
- Maximum relative determinant difference: `7.162738729525389e-05`, occurring at a near-zero determinant; absolute error is the controlling metric there. Median relative difference was `1.703514543334017e-14`.
- Determinant imaginary-part difference: `0` in the compared CSVs.
- `digonto_v3_window`: same bracket/classification/accepted-zero identity; only sub-ulp decimal formatting differs in two pole zero estimates.
- `digonto_v4_window`: same bracket/classification/accepted-zero identity; only sub-ulp decimal formatting differs in two pole zero estimates.
- Accepted true-zero energies: `0.29155609987916675` and `0.30044149754668775` for both accepted modes.
- CPU fallback count: `0`.
- Maximum bridge GPU allocation reported by the CUDA helper: `7,440,384` bytes.
- Full bridge wall time: `1:17:22`; CPU assembly `4641.6017 s`; GPU determinant calls `0.7333 s`; peak RSS `567004 kB`.
- A prior wall-clock runtime for the trusted CPU reference scan was not recorded in its lightweight artifact, so no unsupported CPU-vs-GPU speedup claim is made.

Conclusion: **PASS** for L20/100_A2 reference validation. The GPU determinant output is trusted for subsequent classifier validation, subject to the same corrected raw-reader and physics-path invariants.
