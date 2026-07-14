# CPU assembly bridge audit: L20/100_A2

Status: PASS.

- Reused source: `source/v33h_patched_gpu_cache_oldscale_det_scan.cpp`.
- Reused functions: `validate_metadata`, `scan_gpu_cache`, `select_window_rows`, `load_gpu_row_direct_variant04`, `compute_row`, `make_physics`, and `make_settings`.
- Reader: corrected combined raw `F3inv/Vsel` reader with `variant_04_real_imag_swapped`.
- Physics path: `QC = F3inv + K3df`; `projected_QC = Vsel.adjoint() * QC * Vsel`; `scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6)`; determinant unchanged.
- Matrix storage: Eigen column-major `MatrixXcd`, packed as column-major interleaved complex-double buffers for CUDA.
- Retained row metadata: raw row index, sorted row index, Ecm, Nfull, Nproj, all existing sanity fields, and CPU determinant.
- CPU assembly matched the trusted determinant CSV exactly on the 230-row subset and all 20,000 full rows.

The bridge does not move reader, projection, scaling, or K3df assembly onto the GPU.
