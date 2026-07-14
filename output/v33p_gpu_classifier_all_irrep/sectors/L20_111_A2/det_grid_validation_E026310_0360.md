# Determinant-grid validation: L=20, irrep=111_A2

Status: PASS; finalized scan loaded from the existing cache.

- Cache metadata: `/media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_111_A2/v32zu_Lbyas20_gpu_cache/v32zu_Lbyas20_xi3p444_irrep111_A2_coarse20000_F3inv_Vsel_gpu.bin.meta.json`
- Cache binary: `/media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_111_A2/v32zu_Lbyas20_gpu_cache/v32zu_Lbyas20_xi3p444_irrep111_A2_coarse20000_F3inv_Vsel_gpu.bin`
- Provenance: `v33d_gpu_component_cachegen_all_ingredients`
- Metadata Lbyas/irrep: `20` / `A2`
- Metadata row count/coarseN: `20000` / `20000` scan rows
- Metadata Ecm range: `[0.2631, 0.36]`
- Scan source/regeneration path: `bin/v33h_patched_gpu_cache_oldscale_det_scan --gpu-cache-root /media/digonto/Data/F3inv_cache/v33e_coarse20000_sep_f2k2_L20_111_A2/v32zu_Lbyas20_gpu_cache --Emin 0.2631 --Emax 0.36 --coarseN 20000 --old-scaling true --complex-read-convention variant_04_real_imag_swapped --hard-hermiticity-check true`
- Max `|det_imag|`: `0`
- Dimension jumps: `12`
- Sign-change brackets excluding dimension jumps: `30`
- Scan failures: `0`
- Corrected format-aware combined raw GPU cache reader: confirmed.
- Fixed scaling: `scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6.0)`; no determinant-level `Nproj` rescaling.
- Physics path: `QC = F3inv + K3df`, `projected_QC = Vsel.adjoint() * QC * Vsel`, `det = determinant(scaled_projected_QC)`.
- Cachegen run: **no**; no cache binary was regenerated or copied.

## Dimension jumps

- rows 5287->5288: Ecm 2.88716795839791984e-01->2.88721641082054115e-01, dimensions (40,14) -> (43,15)
- rows 7725->7726: Ecm 3.00529496474823743e-01->3.00534341717085873e-01, dimensions (43,15) -> (46,16)
- rows 9515->9516: Ecm 3.09202480124006196e-01->3.09207325366268326e-01, dimensions (46,16) -> (52,17)
- rows 10649->10650: Ecm 3.14696984849242467e-01->3.14701830091504542e-01, dimensions (52,17) -> (58,18)
- rows 11410->11411: Ecm 3.18384214210710503e-01->3.18389059452972634e-01, dimensions (58,18) -> (61,19)
- rows 11752->11753: Ecm 3.20041287064353197e-01->3.20046132306615327e-01, dimensions (61,19) -> (64,20)
- rows 12324->12325: Ecm 3.22812765638281929e-01->3.22817610880544004e-01, dimensions (64,20) -> (76,23)
- rows 14225->14226: Ecm 3.32023571178558907e-01->3.32028416420821038e-01, dimensions (76,23) -> (88,26)
- rows 15499->15500: Ecm 3.38196409820491040e-01->3.38201255062753114e-01, dimensions (88,26) -> (112,30)
- rows 16456->16457: Ecm 3.42833306665333293e-01->3.42838151907595368e-01, dimensions (112,30) -> (136,34)
- rows 17022->17023: Ecm 3.45575713785689298e-01->3.45580559027951373e-01, dimensions (136,34) -> (148,37)
- rows 17318->17319: Ecm 3.47009905495274773e-01->3.47014750737536848e-01, dimensions (148,37) -> (160,40)
