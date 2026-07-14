# Incomplete-irrep validation progress

## Current status

Projected-basis chunk prototype passed for L20/100_A2.  Incomplete-sector
classifier validation has started in a bounded two-sector batch.

## Rules preserved

- Existing corrected raw determinant outputs are reused where complete; no cachegen or cache regeneration.
- `QC = F3inv + K3df` and `projected_QC = Vsel.adjoint() * QC * Vsel`.
- `scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6.0)`.
- Raw `determinant(scaled_projected_QC)` is primary; no determinant-level Nproj rescaling.
- Only `digonto_v3_window` and `digonto_v4_window` are production classifier modes.

## Sector queue

1. L24/100_A2 — scored; v3/v4 both FAIL acceptance with TP=9, FP=1, FN=0, TN=42; review required
2. L20/110_A2 — scored PASS; v3/v4 both TP=4, FP=0, FN=0, TN=32; accepted window created
3. L20/111_A2 — processed; 30 candidates; v3/v4 each predict 5 true-zero candidates; labels needed
4. L20/200_A2
5. L24/110_A2
6. L24/111_A2
7. L24/200_A2

The processed sectors reused completed corrected CPU raw determinant grids; the
projected-basis chunk was not substituted for the full-sector raw grid.
This preserves raw determinant primacy while avoiding duplicate cache scans.

The accepted-sector fitter is blocked until the L24/100_A2 review is resolved.
No sector is globally accepted until its user labels are scored with FP=0 and FN=0.
