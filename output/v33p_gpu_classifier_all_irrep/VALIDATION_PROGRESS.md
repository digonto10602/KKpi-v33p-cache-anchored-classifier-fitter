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

1. L24/100_A2 — processed; 52 candidates; v3/v4 each predict 10 true-zero candidates; labels needed
2. L20/110_A2 — processed; 36 candidates; v3/v4 each predict 4 true-zero candidates; labels needed
3. L20/111_A2
4. L20/200_A2
5. L24/110_A2
6. L24/111_A2
7. L24/200_A2

The two processed sectors reused completed corrected CPU raw determinant grids;
the projected-basis chunk was not substituted for the full-sector raw grid.
This preserves raw determinant primacy while avoiding duplicate cache scans.

No sector is globally accepted until its user labels are scored with FP=0 and FN=0.
