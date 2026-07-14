# Projected-basis cache format plan

Status: `PROJECTED_BASIS_CHUNK_PROTOTYPE_PASS`; format defined, not yet generated for production sectors.

## Row record

Each selected cache row is stored in a little-endian binary record with:

- magic/version and record byte length;
- row index, `Ecm`, `Lbyas`, `xi`, `Nfull`, and `Nproj`;
- five matrices in column-major `complex<double>` order:
  `scaled_projected_F3inv`, `scaled_projected_K3iso0_basis`,
  `scaled_projected_K3iso1_basis`, `scaled_projected_K3B_basis`, and
  `scaled_projected_K3E_basis`;
- provenance and convention in the JSON sidecar, not duplicated per matrix.

The sidecar records cache path/hash, metadata window, irrep, coarseN, corrected
complex-read convention, `(Lbyas*xi)^6` scaling, physics formulas, row count,
dimension histogram, and source code/checkpoint identity.

## Size estimate

For L20/100_A2 at 20,000 rows, five matrices require approximately
`826,442,160` bytes (`0.770 GiB`) before row headers and sidecar metadata:
`sum_rows(5 * Nproj^2 * 16)`.  This is feasible as an external working cache,
but it must not be copied into the source package or Git repository.

## Use in fitting

For hot-window fitting, store only selected windows.  The fit assembles

`A = Fproj + K3iso0*B0 + K3iso1*B1 + K3B*BB + K3E*BE`

and sends the assembled raw matrix to the existing determinant backend.  Raw
`determinant(A)` remains the classifier/fitter quantity; logabs/logdet are
validation diagnostics only.  A full-sector basis cache is optional and should
be generated only after per-sector raw determinant and classifier parity pass.
