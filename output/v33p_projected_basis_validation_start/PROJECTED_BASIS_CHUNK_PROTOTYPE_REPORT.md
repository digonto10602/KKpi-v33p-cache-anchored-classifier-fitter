# Projected-basis chunk prototype

Status: PROJECTED_BASIS_CHUNK_PROTOTYPE_PASS

- sector: L20/100_A2
- rows: 258 (64 accepted-bracket, 64 false-candidate, 128 mid-grid)
- reader: corrected v33h raw reader, variant_04_real_imag_swapped
- matrix construction: Fproj plus four projected/scaled K3df basis matrices
- max direct-vs-basis matrix absolute difference: 2.84217e-14
- max direct-vs-basis raw determinant absolute difference: 1.6163e-38
- max raw determinant relative difference: 5.92479e-10
- raw determinant sign agreement: 258/258
- GPU basis determinant: not used; existing GPU LU raw bridge was already independently revalidated
- elapsed prototype seconds: 78.0004
- logdet/logabs: not used as classifier or fitter input

The basis path is accepted only if raw matrix, determinant, sign, and bracket behavior agree.
