# L20 hot-window root failure audit

- sector: `L20/000_A1m`
- accepted zero: `0.26932269955846955`
- bracket_id: `1`
- bracket Ecm: `[0.269321, 0.269326]`
- nearest row / center row: `1284`
- original bracket rows: `1284-1285`
- initial hot-window rows: `1234-1334`
- maximum audit rows: `1034-1534`
- original bracket inside initial window: `true`
- reference bracket sign flip: `true`
- local bracket sign flip: `true`
- dimension jump inside original root bracket: `false`
- v3 local true-zero: `true`
- v4 local true-zero: `true`
- max absolute determinant component difference: `2.21818e-21`
- max relative determinant difference: `0.622209`
- direct full-assembly vs reference max absolute difference (bracket context): `0`
- direct full-assembly vs projected-basis max absolute difference (bracket context): `2.43916e-22`
- audit-only one-swap determinant at center row: `-2.02744e-20+i(2.17921e-20)`
- audit-only loaded (trusted-reference-compatible) determinant at center row: `1.37092e-22+i(-0)`
- exact cause of the original failure: **the hot path applied only the shared reader's first variant-04 real/imag swap, while the trusted v33h scan applies a second swap after that reader. The one-swap local determinant lost the accepted sign change.**
- fix applied: **a localized second swap in `load_gpu_coarse_cache_one`; the shared reader, physics formulas, and classifier logic were not changed.**
- post-fix classification: both `digonto_v3_window` and `digonto_v4_window` classify the local bracket as a true zero.
- post-fix consistency: the direct full-assembly local determinant agrees with the trusted grid to max absolute component difference `2.21818e-21`; the projected K3 basis path differs from direct assembly by `2.43916e-22`.

## Dimension sequence

- rows `1034-1534`: Nfull/Nproj `35/5`

## Rows around the original bracket

| row | Ecm | reference det | local det | reference sign | local sign | dimension jump |
|---:|---:|---:|---:|---:|---:|---|
| 1282 | 0.269312 | 2.85809e-21 | 2.61418e-21 | 1 | 1 | no |
| 1283 | 0.269316 | 1.61072e-21 | 1.37587e-21 | 1 | 1 | no |
| 1284 | 0.269321 | 3.62878e-22 | 1.37092e-22 | 1 | 1 | no |
| 1285 | 0.269326 | -8.85428e-22 | -1.10214e-21 | -1 | -1 | no |
| 1286 | 0.269331 | -2.1342e-21 | -2.34183e-21 | -1 | -1 | no |
| 1287 | 0.269336 | -3.38343e-21 | -3.58199e-21 | -1 | -1 | no |

Direct full-assembly context rows (basis disabled):

- row 1282: det=2.85809e-21+i(-0), abs_diff_vs_reference=0
- row 1283: det=1.61072e-21+i(-0), abs_diff_vs_reference=0
- row 1284: det=3.62878e-22+i(-0), abs_diff_vs_reference=0
- row 1285: det=-8.85428e-22+i(0), abs_diff_vs_reference=0
- row 1286: det=-2.1342e-21+i(0), abs_diff_vs_reference=0
- row 1287: det=-3.38343e-21+i(0), abs_diff_vs_reference=0

Debug CSV: `output/v33p_minimal_fitter_000_A1m/L20_hot_window_reference_vs_local_det_debug.csv`
