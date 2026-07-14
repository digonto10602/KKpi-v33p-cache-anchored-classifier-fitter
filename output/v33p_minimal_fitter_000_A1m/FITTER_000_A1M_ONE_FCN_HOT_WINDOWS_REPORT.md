# 000_A1m hot-window one-FCN report

## Status

**FAIL for the fitter gate; hot-window architecture PASS.** The run exited
successfully, but found only 3 of 4 model roots and returned the configured
failure penalty.

## Command

```bash
output/v33p_minimal_fitter_000_A1m/run_one_fcn_hot_windows.sh
```

The script invoked `bin/v33f_k3df_fitter_multiL_v33e` with the hot-window
config and `fcn-once`, under `/usr/bin/time -v`.

## Setup

- Sectors: L20/000_A1m and L24/000_A1m only.
- Existing external raw caches: `/media/digonto/Data/F3inv_cache/`.
- Cache metadata Ecm window: `[0.26310, 0.36]`.
- `Ecm_cutoff=0.335`.
- Classifier: `digonto_v3_window`.
- Determinant backend: `auto` -> `cpu_openmp`.
- `fallback_full_scan=0`.
- K3df: `(73735.840894011912, -972421.14060757787,
  347174.05548116949, -1226756.7068845264)`.

## Local work

| sector | windows | initial rows | rows evaluated | expansions | model roots |
|---|---:|---:|---:|---:|---:|
| L20/000_A1m | 1 | 101 | 602 | 1 | 0 |
| L24/000_A1m | 3 | 303 | 804 | 1 | 3 |
| **total** | **4** | **404** | **1406** | **2** | **3/4** |

No full 20,000-row scan and no fallback occurred. Determinants were
recomputed from current K3df parameters; old determinant CSV values were used
only for window indexing.

## Result and timing

- `chi2 = 1e+100` (failure penalty, not a physical fit result).
- `model_found = 3/4`.
- FCN evaluation time: `0.002416485 s`.
- Cache load and selected-window precompute: `730.296144883 s`.
- Wall time: `12:10.64`.
- User/system CPU: `723.26 s` / `3.03 s`.
- Peak RSS: `7,853,444 kB` (~7.49 GiB).

## Interpretation

The FCN architecture now satisfies the no-full-scan requirement, but
minimization is not safe because the L20 accepted window did not produce a
model root at the fixed starting K3df parameters. The next step is a targeted
local determinant parity/mapping audit against the trusted completed CPU scan;
do not broaden this into a full-grid FCN scan.

Logs: `one_fcn_hot_windows.log` and preserved
`one_fcn_hot_windows_attempt1.log`.
