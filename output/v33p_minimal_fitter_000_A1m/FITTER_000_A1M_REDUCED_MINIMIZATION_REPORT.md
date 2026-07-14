# Reduced 000_A1m minimization report

## Status

`CHECKPOINT_STATUS = COVINV_SENSITIVITY_COMPLETE_AND_000_A1M_REDUCED_FIT_PASS`

## Setup

- Sectors: L20/000_A1m and L24/000_A1m only
- Floating: K3iso0, K3iso1, K3B
- Fixed: K3E `-1226756.7068845264`
- Cutoff: `Ecm_cutoff=0.335`
- Root mode: `accepted_windows`
- Determinant backend: `auto -> cpu_openmp`
- Max FCNs: 200; actual FCNs: 137
- Nominal dof: 1

Starting parameters:

```text
K3iso0 =  73735.840894011912
K3iso1 = -972421.14060757787
K3B    =  347174.05548116949
K3E    = -1226756.7068845264 (fixed)
```

## Results

- Starting chi2: `1.5125961842376174e-05`
- Best logged chi2: `1.4818624346499466e-05` at eval 119
- Final chi2: `1.4818722402508267e-05`
- Minuit valid: `1`
- MIGRAD: completed with a valid Minuit result
- EDM: not printed by this executable
- FCN evaluations: `137`
- Model found: `4/4` on every FCN; L20 `1/1`, L24 `3/3`
- Average/max FCN time: `0.000544274153 / 0.000738530 s`
- Cache load/preparation: `332.483629944 s`
- Peak RSS: `7,654,332 kB`
- Window expansions: `0`
- Fallback/full scans: `0/0`
- Exit status: `0`

Final parameters reported by Minuit:

```text
K3iso0 =  73735.840894011912
K3iso1 = -972421.14060757787
K3B    =  366970.36954496522
K3E    = -1226756.7068845264 (fixed)
```

The executable emitted a warning that its legacy 4-parameter covariance
export saw a 3-row Minuit covariance and zero-filled the missing fourth row.
Therefore parameter uncertainties/covariance status are not reported as
validated quantities here; the spectrum fit itself is valid (`valid=1`,
`model_levels_found=4`).

The result is numerically and logically sane for this reduced plumbing test:
all roots remain present, the objective is finite, and no prohibited full
scan or cache reload occurs inside FCN.
