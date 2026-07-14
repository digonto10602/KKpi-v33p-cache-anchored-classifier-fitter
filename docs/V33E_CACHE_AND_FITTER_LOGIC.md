# v33e cache and fitter logic

## Available v33e components

The v33e production caches have `.bin` plus `.bin.meta.json` sidecars for:

```text
F2
K2inv
G
F3
F3inv
F3inv_Vsel
projected_F3inv
Vsel
plm_config
klm_config
```

## K3df-only mode

Load:

```text
F3inv_Vsel
```

or equivalently load:

```text
F3inv
Vsel
plm_config
klm_config
```

Then build K3df at each cached energy and compute:

```text
QC = F3inv + K3df
projected_scaled_det = det((1/(Lbyas*xi)^6) * Vsel^dagger * QC * Vsel)
```

This is the mode implemented by the inherited v32x/v33a fitter path and used by v33f for the first count checks and 1-FCN benchmark.

## Two-body + K3df mode for Codex to add after K3df-only is validated

Fit parameters:

```text
scatter_KK_swave_a0
scatter_piK_swave_a0
scatter_piK_pwave_a1
K3iso0
K3iso1
K3B
K3E
```

Default two-body values:

```text
piK_swave_a0 = scatter1_00 = 4.04
piK_pwave_a1 = scatter1_10 = -43.2
KK_swave_a0  = scatter2_00 = 4.12
```

For this mode, cached `K2inv` cannot be fixed because it changes at every FCN evaluation. Load `F2`, `G`, `Vsel`, `plm_config`, and `klm_config`; rebuild `K2inv` from current scattering parameters; rebuild `F3` and `F3inv` with the exact generator algebra; then continue with `F3inv + K3df`.

Use the cached `K2inv` and `F3inv` only for validation at default scattering parameters.
