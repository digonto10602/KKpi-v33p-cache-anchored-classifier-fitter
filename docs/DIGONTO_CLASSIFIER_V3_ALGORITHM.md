# Digonto Classifier v3 Algorithm

## Goal

Replace the expensive `v2_refine_points` true-refinement classifier with a coarse-grid-only classifier that uses the six local points surrounding each determinant sign flip.

The target sign-flip observable is

```text
projected_scaled_det(E) = det( (1/(Lbyas*xi)^6) * Vsel^dagger * (F3inv + K3df) * Vsel )
```

Sign flips are searched in `projected_scaled_det` on the fixed GPU-cache grid.  The v3 classifier does not build CPU refinement points.

## Local window

For a sign flip between adjacent grid rows `i` and `i+1`, use

```text
left shoulder:   i-2, i-1, i
right shoulder:  i+1, i+2, i+3
```

The left shoulder is read from far to near as `i-2 -> i-1 -> i`.  The right shoulder is also read from far to near as `i+3 -> i+2 -> i+1`.

## Rescaling

For the local six-point window, define

```text
M = max(abs(y[j])) for j in [i-2, i-1, i, i+1, i+2, i+3]
y_scaled[j] = y[j] / M
```

The shape classifier acts on `abs(y_scaled)`, but the original signs are retained to reject shoulder points that contain extra sign flips.

## Zero-like vs pole-like shoulder

A shoulder is **zero-like** when `abs(y)` clearly decreases as it approaches the flip.

For a three-point left shoulder, this means

```text
abs(y[i-2]) > abs(y[i-1]) > abs(y[i])
```

with a tunable fractional margin.  For the right shoulder, this means

```text
abs(y[i+3]) > abs(y[i+2]) > abs(y[i+1])
```

A shoulder is **pole-like** when `abs(y)` clearly rises as it approaches the flip.

## Dynamic three-to-two trimming

If a three-point shoulder has a local extremum, the farthest point is discarded and the nearest two-point shoulder is tested.  This implements the requested behavior:

```text
if abs(y[i-2]) > abs(y[i-1]) and abs(y[i]) > abs(y[i-1]),
then discard i-2 and test only i-1 -> i for the left shoulder.
```

The same procedure is applied symmetrically on the right shoulder.  Extra sign changes inside a shoulder also cause the farthest offending point to be discarded.

## Candidate classification

Default production behavior in this starter package:

```text
true_zero  if at least one shoulder is zero-like and both shoulders are not pole-like
pole       if either shoulder is pole-like and the zero condition failed
uncertain  otherwise
```

There is a stricter config option:

```text
v3_require_both_shoulders = 1
```

which requires both shoulders to be zero-like before accepting a true zero.

## Tunable parameters

```text
classifier_mode = digonto_v3_window
v3_monotone_tol = 0.02
v3_min_drop_fraction = 0.15
v3_min_pole_rise_fraction = 0.15
v3_require_both_shoulders = 0
```

`v3_min_drop_fraction = 0.15` means each step toward the flip must drop by at least 15 percent to count as a clean zero-like decrease.

## Required validation

Before trusting v3 in production fits, Codex must compare it against the last correct v32x/v32w exact/refined classifier on the same caches and K3df parameters:

1. identical or explainably close accepted true-zero counts per `(Lbyas, irrep)`;
2. same ordering of accepted levels used by the multi-L lattice fit;
3. energy differences below the coarse-grid resolution unless a local v2 refinement is being used as reference;
4. no missing lattice levels under `Ecm_cutoff = 0.335`;
5. faster FCN wall time than v2/refined mode.
