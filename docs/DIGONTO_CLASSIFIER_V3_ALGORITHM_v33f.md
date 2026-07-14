# digonto_classifier_v3 window algorithm

For a sign flip between grid indices `i` and `i+1`, use the six-point window:

```text
left shoulder:  i-2, i-1, i
right shoulder: i+1, i+2, i+3
```

The ordered shoulders are:

```text
left far -> near:  [i-2, i-1, i]
right far -> near: [i+3, i+2, i+1]
```

All local values are rescaled using the max absolute value in the six-point window, so the classifier sees values in `[-1,+1]`.

A true zero is shoulder-like: `abs(y)` decreases toward the sign flip. A pole is blow-up-like: `abs(y)` increases toward the sign flip.

Dynamic trimming is enabled. If the middle point of a three-point shoulder is a local extremum, the far point is dropped and the two points closest to the sign flip are used. This implements the user rule:

```text
if y[i-2] > y[i-1] and y[i] > y[i-1], use y[i-1] -> y[i]
```

The same rule is applied to the right shoulder in its far-to-near ordering.

Labels:

```text
true_zero
pole
uncertain
```

Production v3 classification does not use `refinementN`; exact/refined mode is only a validation oracle.
