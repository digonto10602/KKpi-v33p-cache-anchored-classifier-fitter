# Reduced-parameter fit decision

## Decision

A reduced fit is justified, but not the initially suggested
`K3iso0,K3B`-only fit: K3iso1 is demonstrably active in the finite-difference
audit and should not be fixed without additional evidence.

The minimal defensible reduction is:

```text
float: K3iso0, K3iso1, K3B
fix:   K3E = -1226756.7068845264
```

The fixed K3E value is the validated starting value. With 4 levels and 3
floating parameters, the nominal reduced degrees of freedom are `4-3=1`.
This is more informative than the 4-parameter 000_A1m-only plumbing fit,
though still not a broad physics fit because only one irrep is included.

The reduced fit used `accepted_windows`, `det_backend=auto` resolved to
`cpu_openmp`, `fallback_full_scan=0`, and the same four validated lattice
levels. No classifier or physics formula changed.
