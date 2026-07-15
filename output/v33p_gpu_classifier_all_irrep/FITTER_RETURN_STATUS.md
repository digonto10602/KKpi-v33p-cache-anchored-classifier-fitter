# Fitter return status

The accepted-window one-FCN path now passes for five sectors:

- L20/000_A1m
- L20/100_A2
- L20/110_A2
- L20/111_A2
- L24/000_A1m

The run found 12/12 roots with finite chi2 `0.0002394162713767611`, evaluated
2214 local rows, used `auto -> cpu_openmp`, and performed no fallback or full
scan. Cache/window preparation was `944.73168146 s`; the internal FCN time was
`0.029772724 s`.

Projected-basis hot-window `.bin` and `.json` caches are built and validated
for these five sectors. The fitter does not yet load those projected-basis
files; the current one-FCN uses the existing accepted-window path.

L24/100_A2 remains excluded because both accepted classifier modes have one
false positive at bracket 36. L20/200_A2 is prepared with a blank label
template; later incomplete sectors also still need labels. No minimization was
run.
