# Fitter return status

The validated hot-window fitter remains limited to L20/L24 `000_A1m` plus the
already accepted L20 `100_A2` package.  New sectors are not added to the
production fitter until raw determinant parity, classifier validation, and
user-label scoring are complete.

The projected-basis prototype passed on selected L20/100_A2 rows.  It is
appropriate for later hot-window precomputation, but the two new full-sector
classifier grids reused completed corrected CPU raw determinant outputs.  The
next fitter step is a one-FCN run only after new accepted windows exist; no
full minimization is started by this task.

Processed but waiting for labels: L24/100_A2 and L20/110_A2.  Remaining queue:
L20/111_A2, L20/200_A2, L24/110_A2, L24/111_A2, and L24/200_A2.
