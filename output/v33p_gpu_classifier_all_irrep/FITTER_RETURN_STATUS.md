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

L20/110_A2 passed and has an accepted window set. L24/100_A2 failed with one
false positive in both accepted modes and is excluded pending review. L20/111_A2
is prepared but still needs labels. No one-FCN or minimization was run because
the accepted-sector set is not yet clean.
