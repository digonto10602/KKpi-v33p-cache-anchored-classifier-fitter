# Covariance inverse / pseudoinverse audit

## Result

The covariance treatment is safe for this four-level test. The active
chi-square mode is `raw_cov_inv`:

```text
r = E_lattice - E_model
chi2 = r^T C^+ r
```

## Source and method

- Chi-square: `k3df_fit_v32f::chi_square_v32f` in
  `source/K3df_minuit_fit_v32f_fullF3inv_QCfull_cached_classifier.hpp:508-512`.
- Inverse builder: `symmetric_pseudoinverse_v32f` in the same header,
  around lines 381-395.
- Method: symmetrize `A` as `(A+A^T)/2`, diagonalize with
  `Eigen::SelfAdjointEigenSolver`, invert eigenvalues above the cutoff, then
  reconstruct `V D^+ V^T`.
- This is an eigenvalue pseudoinverse, not a direct inverse, SVD inverse, or
  ridge-regularized inverse.
- Relative cutoff: `rcond=1e-12`; absolute cutoff is
  `max(1e-12*max(abs(eigenvalue)), 1e-300)`.

## Input paths

- Source jackknife data:
  `/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data/20_000_A1m_n0.jack`
  and `24_000_A1m_n{0,1,2}.jack`.
- Generated covariance snapshot:
  `output/v33p_minimal_fitter_000_A1m/chi2_one_fcn_audit/chi2_one_fcn_audit_covariance_allL.dat`
- Generated inverse snapshot:
  `output/v33p_minimal_fitter_000_A1m/chi2_one_fcn_audit/chi2_one_fcn_audit_covariance_inv_allL.dat`

## Matrix diagnostics

```text
eigenvalues(C) =
0.013723783692701535
0.051047101317634833
0.096229380237644283
0.26419944636183840

condition number = 19.25121032782983
```

The matrix is positive definite and not singular or near-singular for this
test. No covariance eigenmodes are discarded: every eigenvalue is far above
the `2.641994463618384e-13` cutoff implied by the largest eigenvalue.

The reconstructed `C^+` is symmetric to about `3e-14`. Numerically,
`C^+ C` differs from the 4x4 identity by at most about `2e-15`; it is an
identity here rather than a lower-rank projector. Therefore the pseudoinverse
behaves as the ordinary inverse for this selected covariance.

The L24 off-diagonal covariance entries are retained. L20/L24 cross-blocks
are zero by the existing multi-L covariance construction.

No chi-square formula or covariance code was changed.
