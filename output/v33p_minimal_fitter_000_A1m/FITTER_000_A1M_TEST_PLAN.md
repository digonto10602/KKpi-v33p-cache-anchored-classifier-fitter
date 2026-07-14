# Minimal fitter test plan: 000_A1m only

Status: **BLOCKED_WAITING_FOR_USER_LABELS_AND_GPU_GATE**.

Scope is only `Lbyas=20,24`, `irrep=000_A1m`, using the metadata window
`[0.26310,0.36]`, accepted classifier modes
`digonto_v3_window` and `digonto_v4_window`, and `Ecm_cutoff=0.335` unless an
existing config says otherwise.

Before one FCN:

1. Complete and score both 000_A1m user-label CSVs.
2. Compare accepted true-zero count with selected lattice levels.
3. Confirm cache load occurs once outside FCN.
4. Confirm no 20,000-row determinant scan occurs inside FCN.
5. Run one FCN/chi-square evaluation and record cold-start and one-FCN times.

Do not start long minimization until these checks pass.
