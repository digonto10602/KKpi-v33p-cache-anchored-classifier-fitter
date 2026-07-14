# GPU fast-track status

`CHECKPOINT_STATUS = GPU_FASTTRACK_BLOCKED_UNTIL_RAW_READER_GPU_GATE`

The requested L20/000_A1m and L24/000_A1m GPU fast-track scans were not run.
The audited existing CUDA backends do not consume the corrected combined raw
cache reader, so using them would violate the checkpoint invariants. No new
label CSVs were created by this blocked step.

After a raw-reader-integrated GPU backend passes CPU parity, create per-sector
outputs under:

```text
output/v33p_gpu_fasttrack_classifier_validation/L20_000_A1m_E026310_0360/
output/v33p_gpu_fasttrack_classifier_validation/L24_000_A1m_E026310_0360/
```

Do not score or accept either sector until the corresponding user-label CSV is
completed.
