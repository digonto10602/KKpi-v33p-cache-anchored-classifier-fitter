# GPU classifier validation driver

Status: READY; not executed in this task.

The driver uses `bin/v33p_cpuassembled_gpu_det_scan` for existing caches and
then the existing CSV-only classifier harness. It must be run serially by
sector after this L20/100_A2 gate. Only `digonto_v3_window` and
`digonto_v4_window` rows are retained for production validation; raw-sign
modes remain candidate generators only.

Sectors: `L20/L24 x {000_A1m,100_A2,110_A2,111_A2,200_A2}`.

Required gate: preserve cache metadata, corrected reader convention, exact
physics formulas, and compare each generated grid to the CPU reference where
available before using it for labels or accepted windows.
