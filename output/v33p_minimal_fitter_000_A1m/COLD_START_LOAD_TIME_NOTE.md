# Cold-start load-time note

Attempt 2 measured:

- cache load and window preparation: `350.256811719 s`
- total wall time: `6:10.17`
- peak RSS: `7,653,700 kB` (~7.30 GiB)

The dominant startup cost is sequential external binary-cache I/O plus
allocation and retention of the raw `F3inv_full` and `Vsel` matrices for both
20,000-row sectors. The FCN itself took `0.000736493 s` and evaluated only 404
local rows, so FCN determinant work is not the cold-start bottleneck.

The hot-window constructor does not precompute projected K3df bases for all
20,000 rows. It precomputes only the maximum accepted-window rows: 501 for
L20 and three 501-row windows for L24, 2,004 rows total. The remaining startup
cost is raw cache loading, CPU-side matrix construction, and memory retention
outside the FCN.

No cold-start optimization was attempted in this pass. The external cache
root remains `/media/digonto/Data/F3inv_cache/`; no cachegen ran and no cache
binary was copied.
