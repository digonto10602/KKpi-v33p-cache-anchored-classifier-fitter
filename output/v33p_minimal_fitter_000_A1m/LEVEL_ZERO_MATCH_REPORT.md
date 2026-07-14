# 000_A1m level/zero match report

Status: **COUNTS MATCH; FCN EXECUTION BLOCKED BY EXPENSIVE WORK INSIDE FCN**.

- Source lattice input: `/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data`
- Loader configuration: `configs/config_v33f_multiL_000A1m_5levels_v33e_v3.in`
- Energy type: `En_lab`, converted by the existing fitter loader to Ecm
- Cutoff: `Ecm_cutoff = 0.335`
- Accepted-zero source: `digonto_v3_window`; `digonto_v4_window` agrees exactly.

## L20/000_A1m

Lattice level:

| level | source | Ecm |
|---:|---|---:|
| 0 | `20_000_A1m_n0.jack` | 0.26922660199004972 |

Accepted true zeros:

| bracket | zero estimate | inside cutoff |
|---:|---:|---|
| 1 | 0.26932269955846955 | true |
| 9 | 0.34538561384780109 | false |
| 10 | 0.35794609133611466 | false |

Count comparison: **1 lattice level = 1 accepted zero** below the cutoff.

## L24/000_A1m

Lattice levels:

| level | source | Ecm |
|---:|---|---:|
| 0 | `24_000_A1m_n0.jack` | 0.26688321699819179 |
| 1 | `24_000_A1m_n1.jack` | 0.32089324050632922 |
| 2 | `24_000_A1m_n2.jack` | 0.32953266907775774 |

Accepted true zeros:

| bracket | zero estimate | inside cutoff |
|---:|---:|---|
| 1 | 0.26661705359335075 | true |
| 9 | 0.32172352122054665 | true |
| 10 | 0.33063143645332377 | true |

Count comparison: **3 lattice levels = 3 accepted zeros** below the cutoff.

The counts match, but no fitter FCN was run because the runtime audit found a
20,000-row determinant/classifier scan inside every FCN evaluation.
