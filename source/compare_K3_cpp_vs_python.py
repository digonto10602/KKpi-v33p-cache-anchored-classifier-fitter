#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np


def add_qc3_paths(qc3_root):
    qc3_root = os.path.abspath(qc3_root)
    sys.path.insert(0, qc3_root)
    sys.path.insert(0, os.path.join(qc3_root, "base_code"))
    sys.path.insert(0, os.path.join(qc3_root, "base_code", "Kdf3"))


def read_cpp_configs(filename):
    rows = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            rows.append({
                "g": int(parts[0]),
                "flavor": int(parts[1]),
                "local": int(parts[2]),
                "nx": int(parts[3]),
                "ny": int(parts[4]),
                "nz": int(parts[5]),
                "px": float(parts[6]),
                "py": float(parts[7]),
                "pz": float(parts[8]),
                "ell": int(parts[9]),
                "m": int(parts[10]),
            })
    return rows


def read_cpp_matrix(filename):
    vals = []
    nmax = -1
    with open(filename, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            r, c, re, im, ab = line.split()
            r = int(r); c = int(c)
            vals.append((r, c, float(re) + 1j * float(im)))
            nmax = max(nmax, r, c)
    M = np.zeros((nmax + 1, nmax + 1), dtype=np.complex128)
    for r, c, z in vals:
        M[r, c] = z
    return M


def local_index_from_ell_m(ell, m):
    if ell == 0 and m == 0:
        return 0
    if ell == 1 and m in (-1, 0, 1):
        return m + 2  # Python K3main local sp order: s, p(-1), p(0), p(+1)
    raise ValueError(f"Only s and p waves are supported by this test, got ell={ell}, m={m}")


def expected_element_from_python(K3main, E, Pvec, row, col, Kiso, K3B_par, K3E_par, M12):
    i = row["flavor"]
    j = col["flavor"]
    pvec = np.array([row["px"], row["py"], row["pz"]], dtype=float)
    kvec = np.array([col["px"], col["py"], col["pz"]], dtype=float)

    waves_out = "sp" if i == 1 else "s"
    waves_in  = "sp" if j == 1 else "s"

    block = K3main.K3_ij_pk(
        E, Pvec, pvec, kvec,
        i, j,
        Kiso, K3B_par, K3E_par,
        M12=M12,
        waves_ij=(waves_out, waves_in),
    )

    a = local_index_from_ell_m(row["ell"], row["m"])
    b = local_index_from_ell_m(col["ell"], col["m"])

    # For flavor-2 sector only s-wave exists, so local index must be 0.
    if i == 2 and a != 0:
        raise ValueError("flavor-2 row has non-s-wave element")
    if j == 2 and b != 0:
        raise ValueError("flavor-2 col has non-s-wave element")

    factor = 1.0
    if i != j:
        factor = 1.0 / np.sqrt(2.0)
    elif i == 2 and j == 2:
        factor = 0.5

    return factor * block[a, b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc3-root", default="QC3_release", help="path to unpacked QC3_release directory")
    ap.add_argument("--configs", default="k3_cpp_configs.dat")
    ap.add_argument("--matrix", default="k3_cpp_matrix.dat")
    ap.add_argument("--nPx", type=int, default=0)
    ap.add_argument("--nPy", type=int, default=0)
    ap.add_argument("--nPz", type=int, default=1)
    ap.add_argument("--Ecm", type=float, default=0.28)
    ap.add_argument("--Lbyas", type=float, default=20.0)
    ap.add_argument("--Kiso", type=float, nargs="+", default=[200.0, 400.0])
    ap.add_argument("--K3B", type=float, default=400.0)
    ap.add_argument("--K3E", type=float, default=300.0)
    ap.add_argument("--max-print", type=int, default=25)
    ap.add_argument("--tol", type=float, default=1.0e-10)
    args = ap.parse_args()

    add_qc3_paths(args.qc3_root)
    import defns
    import K3main

    atmpi = 0.06906
    atmK = 0.09698
    xi = 3.444
    L = xi * args.Lbyas
    nnP = [args.nPx, args.nPy, args.nPz]
    Pvec = np.array([2.0 * np.pi * x / L for x in nnP], dtype=float)
    E = defns.E_to_Ecm(args.Ecm, L, nnP, rev=True)

    rows = read_cpp_configs(args.configs)
    Kcpp = read_cpp_matrix(args.matrix)
    assert len(rows) == Kcpp.shape[0], (len(rows), Kcpp.shape)

    Kpy = np.zeros_like(Kcpp)
    for r, row in enumerate(rows):
        for c, col in enumerate(rows):
            Kpy[r, c] = expected_element_from_python(
                K3main, E, Pvec, row, col,
                args.Kiso, args.K3B, args.K3E,
                M12=[atmK, atmpi],
            )

    diff = Kcpp - Kpy
    absdiff = np.abs(diff)
    max_abs = float(absdiff.max()) if absdiff.size else 0.0
    where = np.unravel_index(np.argmax(absdiff), absdiff.shape)

    denom = np.maximum(np.maximum(np.abs(Kcpp), np.abs(Kpy)), 1.0e-300)
    reldiff = absdiff / denom
    max_rel = float(reldiff.max()) if reldiff.size else 0.0

    print(f"N = {Kcpp.shape[0]}")
    print(f"Ecm = {args.Ecm:.17g}")
    print(f"E   = {E:.17g}")
    print(f"max_abs_diff = {max_abs:.17e} at {where}")
    print(f"max_rel_diff = {max_rel:.17e}")
    print(f"cpp[{where}] = {Kcpp[where]:.17e}")
    print(f"py [{where}] = {Kpy[where]:.17e}")

    bad = np.argwhere(absdiff > args.tol)
    print(f"num_bad_abs_gt_{args.tol:g} = {len(bad)}")
    for idx, (r, c) in enumerate(bad[:args.max_print]):
        row = rows[r]; col = rows[c]
        print(
            f"bad {idx:04d}: r={r} c={c} "
            f"row=(flav={row['flavor']},n=({row['nx']},{row['ny']},{row['nz']}),ell={row['ell']},m={row['m']}) "
            f"col=(flav={col['flavor']},n=({col['nx']},{col['ny']},{col['nz']}),ell={col['ell']},m={col['m']}) "
            f"cpp={Kcpp[r,c]:.17e} py={Kpy[r,c]:.17e} diff={diff[r,c]:.17e} abs={absdiff[r,c]:.3e} rel={reldiff[r,c]:.3e}"
        )

    if max_abs <= args.tol:
        print("PASS")
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
