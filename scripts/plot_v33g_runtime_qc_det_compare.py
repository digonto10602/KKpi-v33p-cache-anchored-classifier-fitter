#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAGIC = 0x5633334752544B33


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def parse_fit_summary(path: Path) -> dict[str, float]:
    vals: dict[str, float] = {}
    for row in read_rows(path):
        if len(row) >= 2 and row[0] in {"K3iso0", "K3iso1", "K3B", "K3E"}:
            vals[row[0]] = float(row[1])
    missing = [k for k in ("K3iso0", "K3iso1", "K3B", "K3E") if k not in vals]
    if missing:
        raise RuntimeError(f"Missing fit parameters in {path}: {', '.join(missing)}")
    return vals


def parse_fit_levels(path: Path, irrep: str) -> list[float]:
    vals: list[float] = []
    ptag, ir = irrep.split("_", 1)
    for row in read_rows(path):
        if len(row) < 9:
            continue
        if row[2] == irrep or (row[3] == ir and row[2].startswith(ptag + "_")):
            vals.append(float(row[8]))
    return sorted(vals)


def parse_zero_file(path: Path, irrep: str) -> list[float]:
    vals: list[float] = []
    ptag, ir = irrep.split("_", 1)
    for row in read_rows(path):
        if len(row) < 6:
            continue
        if row[2] == irrep or (row[3] == ir and row[2].startswith(ptag + "_")):
            if row[5] == "true_zero":
                vals.append(float(row[4]))
    vals.sort()
    return vals


def load_runtime_meta(cache_path: Path) -> dict:
    meta = json.loads(cache_path.with_suffix(cache_path.suffix + ".meta.json").read_text())
    return meta


def read_u64(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buf, off)[0], off + 8


def read_i32(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def read_f64(buf: memoryview, off: int) -> tuple[float, int]:
    return struct.unpack_from("<d", buf, off)[0], off + 8


def read_cpx_matrix(buf: memoryview, off: int) -> tuple[np.ndarray, int]:
    rows, off = read_u64(buf, off)
    cols, off = read_u64(buf, off)
    n = rows * cols
    arr = np.empty((rows, cols), dtype=np.complex128, order="F")
    for c in range(cols):
        for r in range(rows):
            re, off = read_f64(buf, off)
            im, off = read_f64(buf, off)
            arr[r, c] = complex(re, im)
    return arr, off


def load_runtime_cache(cache_path: Path, irrep: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = cache_path.read_bytes()
    buf = memoryview(data)
    header_end = data.index(b"\n") + 1
    header = data[:header_end - 1].decode("utf-8")
    if header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1":
        raise RuntimeError(f"Bad runtime cache header in {cache_path}: {header}")

    off = header_end
    ecm: list[float] = []
    det_scaled: list[float] = []
    signflip_det: list[float] = []

    while off < len(buf):
        magic, off = read_u64(buf, off)
        if magic != MAGIC:
            raise RuntimeError(f"Bad runtime cache record magic in {cache_path}")
        _i, off = read_i32(buf, off)
        _total_dim, off = read_i32(buf, off)
        _proj_dim, off = read_i32(buf, off)
        success, off = read_i32(buf, off)
        _reserved, off = read_i32(buf, off)
        e, off = read_f64(buf, off)
        _en, off = read_f64(buf, off)
        f3inv_proj, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if success != 1:
            continue
        ecm.append(e)
        # QC = F3inv_proj + sum(K_i * basis_i), then scaled by (L*xi)^6.
        # The runtime cache stores projected matrices already.
        q = f3inv_proj + 0.0 * b0 + 0.0 * b1 + 0.0 * bb + 0.0 * be
        det_scaled.append(np.linalg.det(q).real)
        signflip_det.append(0.0)  # placeholder, overwritten later from fit params

    if not ecm:
        raise RuntimeError(f"No usable cache rows found in {cache_path}")
    return np.asarray(ecm, dtype=float), np.asarray(det_scaled, dtype=float), np.asarray(signflip_det, dtype=float)


def load_runtime_cache_with_params(cache_path: Path, params: dict[str, float], Lbyas: float, xi: float, norm_power: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    data = cache_path.read_bytes()
    buf = memoryview(data)
    header_end = data.index(b"\n") + 1
    header = data[:header_end - 1].decode("utf-8")
    if header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1":
        raise RuntimeError(f"Bad runtime cache header in {cache_path}: {header}")

    off = header_end
    ecm: list[float] = []
    det_scaled: list[float] = []
    while off < len(buf):
        magic, off = read_u64(buf, off)
        if magic != MAGIC:
            raise RuntimeError(f"Bad runtime cache record magic in {cache_path}")
        _i, off = read_i32(buf, off)
        _total_dim, off = read_i32(buf, off)
        _proj_dim, off = read_i32(buf, off)
        success, off = read_i32(buf, off)
        _reserved, off = read_i32(buf, off)
        e, off = read_f64(buf, off)
        _en, off = read_f64(buf, off)
        f3inv_proj, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if success != 1:
            continue
        q = f3inv_proj
        q = q + params["K3iso0"] * b0 + params["K3iso1"] * b1 + params["K3B"] * bb + params["K3E"] * be
        q = q / ((Lbyas * xi) ** norm_power)
        ecm.append(e)
        det_scaled.append(np.linalg.det(q).real)

    if not ecm:
        raise RuntimeError(f"No usable cache rows found in {cache_path}")
    return np.asarray(ecm, dtype=float), np.asarray(det_scaled, dtype=float)


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def v3_classify_shoulder(vals_far_to_near: list[float], raw_far_to_near: list[float], tol: float, drop: float, rise: float) -> tuple[str, str]:
    if len(vals_far_to_near) < 2:
        return "uncertain", "too_few_points"
    vals = vals_far_to_near[:]
    raw = raw_far_to_near[:]

    def same_sign(a: float, b: float) -> bool:
        sa, sb = sign(a), sign(b)
        return sa != 0 and sb != 0 and sa == sb

    while len(vals) > 2 and not same_sign(raw[-2], raw[-1]):
        vals.pop(0)
        raw.pop(0)
    while len(vals) > 2 and not same_sign(raw[0], raw[-1]):
        vals.pop(0)
        raw.pop(0)
    if len(vals) < 2:
        return "uncertain", "same_sign_trim_removed_window"

    a = [abs(v) for v in vals]
    if len(a) >= 3:
        middle_local_min = a[0] > a[1] * (1.0 + tol) and a[2] > a[1] * (1.0 + tol)
        middle_local_max = a[1] > a[0] * (1.0 + tol) and a[1] > a[2] * (1.0 + tol)
        if middle_local_min or middle_local_max:
            vals.pop(0)
            raw.pop(0)
            a = [abs(v) for v in vals]

    zero_like = True
    pole_like = True
    for i in range(len(a) - 1):
        far = a[i] + 1.0e-300
        near = a[i + 1] + 1.0e-300
        d = (far - near) / max(far, 1.0e-300)
        r = (near - far) / max(far, 1.0e-300)
        zero_like = zero_like and (d >= drop)
        pole_like = pole_like and (r >= rise)
    if zero_like:
        return "zero", "monotone_abs_det_decrease"
    if pole_like:
        return "pole", "monotone_abs_det_rise"
    return "uncertain", "nonmonotone_or_weak_shoulders"


def v3_true_zeros(ecm: np.ndarray, det: np.ndarray, mono_tol: float = 0.02, drop: float = 0.15, rise: float = 0.15, require_both: bool = False) -> list[float]:
    finite = np.isfinite(ecm) & np.isfinite(det)
    e = ecm[finite]
    y = det[finite]
    order = np.argsort(e, kind="mergesort")
    e = e[order]
    y = y[order]

    zeros: list[float] = []
    for i in range(len(e) - 1):
        si, sj = sign(float(y[i])), sign(float(y[i + 1]))
        if si == 0 or sj == 0 or si == sj:
            continue
        if i < 1 or i + 2 >= len(e):
            continue
        local = [abs(float(y[j])) for j in range(max(0, i - 2), min(len(e), i + 4)) if np.isfinite(y[j])]
        if not local:
            continue
        m = max(local)
        if not (m > 0.0):
            continue
        lv = [float(y[j]) / m for j in range(i - 2, i + 1) if j >= 0 and np.isfinite(y[j])]
        ls = [float(y[j]) for j in range(i - 2, i + 1) if j >= 0 and np.isfinite(y[j])]
        rv = [float(y[j]) / m for j in range(i + 3, i, -1) if j < len(e) and np.isfinite(y[j])]
        rs = [float(y[j]) for j in range(i + 3, i, -1) if j < len(e) and np.isfinite(y[j])]
        Lk, _ = v3_classify_shoulder(lv, ls, mono_tol, drop, rise)
        Rk, _ = v3_classify_shoulder(rv, rs, mono_tol, drop, rise)
        left_zero = Lk == "zero"
        right_zero = Rk == "zero"
        left_pole = Lk == "pole"
        right_pole = Rk == "pole"
        if (require_both and left_zero and right_zero and not (left_pole and right_pole)) or (not require_both and (left_zero or right_zero) and not (left_pole and right_pole)):
            e0 = e[i] - y[i] * (e[i + 1] - e[i]) / (y[i + 1] - y[i])
            if np.isfinite(e0):
                zeros.append(float(e0))
    zeros.sort()
    uniq: list[float] = []
    for z in zeros:
        if not uniq or abs(z - uniq[-1]) > 1.0e-8:
            uniq.append(z)
    return uniq


def v4_merge_true_zeros(true_zeros: list[float], merge_tol: float = 1.0e-3) -> tuple[list[float], list[float]]:
    if not true_zeros:
        return [], []
    zeros = sorted(true_zeros)
    accepted = [zeros[0]]
    centers = [zeros[0]]
    rejected: list[float] = []
    for z in zeros[1:]:
        if abs(z - centers[-1]) <= merge_tol:
            rejected.append(z)
            centers[-1] = (centers[-1] + z) / 2.0
            accepted[-1] = centers[-1]
        else:
            accepted.append(z)
            centers.append(z)
    return accepted, rejected


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot scaled projected QC determinant from v33g runtime cache and overlay two true-zero sets")
    ap.add_argument("--runtime-cache", default="cache/v33g_runtime_k3basis/v33g_Lbyas20_xi3p444_irrep000_A1m_coarse20000_runtime_K3basis.bin")
    ap.add_argument("--fit-summary", default="output_v33g/fit_all_irreps_runtime_ecm0335/debug_v33g_all_irreps_runtime_ecm0335_fit_summary_allL.dat")
    ap.add_argument("--file-zeros", default="output_v33g/fit_all_irreps_runtime_ecm0335/debug_v33g_all_irreps_runtime_ecm0335_bestfit_QC_spectrum_L20.dat")
    ap.add_argument("--fit-levels", default="output_v33g/fit_all_irreps_runtime_ecm0335/debug_v33g_all_irreps_runtime_ecm0335_fit_levels_L20.dat")
    ap.add_argument("--irrep", default="000_A1m")
    ap.add_argument("--nscale", type=int, default=6)
    ap.add_argument("--classifier", choices=("v3", "v4"), default="v3")
    ap.add_argument("--merge-tol", type=float, default=1.0e-3)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--png", default="output_plots/v33g_runtime_000A1m_ecm0335_qc_det_compare.png")
    args = ap.parse_args()

    runtime_cache = Path(args.runtime_cache)
    fit_summary = Path(args.fit_summary)
    file_zeros = Path(args.file_zeros)
    fit_levels = Path(args.fit_levels)

    meta = load_runtime_meta(runtime_cache)
    if meta.get("version") != "v33g_runtime_k3basis_cache":
        raise RuntimeError(f"Unexpected runtime cache version: {meta.get('version')}")
    if int(meta.get("coarseN", -1)) != 20000:
        raise RuntimeError(f"Expected coarseN=20000, found {meta.get('coarseN')}")
    if meta.get("irrep") != args.irrep:
        raise RuntimeError(f"Unexpected irrep in runtime cache: {meta.get('irrep')}")

    params = parse_fit_summary(fit_summary)
    ecm, det = load_runtime_cache_with_params(runtime_cache, params, float(meta["Lbyas"]), float(meta["xi"]), norm_power=6.0)
    fresh_zeros_v3 = v3_true_zeros(ecm, det, mono_tol=0.02, drop=0.15, rise=0.15, require_both=False)
    fresh_zeros_v4, rejected_v4 = v4_merge_true_zeros(fresh_zeros_v3, merge_tol=args.merge_tol)
    file_zero_vals = parse_zero_file(file_zeros, args.irrep)
    model_levels = parse_fit_levels(fit_levels, args.irrep)

    print(f"[plot] runtime_cache={runtime_cache}")
    print(f"[plot] coarseN={meta.get('coarseN')} rows={len(ecm)}")
    print(
        f"[plot] file_true_zeros={len(file_zero_vals)} "
        f"fresh_true_zeros_v3={len(fresh_zeros_v3)} "
        f"fresh_true_zeros_v4={len(fresh_zeros_v4)} "
        f"model_levels={len(model_levels)}"
    )

    fig, ax = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
    for n in range(max(0, args.nscale) + 1):
        yy = det * (10.0 ** n)
        yy = np.asarray(yy, dtype=float)
        yy[~np.isfinite(yy)] = np.nan
        ax.plot(ecm, yy, color="black", linewidth=0.9, alpha=0.32 if n else 0.9, zorder=5)

    for x in model_levels:
        ax.axvline(x, color="darkorange", linestyle="--", linewidth=1.4, alpha=1.0, zorder=20)

    ax.scatter(
        file_zero_vals,
        np.zeros(len(file_zero_vals)),
        s=90,
        marker="o",
        facecolors="white",
        edgecolors="red",
        linewidths=1.5,
        zorder=30,
        label="file true_zero",
    )
    ax.scatter(
        fresh_zeros_v3,
        np.zeros(len(fresh_zeros_v3)),
        s=48,
        marker="o",
        facecolors="white",
        edgecolors="blue",
        linewidths=1.5,
        zorder=31,
        label="fresh classifier_v3 true_zero",
    )
    ax.scatter(
        fresh_zeros_v4,
        np.zeros(len(fresh_zeros_v4)),
        s=72,
        marker="o",
        facecolors="none",
        edgecolors="green",
        linewidths=1.8,
        zorder=32,
        label="fresh classifier_v4 true_zero",
    )
    if rejected_v4:
        ax.scatter(
            rejected_v4,
            np.zeros(len(rejected_v4)),
            s=34,
            marker="x",
            c="green",
            linewidths=1.2,
            zorder=33,
            label="v4 rejected duplicates",
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=1.0, zorder=1)
    ax.set_xlabel(r"$E_{cm}$")
    ax.set_ylabel(r"$\det\left[(V^\dagger(QC)V)/(L\xi)^6\right]$")
    ax.set_title(
        f"v33g runtime projected-QC determinant, L{int(meta['Lbyas'])} {args.irrep}, "
        f"K3iso0={params['K3iso0']:.6g}, K3iso1={params['K3iso1']:.6g}, "
        f"K3B={params['K3B']:.6g}, K3E={params['K3E']:.6g}"
    )
    ax.grid(True, alpha=0.22, linewidth=0.5, zorder=0)
    ax.legend(fontsize=8, loc="best")

    png = Path(args.png)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220)
    print(f"[plot] wrote {png}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
