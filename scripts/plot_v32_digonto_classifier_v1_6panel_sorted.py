#!/usr/bin/env python3
"""
plot_v32_digonto_classifier_v1_6panel_fixed.py

Robust fixed plotter for projected F3^{-1} diagnostics.

Fixes for future packages:
  1. Does NOT encode parameter labels like p010/p100 into the waves tag.
  2. Auto-discovers *_projF3inv_grid.dat and *_projF3inv_eigenvalues_raw.dat by tag/label.
  3. Sorts every loaded table by increasing Ecm before classification/plotting.
  4. Masks huge 10^n-scaled values before sending them to matplotlib, avoiding
     matplotlib transform overflow / Singular matrix errors.
  5. Applies digonto_classifier_v1 after the run:
       - det(projected(F3^{-1})) sign flip, any orientation
       - minAbsEig has a Gaussian-like/local peak
       - minSVprojF3inv has a Gaussian-like/local peak
  6. Draws accepted candidates as dark-gray dashed vertical lines with alpha=0.7.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_YLIMS = [
    (-1.0e5, 1.0e5),    # det(F3inv) signed-log scaled
    (-1.0e5, 1.0e5),    # det(projected F3inv) signed-log scaled
    (-1.0e5, 1.0e5),    # smallest eigenvalue/minAbsEig scaled
    (-1.0e5, 1.0e5),    # tracked raw eigenbranches
    (-1.0e5, 1.0e5),    # minSV scaled
    (-1.0e5, 1.0e5),    # F3inv_iso scaled
]


def parse_columns(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("# columns:"):
                return line.split(":", 1)[1].strip().split()
    raise RuntimeError(f"Could not find '# columns:' line in {path}")


def load_table(path):
    cols = parse_columns(path)
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < len(cols):
                parts += ["nan"] * (len(cols) - len(parts))
            rows.append(parts[:len(cols)])
    out = {}
    for j, c in enumerate(cols):
        vals = []
        is_numeric = True
        for r in rows:
            try:
                vals.append(float(r[j]))
            except Exception:
                is_numeric = False
                break
        if is_numeric:
            out[c] = np.asarray(vals, dtype=float)
        else:
            out[c] = np.asarray([r[j] for r in rows], dtype=object)

    # Critical for iterative/refined meshes: C++ output can contain coarse rows
    # followed by refined rows, or otherwise be non-monotonic in Ecm.  The
    # classifier detects adjacent sign flips, so every column must be reordered
    # consistently by increasing Ecm before any plotting/classification.
    if "Ecm" in out and len(out["Ecm"]) > 1:
        E = np.asarray(out["Ecm"], dtype=float)
        finite = np.isfinite(E)
        # Put finite Ecm rows first in increasing order and leave non-finite rows
        # at the end in stable order. mergesort keeps deterministic ordering for
        # duplicate Ecm values.
        finite_idx = np.where(finite)[0]
        nonfinite_idx = np.where(~finite)[0]
        order = np.concatenate([finite_idx[np.argsort(E[finite_idx], kind="mergesort")], nonfinite_idx])
        for c in list(out.keys()):
            if len(out[c]) == len(order):
                out[c] = out[c][order]

    return cols, out


def find_grid(outdir, tag, label, waves=None):
    outdir = Path(outdir)
    if waves:
        p = outdir / f"{tag}_{label}_{waves}_projF3inv_grid.dat"
        if p.exists():
            return str(p)

    # robust fallback: tag+label and any waves tag
    pats = [
        str(outdir / f"{tag}_{label}_*_projF3inv_grid.dat"),
        str(outdir / f"{tag}_*_projF3inv_grid.dat"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(
            f"Grid file not found. Tried tag={tag!r}, label={label!r}, waves={waves!r} in {outdir}"
        )
    if len(hits) > 1:
        # Prefer exact label hit, then shortest name. Print warning but continue.
        exact = [h for h in hits if f"_{label}_" in Path(h).name]
        hits = exact or hits
        hits = sorted(hits, key=lambda x: (len(Path(x).name), Path(x).name))
        print(f"[plot-fixed-warning] multiple grid files matched; using {hits[0]}", file=sys.stderr)
    return hits[0]


def find_eigs(outdir, tag, label, waves=None):
    outdir = Path(outdir)
    if waves:
        p = outdir / f"{tag}_{label}_{waves}_projF3inv_eigenvalues_raw.dat"
        if p.exists():
            return str(p)
    pats = [
        str(outdir / f"{tag}_{label}_*_projF3inv_eigenvalues_raw.dat"),
        str(outdir / f"{tag}_*_projF3inv_eigenvalues_raw.dat"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    hits = sorted(set(hits))
    if not hits:
        return None
    if len(hits) > 1:
        exact = [h for h in hits if f"_{label}_" in Path(h).name]
        hits = exact or hits
        hits = sorted(hits, key=lambda x: (len(Path(x).name), Path(x).name))
    return hits[0]


def safe_sign(x):
    if not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def interp_zero(x1, y1, x2, y2):
    if np.isfinite(y1) and np.isfinite(y2) and abs(y2 - y1) > 0:
        return x1 - y1 * (x2 - x1) / (y2 - y1)
    return 0.5 * (x1 + x2)


def signflip_indices(data):
    E = data["Ecm"]
    sign = data.get("signDetRe", np.asarray([safe_sign(v) for v in data["detProjF3inv_re"]], dtype=float))
    out = []
    for i in range(len(E) - 1):
        s1 = int(sign[i])
        s2 = int(sign[i + 1])
        if s1 != 0 and s2 != 0 and s1 * s2 < 0:
            out.append(i)
    return out


def peak_ratio(arr, i, outer_points=80, core_points=2):
    n = len(arr)
    lo = max(0, i - outer_points)
    hi = min(n, i + 1 + outer_points)
    clo = max(0, i - core_points)
    chi = min(n, i + 2 + core_points)

    core = np.abs(arr[clo:chi])
    core = core[np.isfinite(core)]

    shoulder = np.concatenate([np.abs(arr[lo:clo]), np.abs(arr[chi:hi])])
    shoulder = shoulder[np.isfinite(shoulder) & (shoulder > 0)]

    if len(core) == 0 or len(shoulder) == 0:
        return np.nan, np.nan, np.nan

    cmax = float(np.max(core))
    smed = float(np.median(shoulder))
    if smed <= 0 or not np.isfinite(smed):
        return np.nan, cmax, smed
    return cmax / smed, cmax, smed


def digonto_classifier_v1(data, peak_threshold=50.0, outer_points=80, core_points=2,
                           reject_dimension_jumps=True):
    E = data["Ecm"]
    det = data["detProjF3inv_re"]
    minEig = data["minAbsEig"]
    minSV = data["minSVprojF3inv"]
    N = data.get("N", np.zeros_like(E))
    P = data.get("proj_dim", np.zeros_like(E))
    sign = data.get("signDetRe", np.asarray([safe_sign(v) for v in det], dtype=float))

    candidates = []
    for i in signflip_indices(data):
        e_cand = interp_zero(E[i], det[i], E[i + 1], det[i + 1])
        s1, s2 = int(sign[i]), int(sign[i + 1])
        sameN = int(N[i]) == int(N[i + 1])
        sameP = int(P[i]) == int(P[i + 1])

        rEig, eig_core, eig_shoulder = peak_ratio(minEig, i, outer_points, core_points)
        rSV, sv_core, sv_shoulder = peak_ratio(minSV, i, outer_points, core_points)

        accepted = (np.isfinite(rEig) and np.isfinite(rSV)
                    and rEig >= peak_threshold and rSV >= peak_threshold)
        if reject_dimension_jumps and not (sameN and sameP):
            accepted = False

        candidates.append({
            "E_candidate": e_cand,
            "E_left": float(E[i]),
            "E_right": float(E[i + 1]),
            "orientation": f"{s1}->{s2}",
            "N_left": int(N[i]),
            "N_right": int(N[i + 1]),
            "proj_dim_left": int(P[i]),
            "proj_dim_right": int(P[i + 1]),
            "sameN": int(sameN),
            "sameProjDim": int(sameP),
            "minEig_peak_ratio": float(rEig) if np.isfinite(rEig) else np.nan,
            "minSV_peak_ratio": float(rSV) if np.isfinite(rSV) else np.nan,
            "accepted": int(accepted),
        })
    return candidates


def write_candidate_report(path, case_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# digonto_classifier_v1 candidate report\n")
        f.write("# rule: det(projected(F3^{-1})) sign flip, any orientation; minAbsEig Gaussian-like peak; minSV Gaussian-like peak\n")
        f.write("# columns: case E_candidate E_left E_right orientation minEig_peak_ratio minSV_peak_ratio accepted N_left N_right proj_dim_left proj_dim_right sameN sameProjDim\n")
        for case, rows in case_rows:
            for c in rows:
                f.write(
                    f"{case} {c['E_candidate']:.15g} {c['E_left']:.15g} {c['E_right']:.15g} "
                    f"{c['orientation']} {c['minEig_peak_ratio']:.8g} {c['minSV_peak_ratio']:.8g} "
                    f"{c['accepted']} {c['N_left']} {c['N_right']} {c['proj_dim_left']} {c['proj_dim_right']} "
                    f"{c['sameN']} {c['sameProjDim']}\n"
                )


def finite_masked_y(E, y, ylim=None):
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(E) & np.isfinite(y)
    if ylim is not None:
        lo, hi = min(ylim), max(ylim)
        pad = 1.05
        mask &= (y >= lo * pad) & (y <= hi * pad)
    return E[mask], y[mask]


def plot_scaled(ax, E, y, nscale, label=None, ylim=None, lw=0.45, alpha=0.65):
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    first = True
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(int(nscale) + 1):
            scale = 10.0 ** n
            yy = y * scale
            xplot, yplot = finite_masked_y(E, yy, ylim=ylim)
            if len(xplot) > 1:
                ax.plot(xplot, yplot, lw=lw, alpha=alpha, label=(label if first else None))
                first = False


def load_eigen_branches(path):
    if not path:
        return None
    cols, data = load_table(path)
    if "Ecm" not in data:
        return None

    # Expected layouts vary. Handle common format:
    # branch_index Ecm eig_re eig_im or Ecm branch eig_re eig_im.
    keys = set(cols)
    if {"branch", "Ecm", "eig_re"}.issubset(keys):
        branch = data["branch"].astype(int)
        E = data["Ecm"]
        y = data["eig_re"]
    elif {"branch_index", "Ecm", "eig_re"}.issubset(keys):
        branch = data["branch_index"].astype(int)
        E = data["Ecm"]
        y = data["eig_re"]
    elif {"Ecm", "eig_re"}.issubset(keys):
        # no branch labels; just plot raw points
        branch = np.zeros_like(data["Ecm"], dtype=int)
        E = data["Ecm"]
        y = data["eig_re"]
    else:
        # Fallback: try columns by position.
        nums = [c for c in cols if c in data and getattr(data[c], "dtype", None) is not None and data[c].dtype.kind in "fc"]
        return None
    return branch, E, y


def add_case(axs, outdir, tag, label, waves, case_label, nscale, ylims, peak_threshold, outer_points, core_points):
    grid = find_grid(outdir, tag, label, waves)
    eigs = find_eigs(outdir, tag, label, waves)
    print(f"[plot-fixed] {case_label}: grid={grid}")
    if eigs:
        print(f"[plot-fixed] {case_label}: eigs={eigs}")

    cols, data = load_table(grid)
    E = data["Ecm"]

    candidates = digonto_classifier_v1(
        data,
        peak_threshold=peak_threshold,
        outer_points=outer_points,
        core_points=core_points,
        reject_dimension_jumps=True,
    )
    accepted = [c for c in candidates if c["accepted"]]
    print(f"[digonto_classifier_v1] {case_label} accepted: " + (", ".join(f"{c['E_candidate']:.12f}" for c in accepted) if accepted else "none"))

    plot_scaled(axs[0], E, data["detFullF3inv_signed_logabs"], nscale, label=case_label, ylim=ylims[0])
    plot_scaled(axs[1], E, data["signed_logabs"], nscale, label=case_label, ylim=ylims[1])
    plot_scaled(axs[2], E, data["minAbsEig"], nscale, label=case_label, ylim=ylims[2])

    eb = load_eigen_branches(eigs)
    if eb is not None:
        branch, Ee, Ye = eb
        for b in sorted(set(branch.astype(int))):
            m = branch.astype(int) == b
            xplot, yplot = finite_masked_y(Ee[m], Ye[m], ylim=ylims[3])
            if len(xplot) > 1:
                axs[3].plot(xplot, yplot, lw=0.35, alpha=0.45)
    else:
        # fallback: show nothing in branch panel except candidate markers
        pass

    plot_scaled(axs[4], E, data["minSVprojF3inv"], nscale, label=case_label, ylim=ylims[4])
    if "F3inv_iso_re" in data:
        plot_scaled(axs[5], E, data["F3inv_iso_re"], nscale, label=case_label, ylim=ylims[5])

    for ax in axs:
        for c in accepted:
            ax.axvline(c["E_candidate"], color="darkgray", linestyle="--", lw=1.15, alpha=0.7, zorder=18)

    return candidates, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir-a", required=True)
    ap.add_argument("--tag-a", required=True)
    ap.add_argument("--case-a", default="case A")
    ap.add_argument("--waves-a", default=None)
    ap.add_argument("--outdir-b", default=None)
    ap.add_argument("--tag-b", default=None)
    ap.add_argument("--case-b", default="case B")
    ap.add_argument("--waves-b", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nscale", type=int, default=300)
    ap.add_argument("--peak-threshold", type=float, default=50.0)
    ap.add_argument("--outer-points", type=int, default=80)
    ap.add_argument("--core-points", type=int, default=2)
    ap.add_argument("--png", required=True)
    ap.add_argument("--candidate-report", required=True)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-fixed-ylims", action="store_true")
    args = ap.parse_args()

    ylims = [None] * 6 if args.no_fixed_ylims else DEFAULT_YLIMS

    fig, axs = plt.subplots(6, 1, figsize=(15.5, 12.8), sharex=True)
    fig.suptitle(f"{args.label}: projected F3inv diagnostics with digonto_classifier_v1", y=0.996, fontsize=14)

    cases = []
    ca, grid_a = add_case(axs, args.outdir_a, args.tag_a, args.label, args.waves_a, args.case_a,
                          args.nscale, ylims, args.peak_threshold, args.outer_points, args.core_points)
    cases.append((args.case_a, ca))

    if args.outdir_b and args.tag_b:
        cb, grid_b = add_case(axs, args.outdir_b, args.tag_b, args.label, args.waves_b, args.case_b,
                              args.nscale, ylims, args.peak_threshold, args.outer_points, args.core_points)
        cases.append((args.case_b, cb))

    titles = [
        "det(F3^{-1}) signed logabs, scaled",
        "det(projected F3^{-1}) signed logabs, scaled",
        "smallest eigenvalue min|lambda|, scaled",
        "tracked/raw eigenvalue branches Re(lambda)",
        "minimum singular value minSV(projected F3^{-1}), scaled",
        "legacy F3iso^{-1} real part, scaled",
    ]
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], loc="left", fontsize=10)
        ax.axhline(0.0, lw=0.7, alpha=0.35)
        ax.grid(True, alpha=0.18)
        if ylims[i] is not None:
            ax.set_ylim(*ylims[i])
    axs[0].legend(loc="upper right", fontsize=9)
    axs[-1].set_xlabel("Ecm")

    write_candidate_report(args.candidate_report, cases)
    Path(args.png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    fig.savefig(args.png, dpi=180)
    print(f"[plot-fixed] wrote {args.png}")
    print(f"[plot-fixed] wrote {args.candidate_report}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
