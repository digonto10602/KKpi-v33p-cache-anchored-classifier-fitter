#!/usr/bin/env python3
"""Plot v33f F3inv_iso n-scale overlays with v3 true zeros as white/black scatter points."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def loadtxt_safe(path: Path, mincols: int = 1) -> np.ndarray:
    try:
        arr = np.loadtxt(path, comments="#")
    except Exception:
        return np.empty((0, mincols))
    if arr.size == 0:
        return np.empty((0, mincols))
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rows.append(line.split())
    except OSError:
        return []
    return rows


def load_true_zeros(path: Path) -> np.ndarray:
    rows = read_rows(path)
    vals: list[float] = []
    for row in rows:
        if len(row) >= 2 and row[1] == "true_zero":
            try:
                vals.append(float(row[0]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="output_v33f/F3inv_iso_visual_test")
    ap.add_argument("--outdir", default="plots/v33f_F3inv_iso_nscale_v3_truezeros")
    ap.add_argument("--nscale", type=int, default=100)
    ap.add_argument("--ylim", type=float, nargs=2, default=[-1e6, 1e6])
    ap.add_argument("--xcol", type=int, default=1)
    ap.add_argument("--ycol", type=int, default=5)
    ap.add_argument("--show", action="store_true", help="Call plt.show() for each generated plot")
    args = ap.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_files = sorted(indir.glob("v33f_F3inv_iso_L*_*.dat"))
    data_files = [p for p in data_files if not p.name.endswith("_v3_truezeros.dat") and not p.name.endswith("_v3_all_candidates.dat")]
    if not data_files:
        raise SystemExit(f"No data files found in {indir}")

    for fp in data_files:
        arr = loadtxt_safe(fp, 9)
        if arr.shape[0] == 0:
            continue
        x = arr[:, args.xcol]
        y = arr[:, args.ycol]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue

        zero_fp = indir / f"{fp.stem}_v3_truezeros.dat"
        zx = load_true_zeros(zero_fp) if zero_fp.exists() else np.array([])
        zy = np.interp(zx, x, y) if zx.size else np.array([])

        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        for k in range(args.nscale + 1):
            ax.plot(x, y * (10.0 ** k), linewidth=0.75, alpha=0.26)
        if zx.size:
            ax.scatter(zx, np.zeros_like(zx), s=46, facecolors="white", edgecolors="black", linewidths=0.9, zorder=5)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylim(args.ylim)
        ax.set_xlabel(r"$E_{cm}$")
        ax.set_ylabel(r"$F_{3}^{-1,iso}(E_{cm}) \times 10^k$")
        ax.set_title(fp.stem + "  |  v3 true zeros: " + str(zx.size))
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        png = outdir / f"{fp.stem}_nscale{args.nscale}_v3zeros.png"
        pdf = outdir / f"{fp.stem}_nscale{args.nscale}_v3zeros.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        if args.show:
            plt.show()
        plt.close(fig)
        print(f"[v33f-plot] wrote {png}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
