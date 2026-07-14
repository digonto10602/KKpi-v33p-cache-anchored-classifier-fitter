#!/usr/bin/env python3
"""Plot F3inv_isotropic diagnostics with n-scale overlays.

Input files are produced by source/extract_F3inv_isotropic_from_gpu_cache_v33a.cpp.
Default column is F3inv_isotropic_rayleigh (column index 5, zero-based after np.loadtxt).
For each input .dat this creates one PNG and one PDF with f(x), f(x)*10^1, ..., f(x)*10^n.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="input .dat files")
    ap.add_argument("--outdir", default="plots/F3inv_isotropic_nscale_v33a")
    ap.add_argument("--nscale", type=int, default=100)
    ap.add_argument("--column", type=int, default=5, help="0-based numeric column to plot; default 5 = F3inv_isotropic_rayleigh")
    ap.add_argument("--ylim", type=float, nargs=2, default=[-1e6, 1e6])
    ap.add_argument("--xcol", type=int, default=1, help="0-based Ecm column; default 1")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fp in map(Path, args.files):
        arr = np.loadtxt(fp, comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        x = arr[:, args.xcol]
        y = arr[:, args.column]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(args.nscale + 1):
            ax.plot(x, y * (10.0 ** k), linewidth=0.8, alpha=0.35)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylim(args.ylim)
        ax.set_xlabel(r"$E_{cm}$")
        ax.set_ylabel(r"$F_{3}^{-1,iso}(E_{cm}) \times 10^k$")
        ax.set_title(fp.stem)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        png = outdir / f"{fp.stem}_nscale{args.nscale}.png"
        pdf = outdir / f"{fp.stem}_nscale{args.nscale}.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"[plot-v33a] wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
