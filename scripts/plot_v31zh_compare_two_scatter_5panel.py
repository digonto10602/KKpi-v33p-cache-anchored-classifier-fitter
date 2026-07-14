#!/usr/bin/env python3
"""Overlay two v31z/v31zd projected F3^{-1} diagnostics on the same 5-panel plot.

This script is independent: run it after the two C++ runs have written the output_*/.dat files.
It overlays all curves from case A and case B using fixed colors and alpha.
"""
import argparse, pathlib, math, warnings
import numpy as np
import matplotlib.pyplot as plt

DARKRED = "darkred"
DARKCYAN = "darkcyan"

# y limits chosen to match the uploaded reference PDF style (test_with_small_ma.pdf)
PDF_YLIMS = [
    (-1.2e10, 2.6e10),   # det(F3^{-1}) signed log |det| and x10^n
    (-1.1e6, 1.1e6),     # det(proj F3^{-1}) signed log |det| and x10^n
    (-0.1e6, 1.55e6),    # min |lambda| and x10^n
    (-45.0, 45.0),       # tracked Re eigenvalue branches
    (-1.2e6, 3.8e6),     # minSVprojF3inv and x10^n
]


def parse_columns(path):
    cols = None
    with open(path) as f:
        for line in f:
            if line.startswith('# columns:'):
                cols = line.split(':', 1)[1].strip().split()
    return cols or []


def load_grid(path):
    cols = parse_columns(path)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            vals = []
            for x in parts[:len(cols)]:
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(np.nan)
            if len(vals) == len(cols):
                rows.append(vals)
    arr = np.array(rows, dtype=float) if rows else np.empty((0, len(cols)))
    return cols, arr


def col(cols, arr, name):
    if name not in cols:
        raise KeyError(f"Missing column {name}; available columns are {cols}")
    return arr[:, cols.index(name)]


def load_nonint(path):
    vals = []
    if not path.exists():
        return vals
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except Exception:
                    pass
    return sorted(set(vals))


def load_raw_eigs(path):
    if not path.exists():
        return {}, []
    by = {}
    order = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                i = int(float(p[0])); E = float(p[1]); z = complex(float(p[3]), float(p[4]))
            except Exception:
                continue
            if i not in by:
                by[i] = []
                order.append((i, E))
            by[i].append(z)
    order.sort()
    for k in by:
        by[k] = np.array(by[k], complex)
    return by, order


def track_eigenbranches(by, order):
    if not order:
        return np.array([]), np.empty((0, 0), complex)
    Es = np.array([E for _, E in order], float)
    maxdim = max(len(by[i]) for i, _ in order)
    branches = np.full((maxdim, len(order)), np.nan + 1j*np.nan, complex)
    prev = None
    for t, (i, E) in enumerate(order):
        vals = np.array(by.get(i, []), complex)
        if vals.size == 0:
            continue
        if prev is None:
            ord0 = np.lexsort((vals.imag, vals.real))
            for b, j in enumerate(ord0[:maxdim]):
                branches[b, t] = vals[j]
            prev = branches[:, t].copy()
            continue
        used = set()
        newcol = np.full(maxdim, np.nan + 1j*np.nan, complex)
        for b, zprev in enumerate(prev):
            if not np.isfinite(zprev.real):
                continue
            d = np.abs(vals - zprev)
            for u in used:
                d[u] = np.inf
            j = int(np.argmin(d))
            if np.isfinite(d[j]):
                newcol[b] = vals[j]
                used.add(j)
        free = [b for b in range(maxdim) if not np.isfinite(newcol[b].real)]
        remain = [j for j in range(vals.size) if j not in used]
        remain = sorted(remain, key=lambda j: (vals[j].real, vals[j].imag))
        for b, j in zip(free, remain):
            newcol[b] = vals[j]
        branches[:, t] = newcol
        prev = newcol.copy()
    return Es, branches


def data_paths(outdir, tag, label, waves):
    outdir = pathlib.Path(outdir)
    stem = f"{tag}_{label}_{waves}"
    return {
        "grid": outdir / f"{stem}_projF3inv_grid.dat",
        "eigs": outdir / f"{stem}_projF3inv_eigenvalues_raw.dat",
        "nonint": outdir / f"{tag}_{label}_nonint_group_degeneracies.dat",
        "stem": stem,
    }


def plot_scaled(ax, E, y, nscale, color, alpha, lw=0.45):
    y = np.asarray(y, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(nscale + 1):
            yy = y * (10.0 ** n)
            mask = np.isfinite(E) & np.isfinite(yy)
            if np.count_nonzero(mask) > 1:
                ax.plot(E[mask], yy[mask], color=color, alpha=alpha, lw=lw)


def scatter_nonint(ax, nonints):
    if nonints:
        ax.scatter(nonints, [0.0] * len(nonints), s=62, facecolors='white',
                   edgecolors='darkred', linewidths=1.7, zorder=20)


def add_case(axs, paths, nscale, color, alpha, label_text):
    if not paths["grid"].exists():
        raise SystemExit(f"Grid file not found: {paths['grid']}")
    cols, arr = load_grid(paths["grid"])
    E = col(cols, arr, 'Ecm')
    det_full = col(cols, arr, 'detFullF3inv_signed_logabs')
    det_proj = col(cols, arr, 'signed_logabs')
    min_eig = col(cols, arr, 'minAbsEig')
    min_sv = col(cols, arr, 'minSVprojF3inv')
    plot_scaled(axs[0], E, det_full, nscale, color, alpha)
    plot_scaled(axs[1], E, det_proj, nscale, color, alpha)
    plot_scaled(axs[2], E, min_eig, nscale, color, alpha)
    plot_scaled(axs[4], E, min_sv, nscale, color, alpha)

    by, order = load_raw_eigs(paths["eigs"])
    Eb, branches = track_eigenbranches(by, order)
    if branches.size:
        for b in range(branches.shape[0]):
            y = branches[b, :].real
            mask = np.isfinite(Eb) & np.isfinite(y)
            if np.count_nonzero(mask) > 1:
                axs[3].plot(Eb[mask], y[mask], color=color, alpha=alpha, lw=0.45)

    # legend handle
    axs[0].plot([], [], color=color, alpha=alpha, lw=2.0, label=label_text)
    return load_nonint(paths["nonint"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir-a', default='output_scatter001')
    ap.add_argument('--tag-a', default='debug_v31zh_M1_1_M2_0p5_scatter001')
    ap.add_argument('--outdir-b', default='output_scatter010')
    ap.add_argument('--tag-b', default='debug_v31zh_M1_1_M2_0p5_scatter010')
    ap.add_argument('--label', default='110_A2')
    ap.add_argument('--waves', default='waves_0')
    ap.add_argument('--nscale', type=int, default=300)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--png', default='output_compare/v31zh_M1_1_M2_0p5_scatter001_vs_scatter010_5panel_overlay.png')
    ap.add_argument('--no-pdf-ylims', action='store_true')
    args = ap.parse_args()

    pa = data_paths(args.outdir_a, args.tag_a, args.label, args.waves)
    pb = data_paths(args.outdir_b, args.tag_b, args.label, args.waves)

    fig, axs = plt.subplots(5, 1, figsize=(15, 10.5), sharex=True)
    fig.suptitle('v31zh F3inv diagnostics overlay: M1=1.0, M2=0.5, 110_A2, waves_0', y=0.995, fontsize=14)
    non_a = add_case(axs, pa, args.nscale, DARKRED, 0.7, 'scatter1_00=scatter2_00=0.01')
    non_b = add_case(axs, pb, args.nscale, DARKCYAN, 0.7, 'scatter1_00=scatter2_00=0.1')
    nonints = sorted(set(non_a) | set(non_b))

    labels = [
        r'det($F_3^{-1}$) signed log $|\det|$ and $\times 10^n$',
        r'det(proj $F_3^{-1}$) signed log $|\det|$ and $\times 10^n$',
        r'smallest eigenvalue diagnostic min$|\lambda|$ and $\times 10^n$',
        r'tracked Re eigenvalue branches',
        r'minSVprojF3inv and $\times 10^n$',
    ]
    for i, ax in enumerate(axs):
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, lw=0.75, alpha=0.45)
        scatter_nonint(ax, nonints)
        if not args.no_pdf_ylims:
            ax.set_ylim(*PDF_YLIMS[i])
    axs[-1].set_xlabel('Ecm')
    axs[0].legend(loc='upper left', fontsize=8)

    png = pathlib.Path(args.png)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(png, dpi=180)
    print(f'[v31zh-plot] wrote {png}')
    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f'[v31zh-plot-warning] could not show plot interactively: {e}')
    plt.close(fig)


if __name__ == '__main__':
    main()
