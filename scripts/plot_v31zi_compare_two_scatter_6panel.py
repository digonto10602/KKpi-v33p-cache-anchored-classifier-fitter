#!/usr/bin/env python3
"""Overlay two v31z/v31zd projected F3^{-1} diagnostics on the same 6-panel plot.

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
    (-1.2e10, 2.6e10),   # F3inv_iso real and x10^n
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



def orientation_ok_v31zn(sL, sR, orientation):
    if not (np.isfinite(sL) and np.isfinite(sR)):
        return False
    if sL == 0 or sR == 0:
        return False
    if orientation == 'plus-to-minus':
        return sL > 0 and sR < 0
    if orientation == 'minus-to-plus':
        return sL < 0 and sR > 0
    return sL * sR < 0


def local_peak_ratio_v31zn(y, i, outer_points=80, core_points=2):
    """Return peak/core ratio relative to side shoulders around sign-flip i,i+1."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    lo = max(0, i - outer_points)
    hi = min(n, i + 2 + outer_points)
    clo = max(0, i - core_points)
    chi = min(n, i + 2 + core_points)
    shoulder_idx = np.r_[lo:clo, chi:hi]
    core_idx = np.arange(clo, chi)
    sh = y[shoulder_idx]
    co = y[core_idx]
    sh = sh[np.isfinite(sh) & (sh > 0)]
    co = co[np.isfinite(co) & (co > 0)]
    if sh.size == 0 or co.size == 0:
        return np.nan, np.nan, np.nan
    shoulder = float(np.nanmedian(sh))
    peak = float(np.nanmax(co))
    trough = float(np.nanmin(co))
    if not np.isfinite(shoulder) or shoulder <= 0:
        return np.nan, peak, shoulder
    return peak / shoulder, peak, shoulder


def digonto_classifier_v1(cols, arr, orientation='any', peak_ratio_threshold=50.0,
                                       outer_points=80, core_points=2):
    """digonto_classifier_v1: blind candidate-zero classifier using only projected F3^{-1} diagnostics.

    Candidate rule used here:
      1) det(projected F3^{-1}) real sign flips, with configurable orientation;
      2) smallest eigenvalue min|lambda| has a local peak;
      3) min singular value minSVprojF3inv has a local peak.

    The default orientation='any' is intentional: in the current data the F3iso^{-1}
    zeros appear with both determinant orientations. Use --candidate-orientation
    plus-to-minus to enforce left positive / right negative only.
    """
    if arr.size == 0:
        return []
    E = col(cols, arr, 'Ecm')
    success = col(cols, arr, 'success')
    sign = col(cols, arr, 'signDetRe')
    min_eig = col(cols, arr, 'minAbsEig')
    min_sv = col(cols, arr, 'minSVprojF3inv')
    rawN = col(cols, arr, 'N') if 'N' in cols else np.full_like(E, np.nan)
    pdim = col(cols, arr, 'proj_dim') if 'proj_dim' in cols else np.full_like(E, np.nan)
    out = []
    for i in range(len(E)-1):
        if success[i] != 1 or success[i+1] != 1:
            continue
        if rawN[i] != rawN[i+1] or pdim[i] != pdim[i+1]:
            continue
        sL, sR = sign[i], sign[i+1]
        if not orientation_ok_v31zn(sL, sR, orientation):
            continue
        eig_ratio, eig_peak, eig_shoulder = local_peak_ratio_v31zn(min_eig, i, outer_points, core_points)
        sv_ratio, sv_peak, sv_shoulder = local_peak_ratio_v31zn(min_sv, i, outer_points, core_points)
        if eig_ratio >= peak_ratio_threshold and sv_ratio >= peak_ratio_threshold:
            out.append({
                'E_left': float(E[i]),
                'E_right': float(E[i+1]),
                'E_candidate': 0.5*(float(E[i]) + float(E[i+1])),
                'orientation': f'{int(sL):+d}->{int(sR):+d}',
                'minEig_peak_ratio': float(eig_ratio),
                'minEig_peak': float(eig_peak),
                'minEig_shoulder': float(eig_shoulder),
                'minSV_peak_ratio': float(sv_ratio),
                'minSV_peak': float(sv_peak),
                'minSV_shoulder': float(sv_shoulder),
                'N': int(rawN[i]) if np.isfinite(rawN[i]) else -1,
                'proj_dim': int(pdim[i]) if np.isfinite(pdim[i]) else -1,
            })
    return out


def draw_digonto_classifier_v1_lines(axs, candidates, label_once=False):
    first = label_once
    for c in candidates:
        E = c['E_candidate']
        for ax in axs:
            ax.axvline(E, color='darkgray', linestyle='--', lw=1.15, alpha=0.7, zorder=18,
                       label='digonto_classifier_v1 zero' if first else None)
            first = False


def write_digonto_classifier_v1_report(path, cases):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('# digonto_classifier_v1 classified candidate zeros from projected F3^{-1} only\n')
        f.write('# rule: det(projected F3^{-1}) sign flip + minAbsEig Gaussian-like peak + minSV Gaussian-like peak\n')
        f.write('# columns: case E_candidate E_left E_right orientation minEig_peak_ratio minSV_peak_ratio minEig_peak minSV_peak minEig_shoulder minSV_shoulder N proj_dim\n')
        for case, cand in cases:
            for c in cand:
                f.write(f"{case} {c['E_candidate']:.15g} {c['E_left']:.15g} {c['E_right']:.15g} "
                        f"{c['orientation']} {c['minEig_peak_ratio']:.8e} {c['minSV_peak_ratio']:.8e} "
                        f"{c['minEig_peak']:.8e} {c['minSV_peak']:.8e} "
                        f"{c['minEig_shoulder']:.8e} {c['minSV_shoulder']:.8e} "
                        f"{c['N']} {c['proj_dim']}\n")

def data_paths(outdir, tag, label, waves):
    outdir = pathlib.Path(outdir)
    stem = f"{tag}_{label}_{waves}"
    return {
        "grid": outdir / f"{stem}_projF3inv_grid.dat",
        "eigs": outdir / f"{stem}_projF3inv_eigenvalues_raw.dat",
        "nonint": outdir / f"{tag}_{label}_nonint_group_degeneracies.dat",
        "stem": stem,
    }


def plot_scaled(ax, E, y, nscale, color, alpha, lw=0.45, ylim=None):
    """Plot f(E)*10^n safely.

    With nscale=300, direct autoscaling can overflow matplotlib transforms even if
    the final plot uses fixed y-limits.  Therefore we mask values outside the
    visible y-window before they ever reach matplotlib.
    """
    y = np.asarray(y, dtype=float)
    E = np.asarray(E, dtype=float)
    if ylim is not None:
        lo, hi = ylim
        # Keep a small pad so curves entering/leaving the window are visible,
        # but never pass 1e300-sized values to matplotlib.
        pad = 1.05
        ylo = min(lo, hi) * pad
        yhi = max(lo, hi) * pad
    else:
        ylo, yhi = -1e100, 1e100
    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(nscale + 1):
            yy = y * (10.0 ** n)
            mask = np.isfinite(E) & np.isfinite(yy) & (yy >= ylo) & (yy <= yhi)
            if np.count_nonzero(mask) > 1:
                ax.plot(E[mask], yy[mask], color=color, alpha=alpha, lw=lw)


def scatter_nonint(ax, nonints):
    if nonints:
        ax.scatter(nonints, [0.0] * len(nonints), s=62, facecolors='white',
                   edgecolors='darkred', linewidths=1.7, zorder=20)


def add_case(axs, paths, nscale, color, alpha, label_text, candidate_args=None):
    if not paths["grid"].exists():
        raise SystemExit(f"Grid file not found: {paths['grid']}")
    cols, arr = load_grid(paths["grid"])
    E = col(cols, arr, 'Ecm')
    det_full = col(cols, arr, 'detFullF3inv_signed_logabs')
    det_proj = col(cols, arr, 'signed_logabs')
    min_eig = col(cols, arr, 'minAbsEig')
    min_sv = col(cols, arr, 'minSVprojF3inv')
    f3inv_iso = col(cols, arr, 'F3inv_iso_re')
    candidates = []
    if candidate_args is not None:
        candidates = digonto_classifier_v1(cols, arr, **candidate_args)
    plot_scaled(axs[0], E, det_full, nscale, color, alpha, ylim=PDF_YLIMS[0])
    plot_scaled(axs[1], E, det_proj, nscale, color, alpha, ylim=PDF_YLIMS[1])
    plot_scaled(axs[2], E, min_eig, nscale, color, alpha, ylim=PDF_YLIMS[2])
    plot_scaled(axs[4], E, min_sv, nscale, color, alpha, ylim=PDF_YLIMS[4])
    plot_scaled(axs[5], E, f3inv_iso, nscale, color, alpha, ylim=PDF_YLIMS[5])

    by, order = load_raw_eigs(paths["eigs"])
    Eb, branches = track_eigenbranches(by, order)
    if branches.size:
        for b in range(branches.shape[0]):
            y = branches[b, :].real
            mask = np.isfinite(Eb) & np.isfinite(y)
            lo, hi = PDF_YLIMS[3]
            mask = mask & (y >= min(lo, hi)*1.05) & (y <= max(lo, hi)*1.05)
            if np.count_nonzero(mask) > 1:
                axs[3].plot(Eb[mask], y[mask], color=color, alpha=alpha, lw=0.45)

    # legend handle
    axs[0].plot([], [], color=color, alpha=alpha, lw=2.0, label=label_text)
    return load_nonint(paths["nonint"]), candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir-a', default='output_scatter001')
    ap.add_argument('--tag-a', default='debug_v31zi_M1_1_M2_0p5_scatter001')
    ap.add_argument('--outdir-b', default='output_scatter010')
    ap.add_argument('--tag-b', default='debug_v31zi_M1_1_M2_0p5_scatter010')
    ap.add_argument('--label', default='110_A2')
    ap.add_argument('--waves', default='', help='Legacy common wave tag. If empty, use --waves-a and --waves-b.')
    ap.add_argument('--waves-a', default='waves_0')
    ap.add_argument('--waves-b', default='waves_0_1')
    ap.add_argument('--nscale', type=int, default=300)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--png', default='output_compare/v31zi_M1_1_M2_0p5_scatter001_vs_scatter010_6panel_overlay.png')
    ap.add_argument('--no-pdf-ylims', action='store_true')
    ap.add_argument('--no-candidate-lines', action='store_true', help='Disable gray dashed classified-candidate lines.')
    ap.add_argument('--candidate-orientation', choices=['any','plus-to-minus','minus-to-plus'], default='any', help='Orientation for det(projected F3inv) sign flips. Default any recovers both observed F3iso-zero orientations.')
    ap.add_argument('--candidate-peak-ratio', type=float, default=50.0, help='Minimum local Gaussian-like peak/shoulder ratio for both smallest eigenvalue min|lambda| and minSV.')
    ap.add_argument('--candidate-outer-points', type=int, default=80, help='Shoulder half-window in grid points for peak classifier.')
    ap.add_argument('--candidate-core-points', type=int, default=2, help='Core half-window in grid points around sign flip.')
    ap.add_argument('--candidate-report', default='', help='Optional output .dat report for classified candidates.')
    args = ap.parse_args()

    waves_a = args.waves if args.waves else args.waves_a
    waves_b = args.waves if args.waves else args.waves_b
    pa = data_paths(args.outdir_a, args.tag_a, args.label, waves_a)
    pb = data_paths(args.outdir_b, args.tag_b, args.label, waves_b)

    fig, axs = plt.subplots(6, 1, figsize=(15, 12.5), sharex=True)
    fig.suptitle(f'v31zr F3inv diagnostics overlay with digonto_classifier_v1: M1=1.0, M2=0.5, {args.label}, {waves_a} vs {waves_b}', y=0.995, fontsize=14)
    if not args.no_pdf_ylims:
        for i, ax in enumerate(axs):
            ax.set_ylim(*PDF_YLIMS[i])
    candidate_args = dict(orientation=args.candidate_orientation,
                          peak_ratio_threshold=args.candidate_peak_ratio,
                          outer_points=args.candidate_outer_points,
                          core_points=args.candidate_core_points)
    non_a, cand_a = add_case(axs, pa, args.nscale, DARKRED, 0.7, 'waves_vec_1={0}, active a=0.01', candidate_args)
    non_b, cand_b = add_case(axs, pb, args.nscale, DARKCYAN, 0.7, 'waves_vec_1={0,1}, active a=0.01', candidate_args)
    nonints = sorted(set(non_a) | set(non_b))
    all_cands = cand_a + cand_b
    print(f'[digonto_classifier_v1] waves0 candidates: ' + ', '.join(f"{c['E_candidate']:.12f}" for c in cand_a))
    print(f'[digonto_classifier_v1] waves01 candidates: ' + ', '.join(f"{c['E_candidate']:.12f}" for c in cand_b))

    labels = [
        r'det($F_3^{-1}$) signed log $|\det|$ and $\times 10^n$',
        r'det(proj $F_3^{-1}$) signed log $|\det|$ and $\times 10^n$',
        r'smallest eigenvalue diagnostic min$|\lambda|$ and $\times 10^n$',
        r'tracked Re eigenvalue branches',
        r'minSVprojF3inv and $\times 10^n$',
        r'$F_{3,\mathrm{iso}}^{-1}=1/(v_{iso}^T F_3 v_{iso})$ real and $\times 10^n$',
    ]
    for i, ax in enumerate(axs):
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, lw=0.75, alpha=0.45)
        scatter_nonint(ax, nonints)
        if not args.no_pdf_ylims:
            ax.set_ylim(*PDF_YLIMS[i])
    if not args.no_candidate_lines:
        draw_digonto_classifier_v1_lines(axs, all_cands, label_once=True)
    report = pathlib.Path(args.candidate_report) if args.candidate_report else pathlib.Path(args.png).with_suffix('.classified_projected_pole_candidates.dat')
    write_digonto_classifier_v1_report(report, [('waves0', cand_a), ('waves01', cand_b)])
    print(f'[digonto_classifier_v1] wrote {report}')
    axs[-1].set_xlabel('Ecm')
    axs[0].legend(loc='upper left', fontsize=8)

    png = pathlib.Path(args.png)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(png, dpi=180)
    print(f'[v31zi-plot] wrote {png}')
    if args.show:
        try:
            plt.show()
        except Exception as e:
            print(f'[v31zi-plot-warning] could not show plot interactively: {e}')
    plt.close(fig)


if __name__ == '__main__':
    main()
