#!/usr/bin/env python3
"""Independent six-panel plotter for v31zi projected F3^{-1} diagnostics.
Can be run without rerunning C++ after output .dat files exist.
"""
import argparse, pathlib
import numpy as np
import matplotlib.pyplot as plt

PDF_YLIMS = [
    (-1.2e10, 2.6e10),
    (-1.1e6, 1.1e6),
    (-0.1e6, 1.55e6),
    (-45.0, 45.0),
    (-1.2e6, 3.8e6),
    (-1.2e10, 2.6e10),
]

def parse_columns(path):
    cols=[]
    with open(path) as f:
        for line in f:
            if line.startswith('# columns:'):
                cols=line.split(':',1)[1].strip().split()
    return cols

def load_grid(path):
    cols=parse_columns(path); rows=[]
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p=line.split(); vals=[]
            for x in p[:len(cols)]:
                try: vals.append(float(x))
                except Exception: vals.append(np.nan)
            if len(vals)==len(cols): rows.append(vals)
    return cols, np.array(rows,float) if rows else np.empty((0,len(cols)))

def col(cols, arr, name):
    if name not in cols:
        raise KeyError(f'Missing column {name}; available={cols}')
    return arr[:, cols.index(name)]

def load_nonint(path):
    vals=[]
    if not path.exists(): return vals
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p=line.split()
            if len(p)>=2:
                try: vals.append(float(p[1]))
                except Exception: pass
    return sorted(set(vals))

def load_raw_eigs(path):
    by={}; order=[]
    if not path.exists(): return by, order
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p=line.split()
            if len(p)<6: continue
            try:
                i=int(float(p[0])); E=float(p[1]); z=complex(float(p[3]),float(p[4]))
            except Exception: continue
            if i not in by:
                by[i]=[]; order.append((i,E))
            by[i].append(z)
    order.sort()
    for k in by: by[k]=np.array(by[k], complex)
    return by, order

def track_eigenbranches(by, order):
    if not order: return np.array([]), np.empty((0,0), complex)
    Es=np.array([E for _,E in order], float)
    maxdim=max(len(by[i]) for i,_ in order)
    branches=np.full((maxdim,len(order)), np.nan+1j*np.nan, complex)
    prev=None
    for t,(i,E) in enumerate(order):
        vals=np.array(by.get(i,[]), complex)
        if vals.size==0: continue
        if prev is None:
            ord0=np.lexsort((vals.imag, vals.real))
            for b,j in enumerate(ord0[:maxdim]): branches[b,t]=vals[j]
            prev=branches[:,t].copy(); continue
        used=set(); newcol=np.full(maxdim,np.nan+1j*np.nan,complex)
        for b,zprev in enumerate(prev):
            if not np.isfinite(zprev.real): continue
            d=np.abs(vals-zprev)
            for u in used: d[u]=np.inf
            j=int(np.argmin(d))
            if np.isfinite(d[j]): newcol[b]=vals[j]; used.add(j)
        free=[b for b in range(maxdim) if not np.isfinite(newcol[b].real)]
        remain=sorted([j for j in range(vals.size) if j not in used], key=lambda j:(vals[j].real, vals[j].imag))
        for b,j in zip(free,remain): newcol[b]=vals[j]
        branches[:,t]=newcol; prev=newcol.copy()
    return Es, branches

def plot_scaled(ax, E, y, nscale, ylim=None):
    """Plot f(E)*10^n safely.

    With nscale=300, direct autoscaling can overflow matplotlib transforms even if
    the final plot uses fixed y-limits. Therefore we mask values outside the
    visible y-window before they ever reach matplotlib.
    """
    E = np.asarray(E, dtype=float)
    y = np.asarray(y, dtype=float)
    if ylim is not None:
        lo, hi = ylim
        pad = 1.05
        ylo = min(lo, hi) * pad
        yhi = max(lo, hi) * pad
    else:
        ylo, yhi = -1e100, 1e100
    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(nscale + 1):
            yy = y * (10.0 ** n)
            m = np.isfinite(E) & np.isfinite(yy) & (yy >= ylo) & (yy <= yhi)
            if np.count_nonzero(m) > 1:
                ax.plot(E[m], yy[m], lw=0.5, alpha=0.75)

def scatter_nonint(ax, vals):
    if vals:
        ax.scatter(vals,[0.0]*len(vals),s=58,facecolors='white',edgecolors='darkred',linewidths=1.6,zorder=10)


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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='output')
    ap.add_argument('--tag', default='debug_v31zi_M1_1_M2_0p5_scatter001')
    ap.add_argument('--label', default='110_A2')
    ap.add_argument('--waves', default='waves_0')
    ap.add_argument('--nscale', type=int, default=300)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--png', default='')
    ap.add_argument('--no-pdf-ylims', action='store_true')
    ap.add_argument('--no-candidate-lines', action='store_true', help='Disable gray dashed classified-candidate lines.')
    ap.add_argument('--candidate-orientation', choices=['any','plus-to-minus','minus-to-plus'], default='any', help='Orientation for det(projected F3inv) sign flips. Default any recovers both observed F3iso-zero orientations.')
    ap.add_argument('--candidate-peak-ratio', type=float, default=50.0, help='Minimum local Gaussian-like peak/shoulder ratio for both smallest eigenvalue min|lambda| and minSV.')
    ap.add_argument('--candidate-outer-points', type=int, default=80, help='Shoulder half-window in grid points for peak classifier.')
    ap.add_argument('--candidate-core-points', type=int, default=2, help='Core half-window in grid points around sign flip.')
    ap.add_argument('--candidate-report', default='', help='Optional output .dat report for classified candidates.')
    args=ap.parse_args()
    outdir=pathlib.Path(args.outdir); stem=f'{args.tag}_{args.label}_{args.waves}'
    grid=outdir/f'{stem}_projF3inv_grid.dat'; eigfile=outdir/f'{stem}_projF3inv_eigenvalues_raw.dat'; nonfile=outdir/f'{args.tag}_{args.label}_nonint_group_degeneracies.dat'
    cols, arr=load_grid(grid)
    if arr.size==0: raise SystemExit(f'No data in {grid}')
    E=col(cols,arr,'Ecm')
    candidate_args=dict(orientation=args.candidate_orientation, peak_ratio_threshold=args.candidate_peak_ratio, outer_points=args.candidate_outer_points, core_points=args.candidate_core_points)
    candidates=digonto_classifier_v1(cols, arr, **candidate_args)
    print('[digonto_classifier_v1] candidates: ' + ', '.join(f"{c['E_candidate']:.12f}" for c in candidates))
    series=[
        ('det($F_3^{-1}$) signed log |det| and $\\times 10^n$', col(cols,arr,'detFullF3inv_signed_logabs')),
        ('det(proj $F_3^{-1}$) signed log |det| and $\\times 10^n$', col(cols,arr,'signed_logabs')),
        ('smallest eigenvalue min$|\\lambda|$ and $\\times 10^n$', col(cols,arr,'minAbsEig')),
        ('tracked Re eigenvalue branches', None),
        ('minSVprojF3inv and $\\times 10^n$', col(cols,arr,'minSVprojF3inv')),
        ('$F_{3,\\mathrm{iso}}^{-1}$ real and $\\times 10^n$', col(cols,arr,'F3inv_iso_re')),
    ]
    nonints=load_nonint(nonfile); by,order=load_raw_eigs(eigfile); Eb,branches=track_eigenbranches(by,order)
    fig,axs=plt.subplots(6,1,figsize=(15,12.5),sharex=True)
    fig.suptitle(f'v31zo F3inv diagnostics with digonto_classifier_v1: {args.label}, {args.waves}', y=0.995, fontsize=14)
    for k,(lab,y) in enumerate(series):
        ax=axs[k]
        ax.set_ylabel(lab)
        ax.grid(True,alpha=0.25)
        if not args.no_pdf_ylims:
            ax.set_ylim(*PDF_YLIMS[k])
        ax.axhline(0,lw=0.7,alpha=0.4)
        scatter_nonint(ax,nonints)
        if k==3:
            if branches.size:
                lo, hi = PDF_YLIMS[3]
                ylo, yhi = min(lo, hi)*1.05, max(lo, hi)*1.05
                for b in range(branches.shape[0]):
                    yy=branches[b,:].real
                    m=np.isfinite(Eb)&np.isfinite(yy)&(yy>=ylo)&(yy<=yhi)
                    if np.count_nonzero(m)>1:
                        ax.plot(Eb[m],yy[m],lw=0.55,alpha=0.65)
        else:
            plot_scaled(ax,E,y,args.nscale, ylim=None if args.no_pdf_ylims else PDF_YLIMS[k])
        if not args.no_candidate_lines:
            for c in candidates:
                ax.axvline(c['E_candidate'], color='darkgray', linestyle='--', lw=1.15, alpha=0.7, zorder=18)
        if not args.no_pdf_ylims:
            ax.set_ylim(*PDF_YLIMS[k])
    report = pathlib.Path(args.candidate_report) if args.candidate_report else pathlib.Path(args.png).with_suffix('.classified_projected_pole_candidates.dat') if args.png else outdir/f'{stem}_classified_projected_pole_candidates.dat'
    write_digonto_classifier_v1_report(report, [(args.tag, candidates)])
    print(f'[digonto_classifier_v1] wrote {report}')
    axs[-1].set_xlabel('Ecm')
    png=pathlib.Path(args.png) if args.png else outdir/f'{stem}_F3inv_6function_scaled_branch_diagnostics.png'
    fig.tight_layout(rect=[0,0,1,0.985]); fig.savefig(png,dpi=180); print(f'[v31zi-plot] wrote {png}')
    if args.show: plt.show()
    plt.close(fig)
if __name__=='__main__': main()
