#!/usr/bin/env python3
"""Independent plotting script for v31zd projected F3^{-1} diagnostics.
Can be run without rerunning the C++ executable as long as the output/*.dat files exist.
"""
import argparse, pathlib, re, math, sys, os
import numpy as np
import matplotlib.pyplot as plt


def parse_columns(path):
    cols=None
    with open(path) as f:
        for line in f:
            if line.startswith('# columns:'):
                cols=line.split(':',1)[1].strip().split()
    return cols or []


def load_grid(path):
    cols=parse_columns(path)
    rows=[]
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p=line.split()
            vals=[]
            for x in p[:len(cols)]:
                try: vals.append(float(x))
                except Exception: vals.append(np.nan)
            if len(vals)==len(cols): rows.append(vals)
    arr=np.array(rows,float) if rows else np.empty((0,len(cols)))
    return cols, arr


def col(cols, arr, name, fallback_idx=None):
    if name in cols:
        return arr[:, cols.index(name)]
    if fallback_idx is not None and arr.shape[1] > fallback_idx:
        return arr[:, fallback_idx]
    raise KeyError(f"Column {name} not found in grid file; available={cols}")


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
    if not path.exists(): return {}, []
    by={}
    order=[]
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            p=line.split()
            if len(p)<6: continue
            try:
                i=int(float(p[0])); E=float(p[1]); z=complex(float(p[3]),float(p[4]))
            except Exception:
                continue
            if i not in by:
                by[i]=[]; order.append((i,E))
            by[i].append(z)
    order.sort()
    for k in by: by[k]=np.array(by[k],complex)
    return by, order


def track_eigenbranches(by, order):
    if not order: return np.array([]), np.empty((0,0),complex)
    Es=np.array([E for _,E in order],float)
    maxdim=max(len(by[i]) for i,_ in order)
    branches=np.full((maxdim,len(order)), np.nan+1j*np.nan, complex)
    prev=None
    prev_assigned=[]
    for t,(i,E) in enumerate(order):
        vals=np.array(by.get(i,[]),complex)
        if vals.size==0: continue
        # start / after dimension jump: sort by real part then imag part for deterministic order
        if prev is None:
            ord0=np.lexsort((vals.imag, vals.real))
            for b,j in enumerate(ord0[:maxdim]): branches[b,t]=vals[j]
            prev=branches[:,t].copy(); continue
        used=set()
        newcol=np.full(maxdim,np.nan+1j*np.nan,complex)
        # greedy nearest-neighbor assignment from previous branches
        for b,zprev in enumerate(prev):
            if not np.isfinite(zprev.real): continue
            if vals.size==0: break
            d=np.abs(vals-zprev)
            for u in used: d[u]=np.inf
            j=int(np.argmin(d))
            if np.isfinite(d[j]):
                newcol[b]=vals[j]; used.add(j)
        # remaining new branches
        free=[b for b in range(maxdim) if not np.isfinite(newcol[b].real)]
        remain=[j for j in range(vals.size) if j not in used]
        remain=sorted(remain, key=lambda j:(vals[j].real, vals[j].imag))
        for b,j in zip(free,remain): newcol[b]=vals[j]
        branches[:,t]=newcol
        prev=newcol.copy()
    return Es, branches


def plot_scaled(ax, E, y, nscale, label, use_abs=False):
    y=np.array(y,float)
    for n in range(nscale+1):
        yy=y*(10.0**n)
        ax.plot(E, yy, lw=0.85, alpha=0.85, label=(label if n==0 else f"x1e{n}"))
    if nscale<=4:
        ax.legend(fontsize=7,ncol=min(nscale+1,5))


def scatter_nonint(ax, nonints):
    if not nonints: return
    ax.scatter(nonints, [0.0]*len(nonints), s=58, facecolors='white', edgecolors='darkred', linewidths=1.6, zorder=8, label='non-interacting')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='output')
    ap.add_argument('--tag', default='debug_v31z_projF3inv_zero_110A2_L24')
    ap.add_argument('--label', default='110_A2')
    ap.add_argument('--waves', default='waves_0_1')
    ap.add_argument('--nscale', type=int, default=10)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--png', default='')
    args=ap.parse_args()
    outdir=pathlib.Path(args.outdir)
    stem=f'{args.tag}_{args.label}_{args.waves}'
    grid=outdir/f'{stem}_projF3inv_grid.dat'
    eigfile=outdir/f'{stem}_projF3inv_eigenvalues_raw.dat'
    nonfile=outdir/f'{args.tag}_{args.label}_nonint_group_degeneracies.dat'
    if not grid.exists():
        raise SystemExit(f'Grid file not found: {grid}')
    cols, arr=load_grid(grid)
    if arr.size==0: raise SystemExit(f'No numeric rows in {grid}')
    E=col(cols,arr,'Ecm')
    det_full=col(cols,arr,'detFullF3inv_signed_logabs', None)
    det_proj=col(cols,arr,'signed_logabs')
    min_eig=col(cols,arr,'minAbsEig')
    min_sv=col(cols,arr,'minSVprojF3inv')
    nonints=load_nonint(nonfile)
    by,order=load_raw_eigs(eigfile)
    Eb,branches=track_eigenbranches(by,order)

    fig,axs=plt.subplots(5,1,figsize=(15,18),sharex=True)
    fig.suptitle(f'v31zd F3inv diagnostics: {args.label}, {args.waves}', y=0.995, fontsize=14)

    plot_scaled(axs[0],E,det_full,args.nscale,'signed log det full F3inv')
    axs[0].set_ylabel('det(F3^{-1}) signed log |det| and x10^n')
    scatter_nonint(axs[0],nonints)

    plot_scaled(axs[1],E,det_proj,args.nscale,'signed log det projected F3inv')
    axs[1].set_ylabel('det(proj F3^{-1}) signed log |det| and x10^n')
    scatter_nonint(axs[1],nonints)

    plot_scaled(axs[2],E,min_eig,args.nscale,'min |eig|')
    axs[2].set_ylabel('smallest eigenvalue diagnostic min|λ| and x10^n')
    scatter_nonint(axs[2],nonints)

    if branches.size:
        for b in range(branches.shape[0]):
            y=branches[b,:].real
            if np.isfinite(y).sum()>1:
                axs[3].plot(Eb,y,lw=0.7,alpha=0.65)
    axs[3].axhline(0.0,lw=0.8,alpha=0.5)
    axs[3].set_ylabel('tracked Re eigenvalue branches')
    scatter_nonint(axs[3],nonints)

    plot_scaled(axs[4],E,min_sv,args.nscale,'minSV')
    axs[4].set_ylabel('minSVprojF3inv and x10^n')
    scatter_nonint(axs[4],nonints)
    axs[4].set_xlabel('Ecm')

    for ax in axs:
        ax.grid(True,alpha=0.25)
        ax.axhline(0.0,lw=0.7,alpha=0.4)
        # Prevent a single scaled curve from making the plot unreadable if nscale is large.
        ylims=ax.get_ylim()
        if not all(np.isfinite(ylims)) or ylims[0]==ylims[1]:
            pass
    png=pathlib.Path(args.png) if args.png else outdir/f'{stem}_F3inv_5function_scaled_branch_diagnostics.png'
    fig.tight_layout(rect=[0,0,1,0.987])
    fig.savefig(png,dpi=180)
    print(f'[v31zd-plot] wrote {png}')
    if args.show:
        try: plt.show()
        except Exception as e: print(f'[v31zd-plot-warning] could not show plot interactively: {e}')
    plt.close(fig)

if __name__=='__main__':
    main()
