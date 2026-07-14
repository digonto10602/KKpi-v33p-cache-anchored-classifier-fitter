#!/usr/bin/env python3
import argparse, math, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap=argparse.ArgumentParser(description='v32s irrep-filtered const-norm-scaled det(projF3inv) n-scale plot with true refined digonto_classifier_v2 zeros')
    ap.add_argument('--output-dir', default='output_v32s_projF3inv_norm_refine_v2')
    ap.add_argument('--tag', default='debug_v32s_projF3inv_norm_refine_v2')
    ap.add_argument('--irrep', required=True, help='Full label, e.g. 000_A1m, 110_A2')
    ap.add_argument('--lattice-targets', default=None)
    ap.add_argument('--nonint-dir', default='output_nonint_v32s')
    ap.add_argument('--nonint-tag', default='debug_v32s_nonint')
    ap.add_argument('--nonint-levels', default=None)
    ap.add_argument('--predictions', default=None, help='Optional explicit v2 refined candidate file')
    ap.add_argument('--nscale', type=int, default=0)
    ap.add_argument('--max-nscale-curves', type=int, default=101)
    ap.add_argument('--ycol', choices=['det_scaled','slogdet_scaled'], default='det_scaled')
    ap.add_argument('--no-legend', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--png', default=None)
    ap.add_argument('--require-nonint-file', action='store_true')
    return ap.parse_args()

def parse_label(label):
    if '_' not in label: raise ValueError(f'Bad irrep label {label}')
    return label.split('_',1)

def read_rows(path):
    rows=[]; path=Path(path)
    if not path.exists(): return rows
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            rows.append(line.split())
    return rows

def gf(row,i,default=math.nan):
    try: return float(row[i])
    except Exception: return default

def infer_nonint_file(nonint_dir, nonint_tag, full_label):
    ptag,ir=parse_label(full_label)
    candidates=[
        Path(nonint_dir)/f'{nonint_tag}_P{ptag}_{ir}_nonint3body.dat',
        Path(nonint_dir)/f'{nonint_tag}_P{ptag}_{ir.replace("m","-").replace("p","+")}_nonint3body.dat',
        Path(nonint_dir)/f'{nonint_tag}_P{ptag}_{ir.replace("m","m").replace("p","p")}_nonint3body.dat',
    ]
    for c in candidates:
        if c.exists(): return c
    return candidates[0]

def load_grid(path,ycol):
    rows=read_rows(path)
    arr=[]
    for r in rows:
        # columns: i Ecm success proj_dim det_scaled signed_slogdet_scaled logabsdet sign
        if len(r)<8: continue
        success=gf(r,2)
        if success < 0.5: continue
        E=gf(r,1)
        y=gf(r,4) if ycol=='det_scaled' else gf(r,5)
        if math.isfinite(E) and math.isfinite(y): arr.append((E,y))
    if not arr: raise RuntimeError(f'No usable grid rows in {path}')
    a=np.array(arr,float); a=a[np.argsort(a[:,0])]
    return a[:,0],a[:,1]

def load_lattice(path,full_label):
    if not path: return []
    ptag,ir=parse_label(full_label)
    vals=[]; scanned=matched=0
    for r in read_rows(path):
        scanned+=1
        # v32s columns: row Lbyas label irrep nPx nPy nPz state E_read err_read lattice_energy_type atP shifted Ecm err
        ok=False; e=math.nan
        if len(r)>=15:
            ok=(r[2]==full_label) or (r[3]==ir and r[2].startswith(ptag+'_'))
            e=gf(r,13)
        elif len(r)>=10:
            ok=(r[2]==full_label) or (r[3]==ir and r[2].startswith(ptag+'_'))
            e=gf(r,8)
        elif len(r)>=4:
            ok=(r[0]==full_label or r[2]==full_label)
            e=gf(r,-2)
        if ok:
            matched+=1
            if math.isfinite(e): vals.append(e)
    vals=sorted(vals)
    print(f'[plot] lattice filter: scanned={scanned} matched={matched} usable={len(vals)} file={path}')
    return vals

def load_nonint(path,Emin,Emax,require=False):
    path=Path(path)
    if not path.exists():
        msg=f'[plot-warning] nonint file not found: {path}'
        if require: raise RuntimeError(msg)
        print(msg); return []
    vals=[]
    for r in read_rows(path):
        # Common nonint3body output: level Ecm ... OR columns containing Ecm in col 1.
        if len(r)>=2 and re.fullmatch(r'[-+]?\d+',r[0]): e=gf(r,1)
        elif len(r)>=5: e=gf(r,4)
        else: e=math.nan
        if math.isfinite(e) and Emin-1e-12 <= e <= Emax+1e-12: vals.append(e)
    vals=sorted(set(round(v,14) for v in vals))
    print(f'[plot] nonint usable={len(vals)} file={path}')
    return vals

def load_true_zero_candidates(path):
    vals=[]; scanned=0
    for r in read_rows(path):
        scanned+=1
        # columns: id label irrep init_id depth bracket_L bracket_R flip_L flip_R E_zero yBL yFL yFR yBR kind reason
        if len(r)>=16:
            kind=r[14]
            e=gf(r,9)
            if kind=='true_zero' and math.isfinite(e): vals.append(e)
    print(f'[plot] true_zero classifier candidates usable={len(vals)} scanned={scanned} file={path}')
    return sorted(vals)

def main():
    args=parse_args()
    out=Path(args.output_dir)
    grid=out/f'{args.tag}_{args.irrep}_projF3inv_det_grid.dat'
    if not grid.exists(): raise FileNotFoundError(f'Missing selected-irrep grid: {grid}')
    E,y=load_grid(grid,args.ycol)
    Emin,Emax=float(np.min(E)),float(np.max(E))
    latfile=args.lattice_targets or str(out/f'{args.tag}_lattice_targets.dat')
    lattice=load_lattice(latfile,args.irrep)
    nonintfile=Path(args.nonint_levels) if args.nonint_levels else infer_nonint_file(args.nonint_dir,args.nonint_tag,args.irrep)
    nonint=load_nonint(nonintfile,Emin,Emax,args.require_nonint_file)
    candfile=Path(args.predictions) if args.predictions else out/f'{args.tag}_{args.irrep}_digonto_classifier_v2_refined_candidates.dat'
    zeros=load_true_zero_candidates(candfile) if candfile.exists() else []
    print(f'[plot] selected irrep={args.irrep} grid_rows={len(E)} lattice={len(lattice)} nonint={len(nonint)} true_zero_candidates={len(zeros)} ycol={args.ycol}')

    nmax=max(0,args.nscale)
    if nmax+1>args.max_nscale_curves:
        print(f'[plot-warning] capping nscale from {nmax} to {args.max_nscale_curves-1}')
        nmax=args.max_nscale_curves-1

    fig,ax=plt.subplots(1,1,figsize=(13,6),constrained_layout=True)
    for x in nonint:
        ax.axvline(x,color='black',linestyle='--',linewidth=1.0,alpha=1.0,zorder=0)
    for x in zeros:
        ax.axvline(x,color='gray',linestyle='--',linewidth=1.2,alpha=1.0,zorder=1)
    if lattice:
        ax.scatter(lattice,[0.0]*len(lattice),s=55,facecolors='white',edgecolors='black',linewidths=1.2,alpha=1.0,zorder=10,label='lattice Ecm')
    for n in range(nmax+1):
        yy=y*(10.0**n)
        yy[~np.isfinite(yy)]=np.nan
        ax.plot(E,yy,linewidth=1.0,alpha=1.0,label=(f'10^{n} f(E)' if n else 'f(E)'))
    ax.axhline(0,color='black',linewidth=0.8,alpha=1.0)
    ylabel={'det_scaled':r'$\det[(V^\dagger F_3^{-1}V)/(L\xi)^6]$', 'slogdet_scaled':r'$\mathrm{slogdet}[(V^\dagger F_3^{-1}V)/(L\xi)^6]$'}[args.ycol]
    ax.set_ylabel(ylabel); ax.set_xlabel(r'$E_{cm}$')
    ax.set_title(f'v32s true-refined digonto_classifier_v2 projected F3inv: {args.irrep}')
    ax.grid(True,alpha=1.0,linewidth=0.4)
    if not args.no_legend: ax.legend(fontsize=8,ncol=4)
    png=Path(args.png) if args.png else Path('output_plots')/f'v32s_{args.irrep}_projF3inv_{args.ycol}_nscale{nmax}.png'
    png.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(png,dpi=220)
    print(f'[plot] wrote {png}')
    if args.show: plt.show()
    plt.close(fig)
if __name__=='__main__': main()
