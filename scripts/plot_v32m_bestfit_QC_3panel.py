#!/usr/bin/env python3
import argparse, glob, math, os, re
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt


def parse_cols(path):
    cols=None
    with open(path) as f:
        for line in f:
            if line.startswith('#') and 'columns:' in line:
                cols=line.split('columns:',1)[1].strip().split()
    if cols is None:
        raise RuntimeError(f'No # columns line found in {path}')
    return cols


def load_table(path):
    cols=parse_cols(path)
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            parts=line.split()
            if len(parts) < len(cols):
                continue
            rows.append(parts[:len(cols)])
    out={c:[] for c in cols}
    for row in rows:
        for c,x in zip(cols,row):
            try:
                out[c].append(float(x))
            except ValueError:
                out[c].append(x)
    for c in cols:
        try:
            out[c]=np.asarray(out[c], dtype=float)
        except Exception:
            out[c]=np.asarray(out[c], dtype=object)
    if 'Ecm' in out and len(out['Ecm'])>1:
        E=np.asarray(out['Ecm'], dtype=float)
        order=np.argsort(E, kind='mergesort')
        for c in list(out.keys()):
            if len(out[c])==len(order):
                out[c]=out[c][order]
    return cols,out


def label_from_grid(path, tag):
    name=Path(path).name
    pre=f'{tag}_'
    suf='_bestfit_QC_closest_eig_grid.dat'
    if name.startswith(pre) and name.endswith(suf):
        return name[len(pre):-len(suf)]
    return name.replace(suf,'')


def read_fit_levels(path):
    if not path or not Path(path).exists():
        return []
    cols,tab=load_table(path)
    rows=[]
    n=len(tab.get('data_Ecm',[]))
    for i in range(n):
        rows.append({c:tab[c][i] for c in cols})
    return rows


def parse_total_momentum(label):
    m=re.match(r'^(\d)(\d)(\d)_', label)
    if not m:
        return (0,0,0)
    return tuple(int(x) for x in m.groups())


def nonint_2Kpi_levels(label, mK=0.09698, mpi=0.06906, xi=3.444, Lbyas=20.0, Emin=0.261, Emax=0.36, nmax=5):
    d=parse_total_momentum(label)
    mom_unit=2.0*math.pi/(xi*Lbyas)
    d2=sum(x*x for x in d)
    out=[]
    rng=range(-nmax,nmax+1)
    vecs=[(i,j,k) for i in rng for j in rng for k in rng]
    for n1 in vecs:      # K
        for n2 in vecs:  # K
            n3=(d[0]-n1[0]-n2[0], d[1]-n1[1]-n2[1], d[2]-n1[2]-n2[2]) # pi
            if any(abs(x)>nmax for x in n3):
                continue
            def omega(m,n):
                n2s=sum(a*a for a in n)
                return math.sqrt(m*m + (mom_unit*mom_unit)*n2s)
            Elab=omega(mK,n1)+omega(mK,n2)+omega(mpi,n3)
            val=Elab*Elab - mom_unit*mom_unit*d2
            if val<=0: continue
            Ecm=math.sqrt(val)
            if Emin-1e-10 <= Ecm <= Emax+1e-10:
                out.append(Ecm)
    out=sorted(out)
    uniq=[]
    for x in out:
        if not uniq or abs(x-uniq[-1])>1e-8:
            uniq.append(x)
    return uniq


def finite_mask(x,y, ylim=None):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    m=np.isfinite(x)&np.isfinite(y)
    if ylim is not None:
        lo,hi=ylim
        m &= (y>=lo)&(y<=hi)
    return x[m],y[m]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='output_fit')
    ap.add_argument('--tag', default='debug_v32m_K3df_fit_fullF3inv_QCfull_oldclassifier')
    ap.add_argument('--fit-levels', default='')
    ap.add_argument('--png', default='output_plots/v32m_bestfit_QC_3panel.png')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--mK', type=float, default=0.09698)
    ap.add_argument('--mpi', type=float, default=0.06906)
    ap.add_argument('--xi', type=float, default=3.444)
    ap.add_argument('--Lbyas', type=float, default=20.0)
    args=ap.parse_args()

    pattern=str(Path(args.output_dir)/f'{args.tag}_*_bestfit_QC_closest_eig_grid.dat')
    files=sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'No grid files found with pattern: {pattern}')

    levels=read_fit_levels(args.fit_levels)
    E_latt=[]
    for r in levels:
        try:
            E_latt.append(float(r['data_Ecm']))
        except Exception:
            pass
    E_latt=np.asarray(E_latt,dtype=float)

    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,constrained_layout=True)
    panels=[
        ('signed_slogdet_QC','signed slogdet det(projected QC)'),
        ('minAbsEig','min |eigenvalue(projected QC)|'),
        ('minSVprojQC','min SV(projected QC)'),
    ]

    all_E=[]
    labels=[]
    for f in files:
        lab=label_from_grid(f,args.tag)
        labels.append(lab)
        cols,tab=load_table(f)
        E=tab['Ecm']
        all_E.extend([float(np.nanmin(E)), float(np.nanmax(E))])
        for ax,(key,ylabel) in zip(axes,panels):
            if key not in tab:
                continue
            x,y=finite_mask(E,tab[key])
            ax.plot(x,y,lw=0.85,alpha=0.9,label=lab)
            ax.set_ylabel(ylabel)
            ax.axhline(0.0,lw=0.7,alpha=0.4)

    Emin=min(all_E) if all_E else 0.261
    Emax=max(all_E) if all_E else 0.36

    # Black vertical non-interacting levels for the momenta/irreps present.
    ni_all=[]
    for lab in labels:
        ni_all += nonint_2Kpi_levels(lab,args.mK,args.mpi,args.xi,args.Lbyas,Emin,Emax,nmax=5)
    ni_all=sorted(ni_all)
    uniq=[]
    for x in ni_all:
        if not uniq or abs(x-uniq[-1])>2e-7:
            uniq.append(x)
    for ax in axes:
        for x in uniq:
            ax.axvline(x,color='black',lw=0.55,alpha=0.45,zorder=0)
        if E_latt.size:
            y0=np.zeros_like(E_latt)
            ax.scatter(E_latt,y0,s=34,facecolors='white',edgecolors='black',marker='o',zorder=10,label='lattice Ecm')
        ax.grid(True,alpha=0.22)

    axes[-1].set_xlabel('Ecm')
    axes[0].set_title('v32m best-fit projected QC diagnostics: old classifier on slogdet(det projected QC)')
    handles,labtxt=axes[0].get_legend_handles_labels()
    # Keep legend readable.
    if len(handles) <= 8:
        axes[0].legend(loc='best',fontsize=8,ncol=2)
    else:
        axes[0].legend(handles[:8],labtxt[:8],loc='best',fontsize=8,ncol=2,title='first entries')

    Path(args.png).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.png,dpi=220)
    print(f'[plot-v32m] wrote {args.png}')
    if args.show:
        plt.show()

if __name__=='__main__':
    main()
