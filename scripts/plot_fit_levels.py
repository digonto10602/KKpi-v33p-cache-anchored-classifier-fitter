#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        p=line.split()
        # row Lbyas label nPx nPy nPz state data_Ecm data_err model_Ecm residual_sigma
        rows.append((p[2], int(p[6]), float(p[7]), float(p[8]), float(p[9]), float(p[10])))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--levels', required=True)
    ap.add_argument('--png', required=True)
    args=ap.parse_args()
    rows=load(args.levels)
    rows=sorted(rows,key=lambda r:(r[0],r[1]))
    x=np.arange(len(rows))
    data=np.array([r[2] for r in rows])
    err=np.array([r[3] for r in rows])
    model=np.array([r[4] for r in rows])
    labels=[f"{r[0]}:{r[1]}" for r in rows]
    fig,ax=plt.subplots(figsize=(max(8,0.45*len(rows)),5.2))
    ax.errorbar(x,data,yerr=err,fmt='o',label='lattice')
    ax.scatter(x,model,marker='x',label='model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=70,ha='right',fontsize=8)
    ax.set_ylabel('Ecm')
    ax.set_title('v32f K3df fit: lattice levels vs model candidates')
    ax.grid(True,alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(args.png).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.png,dpi=180)
    print(f"[plot-fit-levels] wrote {args.png}")
if __name__=='__main__': main()
