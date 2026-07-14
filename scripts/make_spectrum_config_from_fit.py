#!/usr/bin/env python3
import argparse, re
from pathlib import Path

def read_summary(path):
    vals={}
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if not parts: continue
        if parts[0] in ("K3iso0","K3iso1","K3B","K3E"):
            vals[parts[0]]=parts[1]
    return vals

def replace_key(text,key,val):
    pat=re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.M)
    if pat.search(text): return pat.sub(rf"\g<1>{val}", text)
    return text+f"\n{key} = {val}\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    vals=read_summary(args.summary)
    txt=Path(args.template).read_text()
    for k in ("K3iso0","K3iso1","K3B","K3E"):
        txt=replace_key(txt,k,vals.get(k,'0.0'))
    Path(args.out).write_text(txt)
    print(f"[make-spectrum-config] wrote {args.out}")
    print("[make-spectrum-config] K3df:", vals)
if __name__=='__main__': main()
