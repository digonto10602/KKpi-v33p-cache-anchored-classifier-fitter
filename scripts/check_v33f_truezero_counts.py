#!/usr/bin/env python3
"""Lightweight checker for v33f count-matching runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from collections import Counter, defaultdict


def read_summary(path: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    if not path.exists():
        return d
    for line in path.read_text(errors='replace').splitlines():
        parts = line.split()
        if len(parts) >= 2:
            d[parts[0]] = parts[1]
    return d


def suffix_base(path: Path, suffix: str) -> Path:
    name = path.name
    if name.endswith(suffix):
        return path.with_name(name[:-len(suffix)])
    return path.with_suffix('')


def ltag(x: str) -> str:
    s = x
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s.replace('.', 'p')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--log', required=True)
    args = ap.parse_args()
    summary = Path(args.summary)
    log = Path(args.log)
    sd = read_summary(summary)
    pref = suffix_base(summary, '_fit_summary_allL.dat')
    levels_path = pref.with_name(pref.name + '_fit_levels_allL.dat')
    cands_path = pref.with_name(pref.name + '_all_QC_candidates_allL.dat')
    text = log.read_text(errors='replace') if log.exists() else ''
    ndata = int(float(sd.get('ndata', '0')))
    found = int(float(sd.get('model_levels_found', '0')))
    valid = int(float(sd.get('valid', '0')))
    blocks = re.findall(r'\[v32x-FCN\]\s+(L\S+)\s+true_zero=(\d+)\s+pole=(\d+)\s+uncertain=(\d+)', text)
    levels = []
    if levels_path.exists():
        for line in levels_path.read_text(errors='replace').splitlines():
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 11:
                levels.append({
                    'row': parts[0],
                    'Lbyas': parts[1],
                    'file_irrep': parts[2],
                    'internal_irrep': parts[3],
                    'state': parts[4],
                    'level_index': parts[5],
                    'lattice_Ecm': parts[6],
                    'model_Ecm': parts[8],
                })
    block_levels: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in levels:
        block_levels[(row['Lbyas'], row['internal_irrep'])].append(row)
    cands_by_block: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
    if cands_path.exists():
        for line in cands_path.read_text(errors='replace').splitlines():
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                cands_by_block[(parts[0], parts[2])].append(parts)
    print(f"[v33f-check] summary={summary}")
    print(f"[v33f-check] ndata={ndata} model_levels_found={found} valid={valid}")
    for b,tz,p,u in blocks:
        print(f"[v33f-check] {b}: true_zero={tz} pole={p} uncertain={u}")
    if found != ndata or valid != 1:
        print('[v33f-check] FAILED: model true zeros do not cover all selected lattice levels')
        first_bad = None
        for (Lbyas, irrep), rows in block_levels.items():
            model_found = sum(1 for r in rows if r['model_Ecm'] not in {'0', '0.0', '0.0000000000000000'})
            if model_found < len(rows):
                first_bad = (Lbyas, irrep, rows)
                break
        if first_bad is None and block_levels:
            (Lbyas, irrep), rows = next(iter(block_levels.items()))
            first_bad = (Lbyas, irrep, rows)
        if first_bad is not None:
            Lbyas, irrep, rows = first_bad
            dbgdir = summary.parents[1] / 'debug_windows'
            dbgdir.mkdir(parents=True, exist_ok=True)
            dbgpath = dbgdir / f"{ltag(Lbyas)}_{irrep}_mismatch.dat"
            with dbgpath.open('w') as f:
                f.write(f"# mismatch block Lbyas={Lbyas} irrep={irrep}\n")
                f.write("# row Lbyas file_irrep internal_irrep state level_index lattice_Ecm model_Ecm\n")
                for r in rows:
                    f.write(f"{r['row']} {r['Lbyas']} {r['file_irrep']} {r['internal_irrep']} {r['state']} {r['level_index']} {r['lattice_Ecm']} {r['model_Ecm']}\n")
                f.write("# candidate rows from all_QC_candidates_allL.dat\n")
                for parts in cands_by_block.get((Lbyas, irrep), []):
                    f.write(' '.join(parts) + '\n')
            print(f"[v33f-check] first_mismatch_block=Lbyas={Lbyas} irrep={irrep}")
            print(f"[v33f-check] wrote_debug_windows={dbgpath}")
            for parts in cands_by_block.get((Lbyas, irrep), [])[:12]:
                print("[v33f-check] candidate " + ' '.join(parts[:8]))
        return 2
    print('[v33f-check] PASSED: model true-zero count covers selected lattice levels')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
