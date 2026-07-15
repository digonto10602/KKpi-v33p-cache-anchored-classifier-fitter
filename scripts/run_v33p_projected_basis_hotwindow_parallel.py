#!/usr/bin/env python3
"""Run the trusted hot-window basis builder in bounded row chunks and merge it."""
import argparse
import csv
import json
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


FIELDS = ["Lbyas", "irrep", "lattice_level_index", "lattice_Ecm", "bracket_id",
          "E_left_bracket", "E_right_bracket", "zero_estimate_initial", "center_row",
          "row_left", "row_right", "max_row_left", "max_row_right",
          "previous_model_zero", "has_previous_model_zero", "inside_Ecm_cutoff"]


def split_windows(path, jobs):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    indices = []
    for r in rows:
        for i in range(int(r["row_left"]), int(r["row_right"]) + 1):
            indices.append((r, i))
    chunks = [[] for _ in range(min(jobs, max(1, len(indices))))]
    for i, item in enumerate(indices):
        chunks[i % len(chunks)].append(item)
    return rows, chunks


def write_chunk(path, items):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for source, row in items:
            r = dict(source)
            r["row_left"] = str(row)
            r["row_right"] = str(row)
            r["max_row_left"] = str(row)
            r["max_row_right"] = str(row)
            w.writerow(r)


def run_chunk(builder, root, chunk_dir, windows, common):
    cmd = [builder, "--gpu-cache-root", root, "--outdir", str(chunk_dir),
           "--accepted-windows", str(windows), "--accepted-windows-sha256", common["sha"],
           "--git-commit", common["git"], "--Lbyas", str(common["Lbyas"]),
           "--irrep", common["irrep"], "--Emin", "0.26310", "--Emax", "0.36",
           "--coarseN", "20000", "--xi", "3.444",
           "--K3iso0", "73735.840894011912", "--K3iso1", "-972421.14060757787",
           "--K3B", "347174.05548116949", "--K3E", "-1226756.7068845264"]
    env = dict(__import__("os").environ)
    env["OMP_NUM_THREADS"] = "4"
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def merge(outdir, chunk_dirs, count_expected):
    outdir = Path(outdir)
    parts = []
    validations = []
    metas = []
    for d in chunk_dirs:
        b = (d / "projected_basis_hotwindows.bin").read_bytes()
        if b[:8] != b"V33PHWB1":
            raise RuntimeError(f"bad binary magic in {d}")
        version, count = struct.unpack_from("<II", b, 8)
        if version != 1:
            raise RuntimeError(f"unsupported binary version in {d}")
        if count == 0:
            raise RuntimeError(f"empty chunk {d}")
        parts.append(b[16:])
        with (d / "projected_basis_hotwindows_validation.csv").open() as f:
            rr = list(csv.DictReader(f))
        validations.extend(rr)
        metas.append(json.loads((d / "projected_basis_hotwindows.json").read_text()))
    if len(validations) != count_expected:
        raise RuntimeError(f"merged row count {len(validations)} != expected {count_expected}")

    final_bin = outdir / "projected_basis_hotwindows.bin"
    magic = b"V33PHWB1" + struct.pack("<II", 1, len(validations))
    final_bin.write_bytes(magic + b"".join(parts))
    final_csv = outdir / "projected_basis_hotwindows_validation.csv"
    with final_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(validations[0]))
        w.writeheader()
        w.writerows(sorted(validations, key=lambda r: int(r["row_index"])))

    base = metas[0]
    rows = sorted(validations, key=lambda r: int(r["row_index"]))
    base["n_rows_saved"] = len(rows)
    base["row_indices"] = [int(r["row_index"]) for r in rows]
    base["Ecm_values"] = [float(r["Ecm"]) for r in rows]
    base["Ecm_min"] = min(base["Ecm_values"])
    base["Ecm_max"] = max(base["Ecm_values"])
    base["Nproj_values"] = [int(r["Nproj"]) for r in rows]
    base["matrix_dimension_values"] = [int(r["Nfull"]) for r in rows]
    base["validation"] = {
        "max_matrix_abs_diff": max(float(r["matrix_max_abs_diff"]) for r in rows),
        "max_det_abs_diff": max(float(r["det_abs_diff"]) for r in rows),
        "max_det_rel_diff": max(float(r["det_rel_diff"]) for r in rows),
        "sign_agreement": f"{sum(int(r['sign_agreement']) for r in rows)}/{len(rows)}",
        "rows_checked": len(rows)}
    base["file_sizes"] = {"bin_size_bytes": final_bin.stat().st_size,
                           "json_size_bytes": 0}
    json_path = outdir / "projected_basis_hotwindows.json"
    json_path.write_text(json.dumps(base, indent=2) + "\n")
    base["file_sizes"]["json_size_bytes"] = json_path.stat().st_size
    json_path.write_text(json.dumps(base, indent=2) + "\n")
    v = base["validation"]
    status = (v["max_matrix_abs_diff"] < 1e-10 and
              v["max_det_abs_diff"] < 1e-15 and
              v["sign_agreement"] == f"{len(rows)}/{len(rows)}")
    (outdir / "PROJECTED_BASIS_HOTWINDOW_CACHE_REPORT.md").write_text(
        "# Projected-basis hot-window cache\n\n"
        f"Status: {'PROJECTED_BASIS_HOTWINDOW_CACHE_PASS' if status else 'PROJECTED_BASIS_HOTWINDOW_CACHE_FAIL'}\n\n"
        f"- Sector: L{base['Lbyas']}/{base['irrep']}\n"
        f"- Rows saved: {len(rows)}\n"
        f"- Max matrix abs diff: {v['max_matrix_abs_diff']}\n"
        f"- Max raw determinant abs diff: {v['max_det_abs_diff']}\n"
        f"- Max raw determinant relative diff: {v['max_det_rel_diff']}\n"
        f"- Sign agreement: {v['sign_agreement']}\n"
        f"- Binary size: {final_bin.stat().st_size} bytes\n"
        f"- JSON size: {json_path.stat().st_size} bytes\n"
        "- Raw determinant is primary; logdet/logabs are validation-only.\n"
        "- No cachegen or original-cache regeneration was run.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", default="bin/v33p_build_projected_basis_hotwindow_cache")
    ap.add_argument("--gpu-cache-root", default="/media/digonto/Data/F3inv_cache")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--Lbyas", type=int, required=True)
    ap.add_argument("--irrep", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--git", required=True)
    ap.add_argument("--sha", required=True)
    a = ap.parse_args()
    rows, chunks = split_windows(a.windows, a.jobs)
    if not rows or not chunks:
        raise SystemExit("no accepted hot-window rows")
    outdir = Path(a.outdir)
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_root = outdir / ".chunks"
    chunk_root.mkdir()
    chunk_paths = []
    for i, items in enumerate(chunks):
        d = chunk_root / f"chunk_{i:02d}"
        d.mkdir()
        p = d / "accepted_windows.csv"
        write_chunk(p, items)
        chunk_paths.append((d, p))
    common = {"sha": a.sha, "git": a.git, "Lbyas": a.Lbyas, "irrep": a.irrep}
    with ThreadPoolExecutor(max_workers=len(chunk_paths)) as pool:
        futures = [pool.submit(run_chunk, a.builder, a.gpu_cache_root, d, p, common)
                   for d, p in chunk_paths]
        for f in futures:
            f.result()
    merge(outdir, [d for d, _ in chunk_paths], len(sum(chunks, [])))
    shutil.rmtree(chunk_root)


if __name__ == "__main__":
    main()
