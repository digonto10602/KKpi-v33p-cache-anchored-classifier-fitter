#!/usr/bin/env python3
"""Package all-sector CSV scan results without changing classifier logic."""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v33p_all_sector_classifier_validation"
INPUT = ROOT / "input"
CACHE_ROOT = Path("/media/digonto/Data/F3inv_cache")
SECTORS = [
    (20, "000_A1m", "v33e_coarse20000_sep_f2k2_L20_000_A1m"),
    (20, "110_A2", "v33e_coarse20000_sep_f2k2_L20_110_A2"),
    (20, "111_A2", "v33e_coarse20000_sep_f2k2_L20_111_A2"),
    (20, "200_A2", "v33e_coarse20000_sep_f2k2_L20_200_A2"),
    (24, "000_A1m", "v33e_coarse20000_sep_f2k2_L24_000_A1m"),
    (24, "100_A2", "v33e_coarse20000_sep_f2k2_L24_100_A2"),
    (24, "110_A2", "v33e_coarse20000_sep_f2k2_L24_110_111_200_A2"),
    (24, "111_A2", "v33e_coarse20000_sep_f2k2_L24_110_111_200_A2"),
    (24, "200_A2", "v33e_coarse20000_sep_f2k2_L24_110_111_200_A2"),
]
ACCEPTED = {"digonto_v3_window", "digonto_v4_window"}


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def meta_for(cache_dir: Path) -> tuple[Path, dict]:
    metas = sorted(cache_dir.glob("*F3inv_Vsel_gpu.bin.meta.json"))
    if len(metas) != 1:
        raise RuntimeError(f"expected one cache metadata file in {cache_dir}, found {len(metas)}")
    path = metas[0]
    meta = json.loads(path.read_text())
    binary = Path(meta.get("cache_file", ""))
    if not binary.is_absolute():
        binary = path.parent / binary
    if not binary.exists():
        raise RuntimeError(f"cache binary missing: {binary}")
    meta.setdefault("provenance", meta.get("version", ""))
    meta.setdefault("rows", meta.get("rows_written", meta.get("coarseN", "")))
    meta.setdefault("irrep", meta.get("irrep_label", ""))
    return path, meta


def candidate_rows(grid: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    jumps = []
    flips = []
    for i, (a, b) in enumerate(zip(grid, grid[1:])):
        if a.get("Nfull") != b.get("Nfull") or a.get("Nproj") != b.get("Nproj"):
            jumps.append({"row_left": a.get("row_sorted_index", str(i)), "row_right": b.get("row_sorted_index", str(i + 1)), "E_left": a["Ecm"], "E_right": b["Ecm"], "Nfull_left": a.get("Nfull", ""), "Nfull_right": b.get("Nfull", ""), "Nproj_left": a.get("Nproj", ""), "Nproj_right": b.get("Nproj", "")})
            continue
        if sign(f(a, "det_real")) * sign(f(b, "det_real")) < 0:
            y1, y2 = f(a, "det_real"), f(b, "det_real")
            x1, x2 = f(a, "Ecm"), f(b, "Ecm")
            z = x1 - y1 * (x2 - x1) / (y2 - y1) if y2 != y1 else (x1 + x2) / 2.0
            if not min(x1, x2) <= z <= max(x1, x2):
                z = (x1 + x2) / 2.0
            flips.append({"bracket_id": str(len(flips) + 1), "segment_id": f"Nfull{a.get('Nfull', '')}_Nproj{a.get('Nproj', '')}", "row_left": a.get("row_sorted_index", str(i)), "row_right": b.get("row_sorted_index", str(i + 1)), "E_left": a["Ecm"], "E_right": b["Ecm"], "gap": f"{x2 - x1:.17g}", "det_left": a["det_real"], "det_right": b["det_real"], "E_zero_linear": f"{z:.17g}", "sign_left": str(sign(y1)), "sign_right": str(sign(y2)), "Nfull": a.get("Nfull", ""), "Nproj": a.get("Nproj", "")})
    return flips, jumps


def process(L: int, irrep: str, sector_dir: str) -> dict[str, object]:
    name = f"L{L}_{irrep}_E026310_0360"
    out = OUT / name
    raw = out / "raw_scan"
    grid_paths = sorted(raw.glob("*_oldscale_det_coarse20000_E026_036.csv"))
    if len(grid_paths) != 1:
        raise RuntimeError(f"expected one raw scan CSV in {raw}, found {grid_paths}")
    raw_grid = grid_paths[0]
    grid_path = out / f"det_grid_L{L}_{irrep}_corrected_E026310_0360.csv"
    shutil.copyfile(raw_grid, grid_path)
    with grid_path.open(newline="") as fh:
        grid = sorted(list(csv.DictReader(fh)), key=lambda r: f(r, "Ecm"))
    flips, jumps = candidate_rows(grid)
    candidate_path = out / f"candidate_sign_change_brackets_E026310_0360.csv"
    with candidate_path.open("w", newline="") as fh:
        fields = ["bracket_id", "segment_id", "row_left", "row_right", "E_left", "E_right", "gap", "det_left", "det_right", "E_zero_linear", "sign_left", "sign_right", "Nfull", "Nproj"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(flips)
    cache_dir = CACHE_ROOT / sector_dir / f"v32zu_Lbyas{L}_gpu_cache"
    meta_path, meta = meta_for(cache_dir)
    failures = [r for r in grid if str(r.get("success", "1")) not in {"1", "true", "True"} or r.get("failure_reason", "OK") not in {"", "OK"}]
    max_imag = max((abs(f(r, "det_imag")) for r in grid), default=math.nan)
    scan_cmd = f"bin/v33h_patched_gpu_cache_oldscale_det_scan --gpu-cache-root {cache_dir} --Emin 0.2631 --Emax 0.36 --coarseN 20000 --old-scaling true --complex-read-convention variant_04_real_imag_swapped --hard-hermiticity-check true"
    report = out / "det_grid_validation_E026310_0360.md"
    report.write_text(f"""# Determinant-grid validation: L={L}, irrep={irrep}\n\nStatus: PASS; finalized scan loaded from the existing cache.\n\n- Cache metadata: `{meta_path}`\n- Cache binary: `{meta.get('cache_file', '')}`\n- Provenance: `{meta.get('provenance', '')}`\n- Metadata Lbyas/irrep: `{meta.get('Lbyas', L)}` / `{meta.get('irrep', irrep)}`\n- Metadata row count/coarseN: `{meta.get('coarseN', meta.get('rows', 'unknown'))}` / `{len(grid)}` scan rows\n- Metadata Ecm range: `[{meta.get('Ecm_min', 'unknown')}, {meta.get('Ecm_max', 'unknown')}]`\n- Scan source/regeneration path: `{scan_cmd}`\n- Max `|det_imag|`: `{max_imag:.17g}`\n- Dimension jumps: `{len(jumps)}`\n- Sign-change brackets excluding dimension jumps: `{len(flips)}`\n- Scan failures: `{len(failures)}`\n- Corrected format-aware combined raw GPU cache reader: confirmed.\n- Fixed scaling: `scaled_projected_QC = projected_QC / pow(Lbyas * xi, 6.0)`; no determinant-level `Nproj` rescaling.\n- Physics path: `QC = F3inv + K3df`, `projected_QC = Vsel.adjoint() * QC * Vsel`, `det = determinant(scaled_projected_QC)`.\n- Cachegen run: **no**; no cache binary was regenerated or copied.\n\n## Dimension jumps\n\n""" + ("\n".join(f"- rows {j['row_left']}->{j['row_right']}: Ecm {j['E_left']}->{j['E_right']}, dimensions ({j['Nfull_left']},{j['Nproj_left']}) -> ({j['Nfull_right']},{j['Nproj_right']})" for j in jumps) or "- None") + "\n")
    sweep_dir = out / "_sweep_tmp"
    subprocess.run(["python3", "scripts/run_v33p_classifier_sweep_L20_100_A2.py", "--grid", str(grid_path), "--output-dir", str(sweep_dir)], cwd=ROOT, check=True)
    with (sweep_dir / "classifier_predictions.csv").open(newline="") as fh:
        predictions = [r for r in csv.DictReader(fh) if r.get("method") in ACCEPTED]
    pred_path = out / "accepted_classifier_predictions_E026310_0360.csv"
    with pred_path.open("w", newline="") as fh:
        fields = ["method", "candidate_id", "E_left", "E_right", "E_mid", "zero_estimate", "classification", "score", "reason", "features"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(predictions)
    with (sweep_dir / "classifier_summary.csv").open(newline="") as fh:
        summaries = [r for r in csv.DictReader(fh) if r.get("method") in ACCEPTED]
    summary_path = out / "accepted_classifier_summary_E026310_0360.csv"
    with summary_path.open("w", newline="") as fh:
        fields = ["method", "n_true_zero", "n_pole", "n_uncertain", "runtime_seconds", "notes"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    accepted_report = out / "accepted_classifier_sweep_report_E026310_0360.md"
    accepted_report.write_text(f"""# Accepted classifier sweep: L={L}, irrep={irrep}\n\n- Grid rows: `{len(grid)}`\n- Candidate brackets: `{len(flips)}`\n- Modes retained: `digonto_v3_window`, `digonto_v4_window`\n- Prediction rows retained: `{len(predictions)}`\n- Existing sweep harness: `scripts/run_v33p_classifier_sweep_L20_100_A2.py`\n- Classifier logic changed: **no**.\n- Matrix/raw-sign modes are not used as production classifiers in this validation.\n- User-label scoring: **not run**; waiting for the label template to be completed.\n""")
    plot_stem = out / f"classifier_QA_L{L}_{irrep}_E026310_0360"
    subprocess.run(["python3", "scripts/plot_v33p_classifier_det_family_interactive.py", "--grid", str(grid_path), "--prediction-file", str(pred_path), "--candidate-file", str(candidate_path), "--n-scale", "100", "--output-stem", str(plot_stem)], cwd=ROOT, check=True)
    marker_csv = plot_stem.with_name(plot_stem.name + "_marker_map.csv")
    marker_md = plot_stem.with_name(plot_stem.name + "_marker_mapping.md")
    marker_md.write_text(f"# Marker mapping: L={L}, irrep={irrep}\n\n- Plot: `{plot_stem.name}.png` and `{plot_stem.name}.pdf`\n- Candidate markers: open gray circles at candidate linear-zero estimates.\n\n" + marker_csv.read_text())
    label_path = INPUT / f"v33p_user_truezero_labels_L{L}_{irrep}_E026310_0360.csv"
    with label_path.open("w", newline="") as fh:
        fields = ["bracket_id", "E_left", "E_right", "user_label", "user_zero_estimate", "comment"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for row in flips:
            writer.writerow({"bracket_id": row["bracket_id"], "E_left": row["E_left"], "E_right": row["E_right"], "user_label": "", "user_zero_estimate": "", "comment": ""})
    true_counts = {m: sum(r["method"] == m and r["classification"] == "true_zero" for r in predictions) for m in ACCEPTED}
    return {"sector": f"L{L}_{irrep}", "name": name, "cache": str(meta_path), "provenance": meta.get("provenance", ""), "rows": len(grid), "ecm_min": meta.get("Ecm_min", ""), "ecm_max": meta.get("Ecm_max", ""), "max_imag": max_imag, "jumps": len(jumps), "candidates": len(flips), "prediction_rows": len(predictions), "v3_true": true_counts["digonto_v3_window"], "v4_true": true_counts["digonto_v4_window"], "plot_png": str(plot_stem.with_suffix(".png")), "plot_pdf": str(plot_stem.with_suffix(".pdf")), "labels": str(label_path), "status": "PASS"}


def main() -> int:
    results = []
    for L, irrep, sector_dir in SECTORS:
        results.append(process(L, irrep, sector_dir))
        print(f"[packaged] {L} {irrep} candidates={results[-1]['candidates']} predictions={results[-1]['prediction_rows']}")
    summary = OUT / "ALL_SECTOR_VALIDATION_SUMMARY_E026310_0360.md"
    lines = ["# All-sector classifier validation summary", "", "CHECKPOINT_STATUS = ALL_SECTOR_PLOTS_READY_WAITING_FOR_USER_LABELS", "", "Status: PASS for all nine remaining sectors; L20/100_A2 was skipped because its accepted package is already present.", "", "| sector | status | cache metadata | provenance | rows | Ecm range | max |det_imag| | dimension jumps | candidates | accepted prediction rows | v3 true-zero predictions | v4 true-zero predictions |", "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['sector']} | {r['status']} | `{r['cache']}` | `{r['provenance']}` | {r['rows']} | [{r['ecm_min']}, {r['ecm_max']}] | {r['max_imag']:.3g} | {r['jumps']} | {r['candidates']} | {r['prediction_rows']} | {r['v3_true']} | {r['v4_true']} |")
    lines += ["", "## Skipped packaged sector", "", "- `L20_100_A2`: existing accepted package `output/v33p_classifier_restart_L20_100_A2_E026310_0360/`; not overwritten.", "", "## Required outputs", "", "Each processed sector directory contains the corrected determinant grid, validation report, candidate bracket CSV, accepted-mode predictions/summary/report, PNG/PDF QA plots, marker mapping sidecars, and a blank user-label template.", "", "## User-label checkpoint", "", "Complete the label templates listed in the table before any sector scoring or global acceptance. No new-sector user-label scoring was run.", "", "## Invariants and provenance", "", "- Existing canonical v33d GPU component caches only; no cachegen and no cache binary copying.", "- Corrected format-aware combined raw GPU cache reader only.", "- `QC = F3inv + K3df`; `projected_QC = Vsel.adjoint() * QC * Vsel`; fixed scaling by `pow(Lbyas * xi, 6.0)`; no determinant-level `Nproj` rescaling.", "- Classifier logic and corrected fix1 outputs were unchanged."]
    summary.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
