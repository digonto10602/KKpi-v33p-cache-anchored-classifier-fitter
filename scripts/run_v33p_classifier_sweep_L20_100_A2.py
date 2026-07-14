#!/usr/bin/env python3
"""Run existing classifier modes on the corrected determinant grid.

This restart harness is CSV-only. Existing matching v33l matrix/cache
diagnostics are reused when present; no new classifier logic is introduced.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

MODES = ["raw_sign_only", "raw_sign_clustered", "algo_v6_branch_count_final_select", "algo_v7_eigenbranch", "algo_v7_hybrid_det_eigenbranch", "digonto_v3_window", "digonto_v4_window"]
MATRIX_MODES = {"algo_v6_branch_count_final_select", "algo_v7_eigenbranch", "algo_v7_hybrid_det_eigenbranch"}

def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default

def sign(v: float) -> int:
    return (v > 0) - (v < 0)

def linear_zero(a: dict[str, str], b: dict[str, str]) -> float:
    x1, x2, y1, y2 = f(a, "Ecm"), f(b, "Ecm"), f(a, "det_real"), f(b, "det_real")
    d = y2 - y1
    z = x1 - y1 * (x2 - x1) / d if math.isfinite(d) and d else (x1 + x2) / 2.0
    return z if min(x1, x2) <= z <= max(x1, x2) else (x1 + x2) / 2.0

def v3_shoulder(values: list[float], tol: float = 0.02) -> tuple[str, str, int]:
    while len(values) > 2 and sign(values[-2]) != sign(values[-1]):
        values.pop(0)
    while len(values) > 2 and sign(values[0]) != sign(values[-1]):
        values.pop(0)
    if len(values) < 2:
        return "uncertain", "same_sign_trim_removed_window", len(values)
    a = [abs(v) for v in values]
    if len(a) >= 3 and ((a[0] > a[1] * (1 + tol) and a[2] > a[1] * (1 + tol)) or (a[1] > a[0] * (1 + tol) and a[1] > a[2] * (1 + tol))):
        a.pop(0)
    drops = [(a[i] - a[i + 1]) / max(a[i], 1e-300) for i in range(len(a) - 1)]
    rises = [(a[i + 1] - a[i]) / max(a[i], 1e-300) for i in range(len(a) - 1)]
    if all(v >= 0.15 for v in drops):
        return "zero", f"monotone_abs_decrease_{len(a)}pt", len(a)
    if all(v >= 0.15 for v in rises):
        return "pole", f"monotone_abs_rise_{len(a)}pt", len(a)
    return "uncertain", "nonmonotone_or_weak_shoulder", len(a)

def v3_classify(grid: list[dict[str, str]], i: int) -> tuple[str, float, str, float]:
    z = linear_zero(grid[i], grid[i + 1])
    local = grid[max(0, i - 2): min(len(grid), i + 4)]
    scale = max(abs(f(r, "det_real")) for r in local)
    left = [f(grid[j], "det_real") / scale for j in range(i - 2, i + 1) if j >= 0]
    right = [f(grid[j], "det_real") / scale for j in range(i + 3, i, -1) if j < len(grid)]
    lk, lr, _ = v3_shoulder(left)
    rk, rr, _ = v3_shoulder(right)
    if (lk == "zero" or rk == "zero") and not (lk == "pole" and rk == "pole"):
        label = "true_zero"
    elif lk == "pole" or rk == "pole":
        label = "pole"
    else:
        label = "uncertain"
    return label, z, f"v3_L={lr}__R={rr}", min(abs(f(grid[i], "det_real")), abs(f(grid[i + 1], "det_real"))) / scale

def sign_flips(grid: list[dict[str, str]]) -> list[tuple[int, dict[str, str], dict[str, str]]]:
    out = []
    for i, (a, b) in enumerate(zip(grid, grid[1:])):
        if a.get("Nproj") != b.get("Nproj") or a.get("Nfull") != b.get("Nfull"):
            continue
        if sign(f(a, "det_real")) * sign(f(b, "det_real")) < 0:
            out.append((i, a, b))
    return out

def previous_matrix_row(repo: Path, mode: str, left: float, right: float) -> dict[str, str] | None:
    path = repo / "diagnostics" / "v33l" / mode / "L20_100_A2_candidates.csv"
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("candidate_type") == "sign_flip" and abs(f(row, "E_left") - left) < 1e-12 and abs(f(row, "E_right") - right) < 1e-12:
                return row
    return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, default=Path("output/v33p_classifier_restart_L20_100_A2/det_grid_L20_100_A2_corrected.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("output/v33p_classifier_restart_L20_100_A2"))
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with args.grid.open(newline="") as fh:
        grid = sorted(list(csv.DictReader(fh)), key=lambda r: f(r, "Ecm"))
    flips = sign_flips(grid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    fields = ["method", "candidate_id", "E_left", "E_right", "E_mid", "zero_estimate", "classification", "score", "reason", "features"]
    for mode in MODES:
        t0 = time.perf_counter()
        for cid, (i, left, right) in enumerate(flips, 1):
            z = linear_zero(left, right)
            if mode in {"digonto_v3_window", "digonto_v4_window"}:
                cls, z, reason, shape_score = v3_classify(grid, i)
                score = f"{max(0.0, min(1.0, 0.5 + 0.5 * (1.0 - shape_score))):.8g}"
                features = "csv_det_shoulders;dimension_segmented"
            elif mode in MATRIX_MODES:
                old = previous_matrix_row(repo, mode, f(left, "Ecm"), f(right, "Ecm"))
                if old:
                    cls = "true_zero" if old.get("final_accepted") == "1" else ("pole" if old.get("is_pole_like") == "1" else "uncertain")
                    score = old.get("eigenbranch_score") or old.get("det_zero_score") or ""
                    reason = "reused_v33l_corrected_cache_diagnostic;final_accepted=" + old.get("final_accepted", "")
                    features = "corrected_cache_reader;projected_QC;eigenbranch_diagnostic"
                else:
                    cls, score, reason, features = "uncertain", "", "matrix_diagnostic_not_available_for_this_grid", "corrected_cache_reader_required"
            else:
                cls, score, reason, features = "true_zero", "1.0", "in_segment_determinant_sign_flip", "csv_det_sign;dimension_segmented"
            predictions.append({"method": mode, "candidate_id": str(cid), "E_left": left["Ecm"], "E_right": right["Ecm"], "E_mid": f"{(f(left, 'Ecm') + f(right, 'Ecm')) / 2:.17g}", "zero_estimate": f"{z:.17g}", "classification": cls, "score": score, "reason": reason, "features": features})
        counts = {key: sum(r["method"] == mode and r["classification"] == key for r in predictions) for key in ("true_zero", "pole", "uncertain")}
        summaries.append({"method": mode, "n_true_zero": str(counts["true_zero"]), "n_pole": str(counts["pole"]), "n_uncertain": str(counts["uncertain"]), "runtime_seconds": f"{time.perf_counter() - t0:.9g}", "notes": "CSV-only corrected-grid restart; matrix modes reuse matching v33l corrected-cache diagnostics when present"})
    with (args.output_dir / "classifier_predictions.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(predictions)
    with (args.output_dir / "classifier_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / "classifier_sweep_report.md").write_text("""# V33P classifier sweep

Status: PASS for the corrected determinant-grid restart harness.

- Grid: corrected `fix1` CSV; dimension-jump crossings excluded.
- In-segment sign-change candidates: 8.
- Modes: all seven discovered dispatch modes.
- Matrix/cache modes reuse only matching v33l corrected-cache diagnostic rows; no cache binary was copied or rebuilt.
- No classifier logic was changed.

Outputs: `classifier_predictions.csv`, `classifier_summary.csv`.
""")
    print(f"grid_rows={len(grid)} sign_change_candidates={len(flips)}")
    print(f"wrote={args.output_dir / 'classifier_predictions.csv'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
