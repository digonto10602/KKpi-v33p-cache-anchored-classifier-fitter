#!/usr/bin/env python3
"""Prepare L20/200_A2 classifier artifacts from an existing corrected raw scan."""
import csv
import shutil
import subprocess
from pathlib import Path

from prepare_v33p_all_sector_validation import candidate_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "output/v33p_all_sector_classifier_validation/L20_200_A2_E026310_0360/raw_scan/L20_200_A2_oldscale_det_coarse20000_E026_036.csv"
    out = ROOT / "output/v33p_gpu_classifier_all_irrep/sectors/L20_200_A2"
    labels = ROOT / "input/v33p_gpu_user_truezero_labels_L20_200_A2.csv"
    out.mkdir(parents=True, exist_ok=True)
    grid = out / "det_grid_raw_projected_basis_E026310_0360.csv"
    shutil.copyfile(raw, grid)
    with grid.open(newline="") as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda r: float(r["Ecm"]))
    flips, _ = candidate_rows(rows)
    candidate = out / "candidate_sign_change_brackets.csv"
    fields = ["bracket_id", "segment_id", "row_left", "row_right", "E_left", "E_right", "gap", "det_left", "det_right", "E_zero_linear", "sign_left", "sign_right", "Nfull", "Nproj"]
    with candidate.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flips)

    tmp = out / ".sweep_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["python3", "scripts/run_v33p_classifier_sweep_L20_100_A2.py", "--grid", str(grid), "--output-dir", str(tmp)], cwd=ROOT, check=True)
    accepted = {"digonto_v3_window", "digonto_v4_window"}
    with (tmp / "classifier_predictions.csv").open(newline="") as f:
        predictions = [r for r in csv.DictReader(f) if r["method"] in accepted]
    with (out / "accepted_classifier_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    with (tmp / "classifier_summary.csv").open(newline="") as f:
        summaries = [r for r in csv.DictReader(f) if r["method"] in accepted]
    with (out / "classifier_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    with (out / "classifier_summary.md").open("w") as f:
        f.write(f"# L20/200_A2 classifier preparation\n\n- Grid rows: {len(rows)}\n- Candidate count: {len(flips)}\n- Accepted-mode prediction rows: {len(predictions)}\n- Modes: digonto_v3_window, digonto_v4_window\n- Existing classifier logic unchanged.\n")
    plot_stem = out / "classifier_QA"
    subprocess.run(["python3", "scripts/plot_v33p_classifier_det_family_interactive.py", "--grid", str(grid), "--candidate-file", str(candidate), "--prediction-file", str(out / "accepted_classifier_predictions.csv"), "--n-scale", "100", "--output-stem", str(plot_stem)], cwd=ROOT, check=True)
    if not labels.exists():
        with labels.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bracket_id", "E_left", "E_right", "user_label", "user_zero_estimate", "comment"])
            w.writeheader()
            for r in flips:
                w.writerow({"bracket_id": r["bracket_id"], "E_left": r["E_left"], "E_right": r["E_right"], "user_label": "", "user_zero_estimate": "", "comment": ""})
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"rows={len(rows)} candidates={len(flips)} predictions={len(predictions)} labels={labels}")


if __name__ == "__main__":
    main()
