#!/usr/bin/env python3
"""Score existing classifier predictions after user labels are supplied."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

def load_labels(path: Path) -> list[tuple[int, float, bool]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as fh:
            out = []
            for row in csv.DictReader(fh):
                raw = row.get("user_label", "").strip().lower()
                if raw not in {"true", "false", "1", "0", "yes", "no"}:
                    continue
                left, right = float(row["E_left"]), float(row["E_right"])
                out.append((int(row.get("bracket_id", len(out) + 1)), (left + right) / 2, raw in {"true", "1", "yes"}))
            return out
    return [(i + 1, float(line.strip()), True) for i, line in enumerate(path.read_text().splitlines()) if line.strip() and not line.lstrip().startswith("#")]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-file", type=Path, default=Path("output/v33p_classifier_restart_L20_100_A2/classifier_predictions.csv"))
    ap.add_argument("--labels", type=Path, default=Path("input/v33p_user_truezero_labels_L20_100_A2.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("output/v33p_classifier_restart_L20_100_A2"))
    ap.add_argument("--tolerance", type=float, default=8e-6)
    ap.add_argument("--false-positive-penalty", type=float, default=1.0)
    args = ap.parse_args()
    labels = load_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "classifier_score_vs_user_labels.csv"
    report_path = args.output_dir / "classifier_score_report.md"
    if not labels:
        score_path.write_text("method,TP,FP,FN,TN,runtime_seconds,score,acceptable,status\n")
        report_path.write_text("# V33P classifier scores\n\nStatus: WAITING_FOR_USER_LABELS. Fill `input/v33p_user_truezero_labels_L20_100_A2.csv` with true/false labels, then rerun this script.\n")
        print("WAITING_FOR_USER_LABELS")
        return 0
    with args.prediction_file.open(newline="") as fh:
        predictions = list(csv.DictReader(fh))
    with (args.output_dir / "classifier_summary.csv").open(newline="") as fh:
        runtimes = {r["method"]: float(r["runtime_seconds"]) for r in csv.DictReader(fh)}
    rows = []
    for method in sorted({r["method"] for r in predictions}):
        selected = [r for r in predictions if r["method"] == method and r.get("classification") == "true_zero"]
        matched = {i: False for i, _, truth in labels if truth}
        fp = 0
        for pred in selected:
            e = float(pred["zero_estimate"])
            nearest = min(labels, key=lambda x: abs(x[1] - e))
            if nearest[2] and abs(nearest[1] - e) <= args.tolerance:
                matched[nearest[0]] = True
            else:
                fp += 1
        tp = sum(matched.values())
        fn = sum(1 for i, _, truth in labels if truth and not matched.get(i, False))
        tn = max(0, sum(1 for _, _, truth in labels if not truth) - fp)
        score = tp - args.false_positive_penalty * fp - 0.25 * fn
        rows.append({"method": method, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "runtime_seconds": runtimes.get(method, math.nan), "score": score, "acceptable": "YES" if fp == 0 and fn == 0 else "NO", "status": "PASS"})
    rows.sort(key=lambda r: (-float(r["score"]), r["method"]))
    with score_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = ["# V33P classifier scores", "", f"Labels loaded: {len(labels)}", "", "| method | TP | FP | FN | TN | score | acceptable |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    report.extend(f"| {r['method']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | {r['score']} | {r['acceptable']} |" for r in rows)
    report_path.write_text("\n".join(report) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
