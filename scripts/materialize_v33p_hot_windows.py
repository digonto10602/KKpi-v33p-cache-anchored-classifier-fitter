#!/usr/bin/env python3
"""Materialize persisted hot windows from already validated labels/brackets."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def read_targets(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    return read_rows(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-truezeros", required=True)
    ap.add_argument("--candidate-brackets", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--Lbyas", type=int, required=True)
    ap.add_argument("--irrep", required=True)
    ap.add_argument("--cutoff", type=float, default=0.335)
    ap.add_argument("--window-half-rows", type=int, default=50)
    ap.add_argument("--max-window-half-rows", type=int, default=250)
    a = ap.parse_args()

    accepted = [r for r in read_rows(a.accepted_truezeros)
                if r.get("user_label", "").strip().lower() == "true"]
    candidates = {int(r["bracket_id"]): r for r in read_rows(a.candidate_brackets)}
    targets = [r for r in read_targets(a.targets)
               if int(r["Lbyas"]) == a.Lbyas
               and r["irrep_canonical"] == a.irrep
               and str(r.get("selected_under_cutoff", True)).lower() == "true"
               and float(r["Ecm"]) <= a.cutoff]
    # Handle both the newer inside_Ecm_cutoff schema and the older E_zero_linear schema.
    accepted = [r for r in read_rows(a.accepted_truezeros)
                if r.get("user_label", "").strip().lower() == "true"
                and float(r.get("zero_estimate", r.get("E_zero_linear", "inf"))) <= a.cutoff]
    accepted.sort(key=lambda r: float(r.get("zero_estimate", r.get("E_zero_linear", "inf"))))
    if len(accepted) != len(targets):
        raise SystemExit(f"accepted-zero/target count mismatch: {len(accepted)} vs {len(targets)}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Lbyas", "irrep", "lattice_level_index", "lattice_Ecm", "bracket_id",
              "E_left_bracket", "E_right_bracket", "zero_estimate_initial", "center_row",
              "row_left", "row_right", "max_row_left", "max_row_right",
              "previous_model_zero", "has_previous_model_zero", "inside_Ecm_cutoff"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for level, (z, t) in enumerate(zip(accepted, targets)):
            bracket = int(z["bracket_id"])
            c = candidates[bracket]
            center = int(round((int(c["row_left"]) + int(c["row_right"])) / 2))
            lo = max(0, min(center - a.window_half_rows, int(c["row_left"])))
            hi = max(center + a.window_half_rows, int(c["row_right"]))
            w.writerow({
                "Lbyas": a.Lbyas, "irrep": a.irrep, "lattice_level_index": level,
                "lattice_Ecm": t["Ecm"], "bracket_id": bracket,
                "E_left_bracket": c["E_left"], "E_right_bracket": c["E_right"],
                "zero_estimate_initial": z.get("zero_estimate", z.get("E_zero_linear")),
                "center_row": center, "row_left": lo, "row_right": hi,
                "max_row_left": max(0, center - a.max_window_half_rows),
                "max_row_right": center + a.max_window_half_rows,
                "previous_model_zero": z.get("zero_estimate", z.get("E_zero_linear")),
                "has_previous_model_zero": "true", "inside_Ecm_cutoff": "true"})


if __name__ == "__main__":
    main()
