#!/usr/bin/env python3
"""Small no-dependency checkpoint sanity check."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
decision = ROOT / "output/v33p_classifier_restart_L20_100_A2_E026310_0360/ACCEPTED_CLASSIFIER_DECISION_E026310_0360.md"
truezeros = ROOT / "output/v33p_classifier_restart_L20_100_A2_E026310_0360/accepted_truezeros_L20_100_A2_E026310_0360.csv"

assert decision.exists()
text = decision.read_text()
assert "TP/FP/FN/TN: `2/0/0/23`" in text
assert "digonto_v3_window" in text and "digonto_v4_window" in text
with truezeros.open(newline="") as fh:
    rows = list(csv.DictReader(fh))
assert len(rows) == 2
assert [round(float(r["E_zero_linear"]), 12) for r in rows] == [0.291556099879, 0.300441497547]
print("checkpoint_csv_sanity=PASS")
