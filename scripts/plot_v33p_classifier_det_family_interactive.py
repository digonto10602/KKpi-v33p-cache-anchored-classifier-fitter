#!/usr/bin/env python3
"""Plot corrected determinant families and classifier predictions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))

def energy(row: dict[str, str]) -> float | None:
    for key in ("zero_estimate", "E_mid", "E_zero_linear", "E_candidate", "Ecm"):
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            pass
    return None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--candidate-file", type=Path)
    ap.add_argument("--prediction-file", type=Path)
    ap.add_argument("--n-scale", type=int, default=100)
    ap.add_argument("--xlim", nargs=2, type=float)
    ap.add_argument("--ylim", nargs=2, type=float)
    ap.add_argument("--marker-size", type=float, default=350.0)
    ap.add_argument("--output-stem", type=Path)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib
    if args.show and matplotlib.get_backend().lower().endswith("agg"):
        try:
            matplotlib.use("QtAgg")
        except ImportError:
            pass
    import matplotlib.pyplot as plt

    grid = sorted(rows(args.grid), key=lambda r: float(r["Ecm"]))
    x = [float(r["Ecm"]) for r in grid]
    y = [float(r["det_real"]) for r in grid]
    print(f"matplotlib={matplotlib.__version__}")
    print(f"backend={matplotlib.get_backend()}")
    if any("det_imag" in r for r in grid):
        print(f"max_abs_det_imag={max(abs(float(r.get('det_imag', 0.0))) for r in grid):.17g}")

    fig, ax = plt.subplots()
    for k in range(args.n_scale + 1):
        ax.plot(x, [v * (10.0 ** k) for v in y], color="black", linewidth=0.55, linestyle="-")
    ax.axhline(0.0, color="0.45", linewidth=0.7)

    marker_rows: list[dict[str, str]] = []
    if args.candidate_file and args.candidate_file.exists():
        for row in rows(args.candidate_file):
            e = energy(row)
            if e is not None:
                ax.scatter([e], [0.0], s=args.marker_size, facecolors="none", edgecolors="0.35", marker="o", linewidths=1.0)
    predictions = rows(args.prediction_file) if args.prediction_file and args.prediction_file.exists() else []
    methods = sorted({r.get("method", "unknown") for r in predictions})
    colors = list(plt.cm.tab10.colors)
    for i, method in enumerate(methods):
        marker = MARKERS[i % len(MARKERS)]
        color = colors[i % len(colors)]
        marker_rows.append({"method": method, "marker": marker, "color": str(color), "facecolors": "none"})
        points = [r for r in predictions if r.get("method") == method and r.get("classification") == "true_zero"]
        xx = [e for e in (energy(r) for r in points) if e is not None]
        ax.scatter(xx, [0.0] * len(xx), s=args.marker_size, facecolors="none", edgecolors=[color], marker=marker, linewidths=1.4)

    if args.xlim:
        ax.set_xlim(*args.xlim)
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.set_xlabel("Ecm")
    ax.set_ylabel(r"det$_{real}(Ecm)\,10^k$")
    fig.tight_layout()

    stem = args.output_stem or args.grid.with_name("L20_100_A2_classifier_det_family")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    with stem.with_name(stem.name + "_marker_map.csv").open("w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=["method", "marker", "color", "facecolors"])
        out.writeheader()
        out.writerows(marker_rows)
    if args.show:
        plt.show(block=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
