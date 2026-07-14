#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True, help="determinant CSV path; repeat to overlay")
    p.add_argument("--title", default="")
    p.add_argument("--Emin", type=float, default=None)
    p.add_argument("--Emax", type=float, default=None)
    p.add_argument("--ylim", default="auto", help="auto or lo,hi")
    p.add_argument("--show-imag", action="store_true")
    p.add_argument("--show-abs", action="store_true")
    p.add_argument("--show-slogdet", action="store_true")
    p.add_argument("--mark-dimension-jumps", action="store_true")
    p.add_argument("--mark-sign-changes", action="store_true")
    p.add_argument("--save-prefix", default="")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def select_backend(show: bool) -> str:
    if not show:
        matplotlib.use("Agg", force=True)
        return "Agg"
    for backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            pass
    raise RuntimeError("No GUI backend available. Install QtAgg or TkAgg, or run with display forwarding.")


@dataclass
class Series:
    label: str
    rows: list[dict[str, str]]
    jump_x: list[float]
    sign_change_x: list[float]


def load_csv(path: Path, Emin: float | None, Emax: float | None) -> Series:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if Emin is not None:
        rows = [r for r in rows if float(r["Ecm"]) >= Emin]
    if Emax is not None:
        rows = [r for r in rows if float(r["Ecm"]) <= Emax]
    rows.sort(key=lambda r: float(r["Ecm"]))
    jump_x: list[float] = []
    sign_change_x: list[float] = []
    prev = None
    prev_sign = None
    for r in rows:
        nfull = int(r["Nfull"])
        nproj = int(r["Nproj"])
        if prev is not None and (nfull != prev[0] or nproj != prev[1]):
            jump_x.append(float(r["Ecm"]))
        prev = (nfull, nproj)
        det_real = float(r["det_real"])
        sign = 0 if det_real == 0.0 else (1 if det_real > 0 else -1)
        if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
            sign_change_x.append(float(r["Ecm"]))
        if sign != 0:
            prev_sign = sign
    return Series(path.stem, rows, jump_x, sign_change_x)


def parse_ylim(text: str):
    if text == "auto":
        return None
    parts = [p for p in text.replace(":", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--ylim must be 'auto' or 'lo,hi'")
    return float(parts[0]), float(parts[1])


def main() -> int:
    args = parse_args()
    try:
        backend = select_backend(args.show)
    except Exception as exc:
        print(f"[plot] {exc}", file=sys.stderr)
        return 2

    import matplotlib.pyplot as plt

    series = [load_csv(Path(p), args.Emin, args.Emax) for p in args.csv]
    if not series or not any(s.rows for s in series):
        print("[plot] no rows after filtering", file=sys.stderr)
        return 2

    panel_kinds = ["real"]
    if args.show_imag:
        panel_kinds.append("imag")
    if args.show_abs:
        panel_kinds.append("abs")
    if args.show_slogdet:
        panel_kinds.append("slogdet")

    fig, axes = plt.subplots(len(panel_kinds), 1, sharex=True, figsize=(11, 3.2 * len(panel_kinds)))
    if len(panel_kinds) == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for idx, kind in enumerate(panel_kinds):
        ax = axes[idx]
        for sidx, s in enumerate(series):
            color = color_cycle[sidx % len(color_cycle)]
            x = [float(r["Ecm"]) for r in s.rows]
            if kind == "real":
                y = [float(r["det_real"]) for r in s.rows]
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, color=color, label=s.label)
                if args.show_imag:
                    yi = [float(r["det_imag"]) for r in s.rows]
                    ax.plot(x, yi, linestyle="--", linewidth=0.9, color=color, alpha=0.7, label=f"{s.label} imag")
            elif kind == "imag":
                y = [float(r["det_imag"]) for r in s.rows]
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, color=color, label=s.label)
            elif kind == "abs":
                y = [float(r["det_abs"]) for r in s.rows]
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, color=color, label=s.label)
            elif kind == "slogdet":
                y = [float(r["slogdet_logabs"]) for r in s.rows]
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.0, color=color, label=s.label)
        if kind == "real":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        if args.mark_dimension_jumps:
            jump_xs = sorted({x for s in series for x in s.jump_x})
            for x in jump_xs:
                ax.axvline(x, color="0.5", linestyle="--", linewidth=0.8, alpha=0.5)
        if args.mark_sign_changes and kind == "real":
            sign_xs = sorted({x for s in series for x in s.sign_change_x})
            for x in sign_xs:
                ax.axvline(x, color="crimson", linestyle=":", linewidth=0.9, alpha=0.55)
        ax.grid(True, alpha=0.25)
        if kind == "real":
            ax.set_ylabel("real(det)")
        elif kind == "imag":
            ax.set_ylabel("imag(det)")
        elif kind == "abs":
            ax.set_ylabel("|det|")
        else:
            ax.set_ylabel("slogdet logabs")
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Ecm")
    if args.title:
        fig.suptitle(args.title)

    ylim = parse_ylim(args.ylim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    if args.show_imag:
        max_imag = max(abs(float(r["det_imag"])) for s in series for r in s.rows)
        if max_imag <= 1e-15:
            axes[1].text(0.01, 0.95, "imag(det)=0 within machine precision", transform=axes[1].transAxes, fontsize=9, va="top", ha="left", color="0.25")

    fig.tight_layout()

    save_prefix = Path(args.save_prefix) if args.save_prefix else Path(series[0].label + "_interactive")
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = save_prefix.with_suffix(".png")
    pdf_path = save_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[plot] backend={backend}")
    print(f"[plot] saved={png_path}")
    print(f"[plot] saved={pdf_path}")

    if args.show:
        plt.show(block=True)
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
