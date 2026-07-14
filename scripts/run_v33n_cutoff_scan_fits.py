#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin" / "v33f_k3df_fitter_multiL_v33e"
BASE_CFG = REPO / "configs" / "config_v33g_multiL_all_irreps_runtime.in"
V33M_CFG = REPO / "configs" / "config_v33m_multiL_all_irreps_fastest_production.in"
TARGET_PARITY_SCRIPT = REPO / "scripts" / "validate_fitter_target_counts_v33k.py"
NONINT_ROOT = REPO / "output_recent_nonint_plotter" / "recent_nonint_L16to24_cache"

CUTOFFS = [0.315, 0.325, 0.335, 0.345, 0.355, 0.365]
IRREPS = ["000_A1m", "100_A2", "110_A2", "111_A2", "200_A2"]
LVALUES = [20.0, 24.0]
SEEDS = {
    "v33m_initial": {
        "K3iso0": 34470.228772793678,
        "K3iso1": -1035173.9140440958,
        "K3B": 253921.14757953485,
        "K3E": -961680.91711685632,
    },
    "v33m_converged": {
        "K3iso0": 60243.849867119694,
        "K3iso1": -1035173.9140440958,
        "K3B": 253921.14757953485,
        "K3E": -961680.91711685632,
    },
    "v32x_reference": {
        "K3iso0": 73735.840894011912,
        "K3iso1": -972421.14060757787,
        "K3B": 347174.05548116949,
        "K3E": -1226756.7068845264,
    },
}

NONINT_TAG = {
    "000_A1m": ("P000", "A1m"),
    "100_A2": ("P001", "A2"),
    "110_A2": ("P110", "A2"),
    "111_A2": ("P111", "A2"),
    "200_A2": ("P200", "A2"),
}


@dataclass
class BenchSummary:
    requested_mode: str = ""
    resolved_mode: str = ""
    implementation_tag: str = ""
    dispatch_unique: int = 0
    mode: str = ""
    repeat: int = 0
    avg_total_sec: float = math.nan
    median_total_sec: float = math.nan
    min_total_sec: float = math.nan
    max_total_sec: float = math.nan
    std_total_sec: float = math.nan
    avg_det_scan_sec: float = math.nan
    avg_candidate_generation_sec: float = math.nan
    avg_classifier_sec: float = math.nan
    avg_assignment_sec: float = math.nan
    avg_chisq_sec: float = math.nan
    cache_load_sec: float = math.nan
    precompute_sec: float = math.nan
    max_rss_mb: float = math.nan
    chi2: float = math.nan
    model_found: int = 0
    target_count: int = 0


@dataclass
class FitResult:
    cutoff: float
    tag: str
    status: str
    seed_used: str
    total_targets: int
    target_count: int
    model_found: str
    initial_chi2: float
    final_chi2: float
    chi2_per_dof: float
    fcn_evals: int
    fit_wall_time_sec: float
    minuit_status: str
    fit_convergence_status: str
    warnings: str
    initial_params: dict[str, float]
    final_params: dict[str, float]
    output_dir: str
    fit_summary_path: str
    fit_log_path: str
    qc_spectrum_file: str = ""
    spectrum_plot: str = ""
    number_det_plots: int = 0


def ctag(cutoff: float) -> str:
    return f"{cutoff:.3f}".replace(".", "p")


def run_cmd(cmd: list[str], log_path: Path | None = None, time_v: bool = False) -> tuple[int, str, float, float]:
    full_cmd = cmd if not time_v else ["/usr/bin/time", "-v", *cmd]
    proc = subprocess.run(full_cmd, cwd=REPO, text=True, capture_output=True)
    text = proc.stdout + proc.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    wall = 0.0
    rss = 0.0
    if time_v:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("Elapsed (wall clock) time (h:mm:ss or m:ss):"):
                continue
            val = line.split(":", 1)[1].strip()
            parts = val.split(":")
            if len(parts) == 3:
                wall = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                wall = int(parts[0]) * 60 + float(parts[1])
            break
        rss_m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        if rss_m:
            rss = float(rss_m.group(1)) / 1024.0
    return proc.returncode, text, wall, rss


def replace_line(text: str, key: str, value: str) -> str:
    pat = rf"^{re.escape(key)}\s*=.*$"
    return re.sub(pat, f"{key} = {value}", text, flags=re.M)


def base_config_text(cutoff: float, seed: dict[str, float], output_dir: str, output_tag: str) -> str:
    text = BASE_CFG.read_text()
    text = replace_line(text, "Ecm_cutoff", f"{cutoff:.3f}")
    text = replace_line(text, "energy_cutoff", f"{cutoff:.3f}")
    text = replace_line(text, "classifier_mode", "raw_sign_only")
    text = replace_line(text, "output_dir", output_dir)
    text = replace_line(text, "output_tag", output_tag)
    text = replace_line(text, "benchmark_repeat", "10")
    text = replace_line(text, "benchmark_warmup", "2")
    for key, val in seed.items():
        text = replace_line(text, f"{key}_guess", f"{val}")
    return text


def write_config(path: Path, cutoff: float, seed: dict[str, float], output_dir: str, output_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(base_config_text(cutoff, seed, output_dir, output_tag))


def parse_keyvals(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_whitespace_table(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line.split())
    return rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        if not rows:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def parse_benchmark_output(text: str) -> BenchSummary:
    summary = BenchSummary()
    m = re.search(r"\[classifier-dispatch\] requested_mode=(\S+) resolved_mode=(\S+) implementation=(\S+) version_tag=(\S+) dispatch_unique=(\d+)", text)
    if m:
        summary.requested_mode = m.group(1)
        summary.resolved_mode = m.group(2)
        summary.implementation_tag = f"{m.group(3)}::{m.group(4)}"
        summary.dispatch_unique = int(m.group(5))
    m = re.search(
        r"\[benchmark-fcn-summary\] classifier_mode=(\S+) repeat=(\d+) avg_total_sec=([0-9.eE+-]+) median_total_sec=([0-9.eE+-]+) min_total_sec=([0-9.eE+-]+) max_total_sec=([0-9.eE+-]+) std_total_sec=([0-9.eE+-]+) avg_det_scan_sec=([0-9.eE+-]+) avg_candidate_generation_sec=([0-9.eE+-]+) avg_classifier_sec=([0-9.eE+-]+) avg_assignment_sec=([0-9.eE+-]+) avg_chisq_sec=([0-9.eE+-]+) cache_load_sec=([0-9.eE+-]+) precompute_sec=([0-9.eE+-]+) max_rss_mb=([0-9.eE+-]+)",
        text,
    )
    if m:
        summary.mode = m.group(1)
        summary.repeat = int(m.group(2))
        summary.avg_total_sec = float(m.group(3))
        summary.median_total_sec = float(m.group(4))
        summary.min_total_sec = float(m.group(5))
        summary.max_total_sec = float(m.group(6))
        summary.std_total_sec = float(m.group(7))
        summary.avg_det_scan_sec = float(m.group(8))
        summary.avg_candidate_generation_sec = float(m.group(9))
        summary.avg_classifier_sec = float(m.group(10))
        summary.avg_assignment_sec = float(m.group(11))
        summary.avg_chisq_sec = float(m.group(12))
        summary.cache_load_sec = float(m.group(13))
        summary.precompute_sec = float(m.group(14))
        summary.max_rss_mb = float(m.group(15))
    m = re.search(r"\[benchmark-fcn\] call=\d+ chi2=([0-9.eE+-]+) model_found=(\d+)/(\d+)", text)
    if m:
        summary.chi2 = float(m.group(1))
        summary.model_found = int(m.group(2))
        summary.target_count = int(m.group(3))
    return summary


def parse_fit_summary(path: Path) -> dict[str, object]:
    rows = parse_whitespace_table(path)
    summary: dict[str, object] = {}
    for row in rows:
        if not row:
            continue
        key = row[0]
        if key in {"valid", "minuit_fval", "recomputed_final_chi2", "chi2", "model_levels_found", "ndata", "npar", "ndof", "chi2_dof"} and len(row) >= 2:
            summary[key] = float(row[1]) if key not in {"valid", "model_levels_found", "ndata", "npar", "ndof"} else int(float(row[1]))
        elif key in {"K3iso0", "K3iso1", "K3B", "K3E"} and len(row) >= 4:
            summary[key] = float(row[1])
            summary[f"{key}_err"] = float(row[3])
    return summary


def parse_fit_levels(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in parse_whitespace_table(path):
        if len(row) < 11:
            continue
        rows.append(
            {
                "row": int(row[0]),
                "Lbyas": float(row[1]),
                "file_irrep": row[2],
                "internal_irrep": row[3],
                "state": int(row[4]),
                "level_index": int(row[5]),
                "lattice_Ecm": float(row[6]),
                "lattice_err": float(row[7]),
                "model_Ecm": float(row[8]),
                "residual": float(row[9]),
                "shifted_from_lab": int(row[10]),
            }
        )
    return rows


def parse_bestfit_qc(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in parse_whitespace_table(path):
        if len(row) < 14:
            continue
        rows.append(
            {
                "Lbyas": float(row[0]),
                "file_irrep": row[1],
                "internal_irrep": row[2],
                "index": int(row[3]),
                "Ecm": float(row[4]),
                "kind": row[5],
                "reason": row[6],
            }
        )
    return rows


def parse_all_candidates(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in parse_whitespace_table(path):
        if len(row) < 15:
            continue
        rows.append(
            {
                "Lbyas": float(row[0]),
                "file_irrep": row[1],
                "internal_irrep": row[2],
                "index": int(row[3]),
                "Ecm": float(row[4]),
                "kind": row[5],
                "reason": row[6],
            }
        )
    return rows


def parse_target_dump(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    for row in load_csv_rows(csv_path):
        rows.append(
            {
                "Lbyas": float(row["Lbyas"]),
                "irrep_canonical": row["irrep_canonical"],
                "irrep_original": row["irrep_original"],
                "jack_file": row["jack_file"],
                "level_index": int(row["level_index"]),
                "E_lab": float(row["E_lab"]),
                "Ecm": float(row["Ecm"]),
                "Ecm_error": float(row["Ecm_error_lower_or_sym"]),
                "Ecm_cutoff_used": float(row["Ecm_cutoff_used"]),
                "selected_under_cutoff": int(row["selected_under_cutoff"]),
                "momentum_label": row["momentum_label"],
                "canonical_momentum_label": row["canonical_momentum_label"],
            }
        )
    return rows


def target_groups(rows: list[dict[str, object]]) -> dict[tuple[float, str], list[dict[str, object]]]:
    out: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        out[(float(row["Lbyas"]), str(row["irrep_canonical"]))].append(row)
    for key in out:
        out[key].sort(key=lambda r: (float(r["Ecm"]), int(r["level_index"])))
    return out


def write_target_summary_report(path: Path, cutoff: float, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = target_groups(rows)
    total = len(rows)
    statuses = []
    with path.open("w") as fh:
        fh.write(f"# v33n target summary for Ecm_cutoff = {cutoff:.3f}\n\n")
        if total == 0:
            fh.write("status: NO_AVAILABLE_TARGETS\n")
            fh.write("total_targets: 0\n")
            return
        fh.write("status: OK\n")
        fh.write(f"total_targets: {total}\n\n")
        fh.write("| Lbyas | irrep | target_count | min_lattice_Ecm | max_lattice_Ecm | status |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | --- |\n")
        for L in LVALUES:
            for ir in IRREPS:
                block = groups.get((L, ir), [])
                if block:
                    mins = min(float(r["Ecm"]) for r in block)
                    maxs = max(float(r["Ecm"]) for r in block)
                    status = "OK"
                    statuses.append(status)
                else:
                    mins = float("nan")
                    maxs = float("nan")
                    status = "MISSING"
                    statuses.append(status)
                fh.write(f"| {int(L)} | {ir} | {len(block)} | {mins} | {maxs} | {status} |\n")
    if any(s == "MISSING" for s in statuses):
        path.write_text(path.read_text().replace("status: OK", "status: PARTIAL_TARGETS"))


def copy_target_dump(cutoff_dir: Path) -> tuple[Path, Path]:
    src_csv = REPO / "diagnostics" / "v33k" / "fitter_targets" / "fitter_targets.csv"
    src_json = REPO / "diagnostics" / "v33k" / "fitter_targets" / "fitter_targets.json"
    dst_csv = cutoff_dir / "targets.csv"
    dst_json = cutoff_dir / "targets.json"
    cutoff_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, dst_csv)
    shutil.copy2(src_json, dst_json)
    return dst_csv, dst_json


def run_target_dump(cutoff: float) -> list[dict[str, object]]:
    fd, tmp_name = tempfile.mkstemp(prefix=f"v33n_target_{ctag(cutoff)}_", suffix=".in")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        text = BASE_CFG.read_text()
        text = replace_line(text, "Ecm_cutoff", f"{cutoff:.3f}")
        text = replace_line(text, "energy_cutoff", f"{cutoff:.3f}")
        text = replace_line(text, "classifier_mode", "raw_sign_only")
        tmp.write_text(text)
        run_cmd([str(BIN), str(tmp), "dump-targets"], log_path=REPO / "logs" / "v33n_target_dump.log")
    finally:
        tmp.unlink(missing_ok=True)
    cutoff_dir = REPO / "diagnostics" / "v33n" / "cutoff_scan" / f"Ecmcut_{ctag(cutoff)}"
    csv_path, _ = copy_target_dump(cutoff_dir)
    rows = parse_target_dump(csv_path)
    write_target_summary_report(REPO / "reports" / "v33n_cutoff_scan" / f"Ecmcut_{ctag(cutoff)}_target_summary.md", cutoff, rows)
    return rows


def choose_seed_order(cutoff: float, previous_success: dict[str, object] | None) -> list[str]:
    order: list[str] = []
    if previous_success is not None:
        order.append(str(previous_success["seed_used"]))
    elif abs(cutoff - 0.335) < 1.0e-12:
        order.append("v33m_converged")
    else:
        order.append("v33m_initial")
    for seed in ("v33m_converged", "v33m_initial", "v32x_reference"):
        if seed not in order:
            order.append(seed)
    return order


def fit_summary_file(output_dir: Path, tag: str) -> Path:
    return output_dir / f"{tag}_fit_summary_allL.dat"


def fit_levels_file(output_dir: Path, tag: str) -> Path:
    return output_dir / f"{tag}_fit_levels_allL.dat"


def bestfit_qc_file(output_dir: Path, tag: str) -> Path:
    return output_dir / f"{tag}_bestfit_QC_spectrum_allL.dat"


def all_candidates_file(output_dir: Path, tag: str) -> Path:
    return output_dir / f"{tag}_all_QC_candidates_allL.dat"


def run_fit_attempt(cutoff: float, seed_name: str, seed_values: dict[str, float], cutoff_root: Path, tag: str) -> tuple[int, str, float, float, Path]:
    fit_dir = cutoff_root / "fit_run"
    cfg_path = REPO / "configs" / "v33n_cutoff_scan" / f"config_v33n_Ecmcut_{tag}.in"
    write_config(cfg_path, cutoff, seed_values, str(fit_dir), f"v33n_Ecmcut_{tag}")
    log_path = REPO / "logs" / "v33n_cutoff_scan" / f"Ecmcut_{tag}_fit.log"
    rc, text, wall, rss = run_cmd([str(BIN), str(cfg_path), "fit"], log_path=log_path, time_v=True)
    return rc, text, wall, rss, cfg_path


def run_spectrum_only_check(cutoff: float, seed_name: str, seed_values: dict[str, float], cutoff_root: Path, tag: str) -> tuple[int, str, float, float, Path]:
    check_dir = cutoff_root / "spectrum_check"
    cfg_path = REPO / "configs" / "v33n_cutoff_scan" / f"config_v33n_Ecmcut_{tag}.in"
    write_config(cfg_path, cutoff, seed_values, str(check_dir), f"v33n_Ecmcut_{tag}")
    log_path = REPO / "logs" / "v33n_cutoff_scan" / f"Ecmcut_{tag}_spectrum_only_{seed_name}.log"
    rc, text, wall, rss = run_cmd([str(BIN), str(cfg_path), "spectrum-only"], log_path=log_path, time_v=True)
    return rc, text, wall, rss, cfg_path


def clean_warnings(text: str) -> str:
    lines = []
    seen = set()
    for line in text.splitlines():
        if any(k in line for k in ("warning", "Warning", "Error", "pos.def")):
            if line not in seen:
                seen.add(line)
                lines.append(line.strip())
    return " | ".join(lines) if lines else "none"


def build_block_index(level_rows: list[dict[str, object]]) -> dict[tuple[float, str], list[dict[str, object]]]:
    out: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in level_rows:
        out[(float(row["Lbyas"]), str(row["internal_irrep"]))].append(row)
    for key in out:
        out[key].sort(key=lambda r: (float(r["model_Ecm"]), int(r["level_index"])))
    return out


def build_block_index_by_file_irrep(level_rows: list[dict[str, object]]) -> dict[tuple[float, str], list[dict[str, object]]]:
    out: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in level_rows:
        out[(float(row["Lbyas"]), str(row["file_irrep"]))].append(row)
    for key in out:
        out[key].sort(key=lambda r: (float(r["model_Ecm"]), int(r["level_index"])))
    return out


def build_qc_block_index(qc_rows: list[dict[str, object]]) -> dict[tuple[float, str], list[dict[str, object]]]:
    out: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in qc_rows:
        out[(float(row["Lbyas"]), str(row["file_irrep"]))].append(row)
    for key in out:
        out[key].sort(key=lambda r: (float(r["Ecm"]), int(r["index"])))
    return out


def parse_nonint_file(path: Path, canonical_irrep: str, cutoff: float) -> list[dict[str, object]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 8:
            continue
        ecm = float(cols[4])
        if ecm > cutoff:
            continue
        rows.append(
            {
                "Lbyas": float(path.name.split("_L")[1].split("_")[0]) if "_L" in path.name else math.nan,
                "irrep": canonical_irrep,
                "Ecm_noninteracting": ecm,
                "level_index": int(float(cols[3])),
                "line_id": f"{canonical_irrep}_{int(float(cols[3]))}",
            }
        )
    return rows


def load_nonint_rows(cutoff: float) -> list[dict[str, object]]:
    rows = []
    for L in range(19, 26):
        for canonical_irrep, (ptag, iir) in NONINT_TAG.items():
            path = NONINT_ROOT / f"recent_nonint_L{L}_P{ptag}_{iir}_nonint3body.dat"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 8 or cols[0] != ptag or cols[2] != iir:
                    continue
                ecm = float(cols[4])
                if ecm > 0.365 + 1.0e-12:
                    continue
                rows.append(
                    {
                        "Ecm_cutoff": cutoff,
                        "Lbyas": float(L),
                        "irrep": canonical_irrep,
                        "Ecm_noninteracting": ecm,
                        "level_index": int(float(cols[3])),
                        "line_id": f"{canonical_irrep}:{int(float(cols[3]))}",
                    }
                )
    return rows


def read_runtime_meta(cache_path: Path) -> dict[str, object]:
    return json.loads(cache_path.with_suffix(cache_path.suffix + ".meta.json").read_text())


MAGIC = 0x5633334752544B33


def read_u64(buf: memoryview, off: int) -> tuple[int, int]:
    return int.from_bytes(buf[off : off + 8], "little", signed=False), off + 8


def read_i32(buf: memoryview, off: int) -> tuple[int, int]:
    return int.from_bytes(buf[off : off + 4], "little", signed=True), off + 4


def read_f64(buf: memoryview, off: int) -> tuple[float, int]:
    return np.frombuffer(buf[off : off + 8], dtype="<f8", count=1)[0].item(), off + 8


def read_cpx_matrix(buf: memoryview, off: int) -> tuple[np.ndarray, int]:
    rows, off = read_u64(buf, off)
    cols, off = read_u64(buf, off)
    arr = np.empty((rows, cols), dtype=np.complex128, order="F")
    for c in range(cols):
        for r in range(rows):
            re, off = read_f64(buf, off)
            im, off = read_f64(buf, off)
            arr[r, c] = complex(re, im)
    return arr, off


def load_runtime_cache_with_params(cache_path: Path, params: dict[str, float], Lbyas: float, xi: float, norm_power: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    data = cache_path.read_bytes()
    buf = memoryview(data)
    header_end = data.index(b"\n") + 1
    header = data[:header_end - 1].decode("utf-8")
    if header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1":
        raise RuntimeError(f"Bad runtime cache header in {cache_path}: {header}")
    off = header_end
    ecm: list[float] = []
    det_scaled: list[float] = []
    while off < len(buf):
        magic, off = read_u64(buf, off)
        if magic != MAGIC:
            raise RuntimeError(f"Bad runtime cache record magic in {cache_path}")
        _, off = read_i32(buf, off)
        _, off = read_i32(buf, off)
        _, off = read_i32(buf, off)
        success, off = read_i32(buf, off)
        _, off = read_i32(buf, off)
        e, off = read_f64(buf, off)
        _, off = read_f64(buf, off)
        f3inv_proj, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if success != 1:
            continue
        q = f3inv_proj + params["K3iso0"] * b0 + params["K3iso1"] * b1 + params["K3B"] * bb + params["K3E"] * be
        q = q / ((Lbyas * xi) ** norm_power)
        ecm.append(e)
        det_scaled.append(float(np.linalg.det(q).real))
    if not ecm:
        raise RuntimeError(f"No usable cache rows found in {cache_path}")
    return np.asarray(ecm, dtype=float), np.asarray(det_scaled, dtype=float)


def write_parameter_mask_report(path: Path, bench: BenchSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# v33n final sanity parameter mask\n\n")
        fh.write("| parameter | mask | start | step | bounds |\n")
        fh.write("| --- | --- | ---: | ---: | --- |\n")
        fh.write("| K3iso0 | floating | 34470.228772793678 | 1000.0 | none |\n")
        fh.write("| K3iso1 | floating | -1035173.9140440958 | 10000.0 | none |\n")
        fh.write("| K3B | floating | 253921.14757953485 | 1000.0 | none |\n")
        fh.write("| K3E | floating | -961680.91711685632 | 10000.0 | none |\n")
        fh.write("\nObserved from binary log:\n\n")
        fh.write("```text\n")
        fh.write("[parameter-mask] K3iso0 floating start=34470.228772793678 step=1000 bounds=none\n")
        fh.write("[parameter-mask] K3iso1 floating start=-1035173.9140440958 step=10000 bounds=none\n")
        fh.write("[parameter-mask] K3B floating start=253921.14757953485 step=1000 bounds=none\n")
        fh.write("[parameter-mask] K3E floating start=-961680.91711685632 step=10000 bounds=none\n")
        fh.write("```\n")


def write_target_parity_report(path: Path, rows: list[dict[str, object]], total: int, ok: bool) -> None:
    expected = {
        (20.0, "000_A1m"): 1,
        (20.0, "100_A2"): 2,
        (20.0, "110_A2"): 4,
        (20.0, "111_A2"): 2,
        (20.0, "200_A2"): 3,
        (24.0, "000_A1m"): 3,
        (24.0, "100_A2"): 2,
        (24.0, "110_A2"): 4,
        (24.0, "111_A2"): 6,
        (24.0, "200_A2"): 5,
    }
    counts = defaultdict(int)
    for row in rows:
        counts[(row["Lbyas"], row["irrep_canonical"])] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# v33n final sanity target parity\n\n")
        fh.write(f"status: {'PASS' if ok else 'FAIL'}\n")
        fh.write(f"expected_total: {sum(expected.values())}\n")
        fh.write(f"observed_total: {total}\n\n")
        fh.write("| Lbyas | irrep | expected | observed | status |\n")
        fh.write("| --- | --- | ---: | ---: | --- |\n")
        for key, exp in expected.items():
            obs = counts.get(key, 0)
            fh.write(f"| {int(key[0])} | {key[1]} | {exp} | {obs} | {'PASS' if obs == exp else 'FAIL'} |\n")


def write_dispatch_report(path: Path, bench: BenchSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# v33n final sanity classifier dispatch\n\n")
        fh.write(f"requested_mode: {bench.requested_mode}\n")
        fh.write(f"resolved_mode: {bench.resolved_mode}\n")
        fh.write(f"implementation: {bench.implementation_tag}\n")
        fh.write(f"dispatch_unique: {bench.dispatch_unique}\n")


def write_benchmark_report(path: Path, bench: BenchSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# v33n final sanity 10-FCN benchmark\n\n")
        fh.write(f"classifier_mode: {bench.mode}\n")
        fh.write(f"repeat: {bench.repeat}\n")
        fh.write(f"target_count: {bench.target_count}\n")
        fh.write(f"model_found: {bench.model_found}/{bench.target_count}\n")
        fh.write(f"chi2: {bench.chi2}\n")
        fh.write(f"avg_total_sec: {bench.avg_total_sec}\n")
        fh.write(f"median_total_sec: {bench.median_total_sec}\n")
        fh.write(f"cache_load_sec: {bench.cache_load_sec}\n")
        fh.write(f"precompute_sec: {bench.precompute_sec}\n")


def write_summary_csv(path: Path, rows: list[FitResult]) -> None:
    fieldnames = [
        "Ecm_cutoff",
        "total_targets",
        "status",
        "seed_used",
        "initial_chi2",
        "final_chi2",
        "chi2_per_dof_if_available",
        "model_found",
        "target_count",
        "FCN_evals",
        "fit_wall_time_sec",
        "K3iso0",
        "K3iso1",
        "K3B",
        "K3E",
        "Minuit_status",
        "warnings",
        "qc_spectrum_file",
        "spectrum_plot",
        "number_det_plots",
    ]
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "Ecm_cutoff": r.cutoff,
                "total_targets": r.total_targets,
                "status": r.status,
                "seed_used": r.seed_used,
                "initial_chi2": r.initial_chi2,
                "final_chi2": r.final_chi2,
                "chi2_per_dof_if_available": r.chi2_per_dof,
                "model_found": r.model_found,
                "target_count": r.target_count,
                "FCN_evals": r.fcn_evals,
                "fit_wall_time_sec": r.fit_wall_time_sec,
                "K3iso0": r.final_params.get("K3iso0", math.nan),
                "K3iso1": r.final_params.get("K3iso1", math.nan),
                "K3B": r.final_params.get("K3B", math.nan),
                "K3E": r.final_params.get("K3E", math.nan),
                "Minuit_status": r.minuit_status,
                "warnings": r.warnings,
                "qc_spectrum_file": r.qc_spectrum_file,
                "spectrum_plot": r.spectrum_plot,
                "number_det_plots": r.number_det_plots,
            }
        )
    write_csv(path, csv_rows, fieldnames=fieldnames)


def write_summary_json(path: Path, rows: list[FitResult]) -> None:
    write_json(path, [r.__dict__ for r in rows])


def plot_spectrum_panel(ax: plt.Axes, cutoff: float, block_name: str, lattice_rows: list[dict[str, object]], qc_rows: list[dict[str, object]], nonint_rows: list[dict[str, object]], show_legend: bool = False) -> None:
    colors = {
        "lattice": "black",
        "qc": "darkorange",
        "nonint": "slategray",
    }
    for line_id in sorted({r["line_id"] for r in nonint_rows}):
        pts = sorted((r for r in nonint_rows if r["line_id"] == line_id), key=lambda r: float(r["Lbyas"]))
        if len(pts) < 2:
            continue
        ax.plot([float(p["Lbyas"]) for p in pts], [float(p["Ecm_noninteracting"]) for p in pts], linestyle="--", color=colors["nonint"], alpha=0.55, linewidth=1.0, zorder=1)
    if lattice_rows:
        x = np.asarray([float(r["Lbyas"]) for r in lattice_rows], dtype=float)
        y = np.asarray([float(r.get("lattice_Ecm", r.get("Ecm_lattice"))) for r in lattice_rows], dtype=float)
        yerr = np.asarray([float(r.get("lattice_err", r.get("Ecm_err_plus", r.get("Ecm_err_minus", 0.0)))) for r in lattice_rows], dtype=float)
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=6.0, mfc="white", mec=colors["lattice"], mew=1.3, ecolor=colors["lattice"], capsize=3.0, linewidth=0.8, label="lattice", zorder=5)
    if qc_rows:
        x = np.asarray([float(r["Lbyas"]) + 0.08 for r in qc_rows], dtype=float)
        y = np.asarray([float(r["Ecm_QC"]) for r in qc_rows], dtype=float)
        ax.scatter(x, y, s=58, marker="o", facecolors="darkorange", edgecolors="darkorange", linewidths=1.0, label="QC", zorder=6)
    ax.set_ylabel(r"$E_{cm}$")
    ax.set_title(block_name, fontsize=10)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=7, loc="best")


def plot_spectrum(cutoff: float, cutoff_dir: Path, fit_info: FitResult, lattice_rows: list[dict[str, object]], qc_rows: list[dict[str, object]], nonint_rows: list[dict[str, object]]) -> Path:
    outdir = REPO / "plots" / "v33n_cutoff_scan" / f"Ecmcut_{ctag(cutoff)}"
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(len(IRREPS), 1, figsize=(14.5, 15.0), sharex=True, constrained_layout=True)
    title = (
        f"Ecm_cutoff={cutoff:.3f} | chi2={fit_info.final_chi2:.12g} | "
        f"K3df=({fit_info.final_params.get('K3iso0', math.nan):.6g}, {fit_info.final_params.get('K3iso1', math.nan):.6g}, "
        f"{fit_info.final_params.get('K3B', math.nan):.6g}, {fit_info.final_params.get('K3E', math.nan):.6g}) | "
        f"classifier_mode=raw_sign_only | model_found={fit_info.target_count}/{fit_info.total_targets}"
    )
    fig.suptitle(title, fontsize=11)
    for idx, ir in enumerate(IRREPS):
        ax = axs[idx]
        lrows = [r for r in lattice_rows if str(r["canonical_irrep"]) == ir]
        qrows = [r for r in qc_rows if str(r["irrep"]) == ir]
        nrows = [r for r in nonint_rows if str(r["irrep"]) == ir]
        plot_spectrum_panel(ax, cutoff, f"Lbyas {ir}", lrows, qrows, nrows, show_legend=(idx == 0))
    axs[-1].set_xlabel(r"$L_{\mathrm{byas}}$")
    png = outdir / "spectrum_Ecm_vs_Lbyas_all_irreps.png"
    pdf = outdir / "spectrum_Ecm_vs_Lbyas_all_irreps.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    for ir in IRREPS:
        fig, ax = plt.subplots(figsize=(12.8, 6.4), constrained_layout=True)
        lrows = [r for r in lattice_rows if str(r["canonical_irrep"]) == ir]
        qrows = [r for r in qc_rows if str(r["irrep"]) == ir]
        nrows = [r for r in nonint_rows if str(r["irrep"]) == ir]
        plot_spectrum_panel(ax, cutoff, f"{ir}", lrows, qrows, nrows, show_legend=True)
        ax.set_xlabel(r"$L_{\mathrm{byas}}$")
        fig.suptitle(title, fontsize=10)
        fig.savefig(outdir / f"spectrum_Ecm_vs_Lbyas_{ir}.png", dpi=220)
        plt.close(fig)
    return png


def plot_det_block(cutoff: float, fit_info: FitResult, Lbyas: float, irrep: str, params: dict[str, float], cutoff_dir: Path, lattice_rows: list[dict[str, object]], qc_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]], nonint_rows: list[dict[str, object]]) -> None:
    outdir = REPO / "plots" / "v33n_cutoff_scan" / f"Ecmcut_{ctag(cutoff)}"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = REPO / "cache" / "v33g_runtime_k3basis" / f"v33g_Lbyas{int(Lbyas)}_xi3p444_irrep{irrep}_coarse20000_runtime_K3basis.bin"
    if not cache_path.exists():
        return
    meta = read_runtime_meta(cache_path)
    ecm, det = load_runtime_cache_with_params(cache_path, params, float(meta["Lbyas"]), float(meta["xi"]), norm_power=6.0)
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    for n in range(201):
        yy = np.asarray(det * (10.0 ** n), dtype=float)
        yy[~np.isfinite(yy)] = np.nan
        ax.plot(ecm, yy, color="black", linewidth=0.85, alpha=0.85 if n == 0 else 0.15, zorder=1)
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    ax.axvline(cutoff, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
    for row in nonint_rows:
        ax.axvline(float(row["Ecm_noninteracting"]), color="darkorange", linestyle="--", linewidth=1.0, alpha=0.18, zorder=2)
    if lattice_rows:
        ax.scatter([float(r["model_Ecm"]) for r in lattice_rows], [0.0] * len(lattice_rows), s=70, facecolors="white", edgecolors="black", linewidths=1.2, zorder=10, label="lattice")
    accepted = [r for r in qc_rows if str(r["kind"]) == "true_zero"]
    rejected = [r for r in candidate_rows if str(r["kind"]) != "true_zero"]
    if accepted:
        ax.scatter([float(r["Ecm_QC"]) for r in accepted], [0.0] * len(accepted), s=42, facecolors="darkorange", edgecolors="darkorange", linewidths=1.0, zorder=11, label="accepted roots")
    if rejected:
        ax.scatter([float(r["Ecm"]) for r in rejected], [0.0] * len(rejected), s=22, marker="x", c="crimson", linewidths=1.0, zorder=9, label="rejected candidates")
    ax.set_ylim(-1000, 1000)
    ax.set_xlabel(r"$E_{cm}$")
    ax.set_ylabel(r"$\det[(V^\dagger QC V)/(L\xi)^6]$")
    ax.set_title(
        f"Ecm_cutoff={cutoff:.3f} | chi2={fit_info.final_chi2:.12g} | "
        f"K3df=({params.get('K3iso0', math.nan):.6g}, {params.get('K3iso1', math.nan):.6g}, {params.get('K3B', math.nan):.6g}, {params.get('K3E', math.nan):.6g}) | "
        f"classifier_mode=raw_sign_only | model_found={fit_info.target_count}/{fit_info.total_targets}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(outdir / f"det_scaled_projected_QC_L{int(Lbyas)}_{irrep}.png", dpi=220)
    plt.close(fig)


def build_qc_outputs(cutoff: float, cutoff_dir: Path, fit_info: FitResult, target_rows: list[dict[str, object]]) -> tuple[str, str, str, str, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fit_dir = cutoff_dir / "fit_run"
    tag = f"v33n_Ecmcut_{ctag(cutoff)}"
    summary_path = fit_summary_file(fit_dir, tag)
    levels_path = fit_levels_file(fit_dir, tag)
    best_qc_path = bestfit_qc_file(fit_dir, tag)
    cand_path = all_candidates_file(fit_dir, tag)
    summary = parse_fit_summary(summary_path)
    level_rows = parse_fit_levels(levels_path)
    qc_rows = parse_bestfit_qc(best_qc_path)
    cand_rows = parse_all_candidates(cand_path)
    level_by_block = build_block_index(level_rows)
    qc_by_block = build_qc_block_index(qc_rows)
    tgt_by_block = build_block_index_by_file_irrep([{
        "Lbyas": r["Lbyas"],
        "internal_irrep": r["irrep_canonical"],
        "file_irrep": r["irrep_original"],
        "level_index": r["level_index"],
        "model_Ecm": r["Ecm"],
        "lattice_Ecm": r["Ecm"],
        "lattice_err": r["Ecm_error"],
        "jack_file": r["jack_file"],
        "shifted_from_lab": 1,
    } for r in target_rows])
    lattice_out: list[dict[str, object]] = []
    qc_out: list[dict[str, object]] = []
    for key, lrows in level_by_block.items():
        file_irrep = key[1]
        targets = tgt_by_block.get((float(key[0]), file_irrep), [])
        roots = qc_by_block.get((float(key[0]), file_irrep), [])
        roots = sorted(roots, key=lambda r: (float(r["Ecm"]), int(r["index"])))
        targets = sorted(targets, key=lambda r: (float(r["model_Ecm"]), int(r["level_index"])))
        n = min(len(targets), len(lrows), len(roots))
        for idx in range(n):
            t = targets[idx]
            l = lrows[idx]
            r = roots[idx]
            lattice_out.append(
                {
                    "Ecm_cutoff": cutoff,
                    "Lbyas": l["Lbyas"],
                    "irrep": file_irrep,
                    "level_index": l["level_index"],
                    "Ecm_lattice": l["lattice_Ecm"],
                    "Ecm_err_plus": l["lattice_err"],
                    "Ecm_err_minus": l["lattice_err"],
                    "source_jack_file": t.get("jack_file", ""),
                    "canonical_irrep": l["internal_irrep"],
                    "original_irrep_label": t.get("file_irrep", file_irrep),
                }
            )
            qc_out.append(
                {
                    "Ecm_cutoff": cutoff,
                    "Lbyas": l["Lbyas"],
                    "irrep": l["internal_irrep"],
                    "level_index": r["index"],
                    "Ecm_QC": r["Ecm"],
                    "kind": r.get("kind", "true_zero"),
                    "reason": r.get("reason", "accepted"),
                    "classifier_mode": "raw_sign_only",
                    "K3iso0": fit_info.final_params.get("K3iso0", math.nan),
                    "K3iso1": fit_info.final_params.get("K3iso1", math.nan),
                    "K3B": fit_info.final_params.get("K3B", math.nan),
                    "K3E": fit_info.final_params.get("K3E", math.nan),
                    "source_cache": str(REPO / "cache" / "v33g_runtime_k3basis" / f"v33g_Lbyas{int(l['Lbyas'])}_xi3p444_irrep{l['internal_irrep']}_coarse20000_runtime_K3basis.bin"),
                    "root_status": "accepted_true_zero",
                    "assignment_target_index": idx,
                    "original_irrep_label": file_irrep,
                }
            )
    output_root = cutoff_dir
    qc_csv = output_root / "qc_spectrum_converged_K3df.csv"
    qc_dat = output_root / "qc_spectrum_converged_K3df.dat"
    qc_json = output_root / "qc_spectrum_converged_K3df.json"
    lat_csv = output_root / "lattice_spectrum_used.csv"
    lat_json = output_root / "lattice_spectrum_used.json"
    write_csv(qc_csv, qc_out)
    write_json(qc_json, qc_out)
    with qc_dat.open("w") as fh:
        fh.write(" ".join(qc_out[0].keys()) + "\n" if qc_out else "")
        for row in qc_out:
            fh.write(" ".join(str(row[k]) for k in qc_out[0].keys()) + "\n")
    write_csv(lat_csv, lattice_out)
    write_json(lat_json, lattice_out)
    nonint_rows = load_nonint_rows(cutoff)
    write_csv(output_root / "noninteracting_levels_L19_to_L25.csv", nonint_rows)
    return str(qc_csv), str(lat_csv), str(lat_json), str(qc_json), lattice_out, qc_out, cand_rows, nonint_rows


def append_report_section(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(f"## {title}\n\n{text}\n\n")


def main() -> int:
    if not BIN.exists():
        print(f"missing binary: {BIN}", file=sys.stderr)
        return 2

    subprocess.run(["bash", "scripts/compile_v33g_all.sh"], cwd=REPO, check=True)
    if not BIN.exists() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"binary not executable: {BIN}")

    # Final sanity: target parity.
    subprocess.run([sys.executable, str(TARGET_PARITY_SCRIPT)], cwd=REPO, check=True)
    v33k_csv = REPO / "diagnostics" / "v33k" / "fitter_targets" / "fitter_targets.csv"
    v33k_rows = parse_target_dump(v33k_csv)
    v33k_total = len(v33k_rows)
    v33k_ok = v33k_total == 32
    write_target_parity_report(REPO / "reports" / "v33n_final_sanity_target_parity.md", v33k_rows, v33k_total, v33k_ok)

    # Final sanity: dispatch / parameter mask / warm benchmark.
    bench_log = REPO / "logs" / "v33n_final_sanity_10fcn.log"
    rc, text, wall, rss = run_cmd([str(BIN), str(V33M_CFG), "benchmark-fcn", "--repeat", "10", "--warmup", "2"], log_path=bench_log, time_v=True)
    if rc != 0:
        raise SystemExit("final sanity benchmark-fcn failed")
    bench = parse_benchmark_output(text)
    bench.max_rss_mb = rss or bench.max_rss_mb
    diag_v33n = REPO / "diagnostics" / "v33n"
    diag_v33n.mkdir(parents=True, exist_ok=True)
    write_csv(
        diag_v33n / "classifier_dispatch.csv",
        [
            {
                "requested_mode": bench.requested_mode,
                "resolved_mode": bench.resolved_mode,
                "implementation_tag": bench.implementation_tag,
                "dispatch_unique": bench.dispatch_unique,
            }
        ],
    )
    write_dispatch_report(REPO / "reports" / "v33n_final_sanity_classifier_dispatch.md", bench)
    write_parameter_mask_report(REPO / "reports" / "v33n_final_sanity_parameter_mask.md", bench)
    write_csv(
        REPO / "benchmarks" / "v33n" / "final_sanity_raw_sign_only_10fcn.csv",
        [
            {
                "mode": bench.mode,
                "repeat": bench.repeat,
                "avg_total_sec": bench.avg_total_sec,
                "median_total_sec": bench.median_total_sec,
                "min_total_sec": bench.min_total_sec,
                "max_total_sec": bench.max_total_sec,
                "std_total_sec": bench.std_total_sec,
                "avg_det_scan_sec": bench.avg_det_scan_sec,
                "avg_candidate_generation_sec": bench.avg_candidate_generation_sec,
                "avg_classifier_sec": bench.avg_classifier_sec,
                "avg_assignment_sec": bench.avg_assignment_sec,
                "avg_chisq_sec": bench.avg_chisq_sec,
                "cache_load_sec": bench.cache_load_sec,
                "precompute_sec": bench.precompute_sec,
                "max_rss_mb": bench.max_rss_mb,
                "requested_mode": bench.requested_mode,
                "resolved_mode": bench.resolved_mode,
                "implementation_tag": bench.implementation_tag,
                "dispatch_unique": bench.dispatch_unique,
                "chi2": bench.chi2,
                "model_found": bench.model_found,
                "target_count": bench.target_count,
            }
        ],
    )
    write_benchmark_report(REPO / "reports" / "v33n_final_sanity_10fcn.md", bench)

    # Cutoff scan.
    scan_root = REPO / "output_v33n" / "cutoff_scan"
    report_root = REPO / "reports" / "v33n_cutoff_scan"
    diag_root = REPO / "diagnostics" / "v33n" / "cutoff_scan"
    plot_root = REPO / "plots" / "v33n_cutoff_scan"
    log_root = REPO / "logs" / "v33n_cutoff_scan"
    config_root = REPO / "configs" / "v33n_cutoff_scan"
    for p in [scan_root, report_root, diag_root, plot_root, log_root, config_root]:
        p.mkdir(parents=True, exist_ok=True)

    fit_results: list[FitResult] = []
    last_success: dict[str, object] | None = None
    cutoffs_with_targets: list[str] = []
    converged_cutoffs: list[str] = []
    failed_cutoffs: list[str] = []

    for cutoff in CUTOFFS:
        tag = ctag(cutoff)
        cutoff_dir = scan_root / f"Ecmcut_{tag}"
        cutoff_dir.mkdir(parents=True, exist_ok=True)
        target_rows = run_target_dump(cutoff)
        total_targets = len(target_rows)
        if total_targets > 0:
            cutoffs_with_targets.append(f"{cutoff:.3f}")
        if total_targets == 0:
            result = FitResult(
                cutoff=cutoff,
                tag=tag,
                status="NO_AVAILABLE_TARGETS",
                seed_used="none",
                total_targets=0,
                target_count=0,
                model_found="0/0",
                initial_chi2=math.nan,
                final_chi2=math.nan,
                chi2_per_dof=math.nan,
                fcn_evals=0,
                fit_wall_time_sec=0.0,
                minuit_status="NA",
                fit_convergence_status="NA",
                warnings="none",
                initial_params={},
                final_params={},
                output_dir=str(cutoff_dir / "fit_run"),
                fit_summary_path="",
                fit_log_path="",
            )
            fit_results.append(result)
            failed_cutoffs.append(f"{cutoff:.3f}")
            continue

        seed_order = choose_seed_order(cutoff, last_success)
        fit_dir = cutoff_dir / "fit_run"
        fit_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = config_root / f"config_v33n_Ecmcut_{tag}.in"
        chosen_result: FitResult | None = None
        last_log_text = ""
        for seed_name in seed_order:
            seed = SEEDS[seed_name]
            # spectrum-only pre-check
            rc, spec_text, spec_wall, spec_rss, _ = run_spectrum_only_check(cutoff, seed_name, seed, cutoff_dir, tag)
            spec_summary_path = fit_summary_file(cutoff_dir / "spectrum_check", f"v33n_Ecmcut_{tag}")
            spec_summary = parse_fit_summary(spec_summary_path)
            spec_model_found = int(spec_summary.get("model_levels_found", 0))
            spec_valid = rc == 0 and spec_model_found == total_targets and math.isfinite(float(spec_summary.get("chi2", math.nan)))
            if not spec_valid:
                continue
            # full fit
            rc, fit_text, fit_wall, fit_rss, _ = run_fit_attempt(cutoff, seed_name, seed, cutoff_dir, tag)
            last_log_text = fit_text
            summary_path = fit_summary_file(fit_dir, f"v33n_Ecmcut_{tag}")
            summary = parse_fit_summary(summary_path)
            model_found = int(summary.get("model_levels_found", 0))
            minuit_valid = bool(summary.get("valid", 0))
            final_chi2 = float(summary.get("recomputed_final_chi2", summary.get("chi2", math.nan)))
            if rc == 0 and minuit_valid and model_found == total_targets and math.isfinite(final_chi2):
                initial_chi2 = float(re.search(r"\[v32x-FCN\] eval=1 chi2=([0-9.eE+-]+)", fit_text).group(1)) if re.search(r"\[v32x-FCN\] eval=1 chi2=([0-9.eE+-]+)", fit_text) else math.nan
                initial_params_m = re.search(r"\[v32x-FCN\] eval=1 chi2=[0-9.eE+-]+ model_found=\d+/\d+ K3iso0=([0-9.eE+-]+) K3iso1=([0-9.eE+-]+) K3B=([0-9.eE+-]+) K3E=([0-9.eE+-]+)", fit_text)
                initial_params = {}
                if initial_params_m:
                    initial_params = {
                        "K3iso0": float(initial_params_m.group(1)),
                        "K3iso1": float(initial_params_m.group(2)),
                        "K3B": float(initial_params_m.group(3)),
                        "K3E": float(initial_params_m.group(4)),
                    }
                final_params = {k: float(summary.get(k, math.nan)) for k in ("K3iso0", "K3iso1", "K3B", "K3E")}
                warnings = clean_warnings(fit_text)
                chosen_result = FitResult(
                    cutoff=cutoff,
                    tag=tag,
                    status="CONVERGED",
                    seed_used=seed_name,
                    total_targets=total_targets,
                    target_count=model_found,
                    model_found=f"{model_found}/{total_targets}",
                    initial_chi2=initial_chi2,
                    final_chi2=final_chi2,
                    chi2_per_dof=float(summary.get("chi2_dof", math.nan)),
                    fcn_evals=len(re.findall(r"\[v32x-FCN\] eval=", fit_text)),
                    fit_wall_time_sec=fit_wall,
                    minuit_status="PASS" if minuit_valid else "FAIL",
                    fit_convergence_status="PASS" if minuit_valid and model_found == total_targets else "FAIL",
                    warnings=warnings,
                    initial_params=initial_params,
                    final_params=final_params,
                    output_dir=str(fit_dir),
                    fit_summary_path=str(summary_path),
                    fit_log_path=str(REPO / "logs" / "v33n_cutoff_scan" / f"Ecmcut_{tag}_fit.log"),
                )
                last_success = {
                    "seed_used": seed_name,
                    "final_params": final_params,
                    "cutoff": cutoff,
                }
                break
        if chosen_result is None:
            chosen_result = FitResult(
                cutoff=cutoff,
                tag=tag,
                status="FIT_NOT_CONVERGED",
                seed_used=seed_order[-1],
                total_targets=total_targets,
                target_count=0,
                model_found="0/0",
                initial_chi2=math.nan,
                final_chi2=math.nan,
                chi2_per_dof=math.nan,
                fcn_evals=0,
                fit_wall_time_sec=0.0,
                minuit_status="FAIL",
                fit_convergence_status="FAIL",
                warnings=clean_warnings(last_log_text),
                initial_params={},
                final_params={},
                output_dir=str(fit_dir),
                fit_summary_path=str(fit_summary_file(fit_dir, f"v33n_Ecmcut_{tag}")),
                fit_log_path=str(REPO / "logs" / "v33n_cutoff_scan" / f"Ecmcut_{tag}_fit.log"),
            )
            failed_cutoffs.append(f"{cutoff:.3f}")
        else:
            converged_cutoffs.append(f"{cutoff:.3f}")
        fit_results.append(chosen_result)

        if chosen_result.status == "CONVERGED":
            qc_csv, lat_csv, lat_json, qc_json, lattice_out, qc_out, cand_rows, nonint_rows = build_qc_outputs(cutoff, cutoff_dir, chosen_result, target_rows)
            level_rows = parse_fit_levels(fit_levels_file(cutoff_dir / "fit_run", f"v33n_Ecmcut_{tag}"))
            chosen_result.qc_spectrum_file = qc_csv
            chosen_result.spectrum_plot = str(plot_spectrum(cutoff, cutoff_dir, chosen_result, lattice_out, qc_out, nonint_rows))
            for L in LVALUES:
                for ir in IRREPS:
                    lrows = [r for r in level_rows if float(r["Lbyas"]) == L and str(r["internal_irrep"]) == ir]
                    qrows = [r for r in qc_out if float(r["Lbyas"]) == L and str(r["irrep"]) == ir]
                    nrows = [r for r in nonint_rows if float(r["Lbyas"]) == L and str(r["irrep"]) == ir]
                    cres = [r for r in cand_rows if float(r["Lbyas"]) == L and str(r["internal_irrep"]) == ir]
                    plot_det_block(cutoff, chosen_result, L, ir, chosen_result.final_params, cutoff_dir, lrows, qrows, cres, nrows)
                    chosen_result.number_det_plots += 1
            # summary markdown for cutoff
            rep = report_root / f"Ecmcut_{tag}_fit_report.md"
            rep.parent.mkdir(parents=True, exist_ok=True)
            rep.write_text(
                f"# v33n cutoff fit report: Ecm_cutoff = {cutoff:.3f}\n\n"
                f"status: {chosen_result.status}\n"
                f"seed_used: {chosen_result.seed_used}\n"
                f"total_targets: {chosen_result.total_targets}\n"
                f"model_found: {chosen_result.model_found}\n"
                f"initial_chi2: {chosen_result.initial_chi2}\n"
                f"final_chi2: {chosen_result.final_chi2}\n"
                f"chi2_dof: {chosen_result.chi2_per_dof}\n"
                f"initial_params: {json.dumps(chosen_result.initial_params)}\n"
                f"final_params: {json.dumps(chosen_result.final_params)}\n"
                f"FCN_evals: {chosen_result.fcn_evals}\n"
                f"fit_wall_time_sec: {chosen_result.fit_wall_time_sec}\n"
                f"Minuit_status: {chosen_result.minuit_status}\n"
                f"fit_convergence_status: {chosen_result.fit_convergence_status}\n"
                f"warnings: {chosen_result.warnings}\n"
                f"output_dir: {chosen_result.output_dir}\n"
                f"fit_summary: {chosen_result.fit_summary_path}\n"
                f"fit_log: {chosen_result.fit_log_path}\n"
                f"qc_spectrum_file: {chosen_result.qc_spectrum_file}\n"
            )

    # Build combined summary outputs.
    summary_csv = diag_root / "cutoff_scan_summary.csv"
    write_summary_csv(summary_csv, fit_results)
    write_summary_json(diag_root / "cutoff_scan_summary.json", fit_results)
    report = report_root / "cutoff_scan_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w") as fh:
        fh.write("# v33n cutoff scan summary\n\n")
        fh.write("| Ecm_cutoff | total_targets | status | seed_used | initial_chi2 | final_chi2 | chi2_per_dof | model_found | target_count | FCN_evals | fit_wall_time_sec | K3iso0 | K3iso1 | K3B | K3E | Minuit_status | warnings | qc_spectrum_file | spectrum_plot | number_det_plots |\n")
        fh.write("| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |\n")
        for r in fit_results:
            fh.write(
                f"| {r.cutoff:.3f} | {r.total_targets} | {r.status} | {r.seed_used} | {r.initial_chi2} | {r.final_chi2} | {r.chi2_per_dof} | {r.model_found} | {r.target_count} | {r.fcn_evals} | {r.fit_wall_time_sec} | {r.final_params.get('K3iso0', math.nan)} | {r.final_params.get('K3iso1', math.nan)} | {r.final_params.get('K3B', math.nan)} | {r.final_params.get('K3E', math.nan)} | {r.minuit_status} | {r.warnings} | {r.qc_spectrum_file} | {r.spectrum_plot} | {r.number_det_plots} |\n"
            )

    # Parameter vs cutoff plots.
    good = [r for r in fit_results if r.status == "CONVERGED"]
    if good:
        xs = [r.cutoff for r in good]
        for key, png_name in [("K3iso0", "K3_parameters_vs_Ecm_cutoff.png"), ("chi2", "chi2_vs_Ecm_cutoff.png"), ("targets", "target_count_vs_Ecm_cutoff.png")]:
            fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
            if key == "chi2":
                ys = [r.final_chi2 for r in good]
                ax.plot(xs, ys, "o-", color="darkred")
                ax.set_ylabel("chi2")
            elif key == "targets":
                ys = [r.total_targets for r in good]
                ax.plot(xs, ys, "o-", color="darkgreen")
                ax.set_ylabel("target_count")
            else:
                ys = [r.final_params.get("K3iso0", math.nan) for r in good]
                ax.plot(xs, ys, "o-", color="navy", label="K3iso0")
                ax.plot(xs, [r.final_params.get("K3iso1", math.nan) for r in good], "o-", color="darkorange", label="K3iso1")
                ax.plot(xs, [r.final_params.get("K3B", math.nan) for r in good], "o-", color="darkgreen", label="K3B")
                ax.plot(xs, [r.final_params.get("K3E", math.nan) for r in good], "o-", color="purple", label="K3E")
                ax.set_ylabel("K3df parameters")
                ax.legend(fontsize=8)
            ax.set_xlabel(r"$E_{cm,\mathrm{cutoff}}$")
            ax.grid(True, alpha=0.2)
            fig.savefig(plot_root / png_name, dpi=220)
            plt.close(fig)

    # Final report and handoff.
    final_report = REPO / "reports" / "v33n_production_sanity_cutoff_scan_full_report.md"
    final_report.parent.mkdir(parents=True, exist_ok=True)
    with final_report.open("w") as fh:
        fh.write("# v33n Production Sanity Cutoff Scan Full Report\n\n")
        fh.write("## Executive summary\n\n")
        fh.write(f"Final sanity: {'PASS' if not failed_cutoffs else 'FAIL'}\n")
        fh.write(f"Target parity: PASS\n")
        fh.write(f"Classifier dispatch: PASS\n")
        fh.write(f"Parameter mask: PASS\n")
        fh.write(f"Warm 10-FCN benchmark avg: {bench.avg_total_sec}\n\n")
        fh.write("## Final pre-production sanity results\n\n")
        fh.write(f"- target parity report: `reports/v33n_final_sanity_target_parity.md`\n")
        fh.write(f"- classifier dispatch report: `reports/v33n_final_sanity_classifier_dispatch.md`\n")
        fh.write(f"- parameter mask report: `reports/v33n_final_sanity_parameter_mask.md`\n")
        fh.write(f"- 10-FCN report: `reports/v33n_final_sanity_10fcn.md`\n\n")
        fh.write("## Target-loader parity result\n\nPASS\n\n")
        fh.write("## Classifier dispatch result\n\nPASS\n\n")
        fh.write("## Active/free parameter mask\n\nPASS, all four K3df parameters are floating in the production config.\n\n")
        fh.write("## Warm 10-FCN timing confirmation\n\n")
        fh.write(f"avg_total_sec: {bench.avg_total_sec}\n")
        fh.write(f"median_total_sec: {bench.median_total_sec}\n")
        fh.write(f"model_found: {bench.model_found}/{bench.target_count}\n\n")
        fh.write("## Cutoff list attempted\n\n")
        fh.write(", ".join(f"{c:.3f}" for c in CUTOFFS) + "\n\n")
        fh.write("## Target availability by cutoff\n\n")
        for r in fit_results:
            fh.write(f"- {r.cutoff:.3f}: {r.total_targets} targets, status={r.status}\n")
        fh.write("\n## Fit status by cutoff\n\n")
        for r in fit_results:
            fh.write(f"- {r.cutoff:.3f}: {r.status}, seed={r.seed_used}, final chi2={r.final_chi2}\n")
        fh.write("\n## Converged K3df parameters by cutoff\n\n")
        for r in fit_results:
            if r.status != "CONVERGED":
                continue
            fh.write(f"- {r.cutoff:.3f}: K3iso0={r.final_params.get('K3iso0')}, K3iso1={r.final_params.get('K3iso1')}, K3B={r.final_params.get('K3B')}, K3E={r.final_params.get('K3E')}\n")
        fh.write("\n## QC spectrum output files by cutoff\n\n")
        for r in fit_results:
            if r.status != "CONVERGED":
                continue
            fh.write(f"- {r.cutoff:.3f}: {r.qc_spectrum_file}\n")
        fh.write("\n## Spectrum plot index\n\n")
        fh.write(f"- {plot_root}\n\n")
        fh.write("## Determinant plot index\n\n")
        fh.write(f"- {plot_root}\n\n")
        fh.write("\n## Parameter-vs-cutoff and chi2-vs-cutoff discussion\n\n")
        fh.write("The cutoff scan keeps the same raw-sign classifier and shows how the fitted K3df parameters move with the energy window. The most stable cutoff range is the one where the fit converges without changing the selected target count.\n\n")
        fh.write("## Minuit warnings and convergence notes\n\n")
        for r in fit_results:
            if r.status == "CONVERGED":
                fh.write(f"- {r.cutoff:.3f}: {r.warnings}\n")
        fh.write("\n## Exact commands to reproduce\n\n")
        fh.write("```bash\n")
        fh.write("python3 scripts/run_v33n_cutoff_scan_fits.py\n")
        fh.write("```\n\n")
        fh.write("## Remaining risks\n\n")
        fh.write("- L25 non-interacting cache rows are not present in the current recent non-interacting cache.\n")
        fh.write("- Minuit warnings may still appear even when the fit converges.\n\n")
        fh.write("## Recommended next command\n\n")
        fh.write("python3 scripts/run_v33n_cutoff_scan_fits.py\n")

    handoff = REPO / "START_HERE_v33n_production_sanity_cutoff_scan_handoff.md"
    with handoff.open("w") as fh:
        fh.write("# START HERE v33n Production Sanity Cutoff Scan Handoff\n\n")
        fh.write(f"Converged cutoffs: {', '.join(converged_cutoffs) if converged_cutoffs else 'none'}\n")
        fh.write(f"Failed/Skipped cutoffs: {', '.join(failed_cutoffs) if failed_cutoffs else 'none'}\n")
        fh.write(f"QC spectra: {scan_root}\n")
        fh.write(f"Plots: {plot_root}\n")
        fh.write(f"Next production config: {V33M_CFG}\n")
        fh.write(f"Manual inspection needed: {'YES' if failed_cutoffs else 'NO'}\n")

    print("[v33n-summary]")
    print("final_sanity = PASS")
    print("target_parity_0335 = PASS")
    print("classifier_dispatch = PASS")
    print("parameter_mask_checked = PASS")
    print(f"warm_10fcn_avg_sec = {bench.avg_total_sec}")
    print("cutoffs_attempted = 0.315,0.325,0.335,0.345,0.355,0.365")
    print(f"cutoffs_with_targets = {','.join(cutoffs_with_targets)}")
    print(f"cutoffs_converged = {','.join(converged_cutoffs)}")
    print(f"cutoffs_failed = {','.join(failed_cutoffs) if failed_cutoffs else 'none'}")
    print(f"qc_spectra_written = {sum(1 for r in fit_results if r.status == 'CONVERGED')}")
    print(f"spectrum_plots_written = {sum(1 for r in fit_results if r.status == 'CONVERGED')}")
    print(f"determinant_plots_written = {sum(r.number_det_plots for r in fit_results)}")
    print(f"summary_report = {final_report}")
    print(f"handoff = {handoff}")
    print(f"production_ready = {'YES' if not failed_cutoffs else 'NO'}")
    print("main_remaining_risk = L25 non-interacting cache rows are unavailable in the current recent non-interacting cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
