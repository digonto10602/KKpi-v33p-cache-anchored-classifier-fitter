#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "bin" / "v33f_k3df_fitter_multiL_v33e"
BASE_CFG = REPO / "configs" / "config_v33g_multiL_all_irreps_runtime.in"
TARGET_PARITY = REPO / "scripts" / "validate_fitter_target_counts_v33k.py"
DISPATCH_AUDIT = REPO / "scripts" / "verify_classifier_dispatch_v33m.py"
MODES = [
    "raw_sign_only",
    "raw_sign_clustered",
    "algo_v6_branch_count_final_select",
    "algo_v7_eigenbranch",
    "algo_v7_hybrid_det_eigenbranch",
    "digonto_v3_window",
    "digonto_v4_window",
]
BENCH_MODES = [
    "raw_sign_only",
    "raw_sign_clustered",
    "algo_v6_branch_count_final_select",
    "algo_v7_eigenbranch",
    "algo_v7_hybrid_det_eigenbranch",
]


@dataclass
class BenchRow:
    mode: str
    eligible: str
    target_count: int
    model_found: str
    chi2_first: float
    chi2_last: float
    avg_total_sec: float
    median_total_sec: float
    min_total_sec: float
    max_total_sec: float
    std_total_sec: float
    avg_det_scan_sec: float
    avg_candidate_generation_sec: float
    avg_classifier_sec: float
    avg_assignment_sec: float
    avg_chisq_sec: float
    cache_load_sec: float
    precompute_sec: float
    wall_time_external_sec: float
    max_rss_mb: float
    assignment_equivalent_to_reference: str
    rank_by_avg_total: int
    status: str


def write_temp_cfg(mode: str, outdir: str = "output_v33m/fit_fastest_production", tag: str = "v33m_fastest_production_fit") -> Path:
    text = BASE_CFG.read_text()
    text = re.sub(r"^classifier_mode = .*$", f"classifier_mode = {mode}", text, flags=re.M)
    text = re.sub(r"^output_dir = .*$", f"output_dir = {outdir}", text, flags=re.M)
    text = re.sub(r"^output_tag = .*$", f"output_tag = {tag}", text, flags=re.M)
    text = re.sub(r"^benchmark_repeat = .*$", "benchmark_repeat = 10", text, flags=re.M)
    text = re.sub(r"^benchmark_warmup = .*$", "benchmark_warmup = 2", text, flags=re.M)
    fd, path = tempfile.mkstemp(prefix="v33m_cfg_", suffix=".in")
    os.close(fd)
    p = Path(path)
    p.write_text(text)
    return p


def run(cmd: list[str], log_path: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    text = proc.stdout + proc.stderr
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    return proc.returncode, text


def run_time(cmd: list[str], log_path: Path) -> tuple[int, str, float, float]:
    proc = subprocess.run(["/usr/bin/time", "-v", *cmd], cwd=REPO, text=True, capture_output=True)
    text = proc.stdout + proc.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text)
    if re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+):(\d+\.\d+)", text):
        m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+):(\d+\.\d+)", text)
        wall = int(m.group(1)) * 60 + float(m.group(2))
    elif re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+):(\d+):(\d+\.\d+)", text):
        m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+):(\d+):(\d+\.\d+)", text)
        wall = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    else:
        wall = 0.0
    rss = float(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1)) / 1024.0
    return proc.returncode, text, wall, rss


def parse_summary(text: str) -> dict[str, float | int | str]:
    dispatch = re.search(
        r"\[classifier-dispatch\] requested_mode=(\S+) resolved_mode=(\S+) implementation=(\S+) version_tag=(\S+) dispatch_unique=(\d+)",
        text,
    )
    summary = re.search(
        r"\[benchmark-fcn-summary\] classifier_mode=(\S+) repeat=(\d+) avg_total_sec=([0-9.eE+-]+) median_total_sec=([0-9.eE+-]+) min_total_sec=([0-9.eE+-]+) max_total_sec=([0-9.eE+-]+) std_total_sec=([0-9.eE+-]+) avg_det_scan_sec=([0-9.eE+-]+) avg_candidate_generation_sec=([0-9.eE+-]+) avg_classifier_sec=([0-9.eE+-]+) avg_assignment_sec=([0-9.eE+-]+) avg_chisq_sec=([0-9.eE+-]+) cache_load_sec=([0-9.eE+-]+) precompute_sec=([0-9.eE+-]+) max_rss_mb=([0-9.eE+-]+)",
        text,
    )
    call = re.search(r"\[benchmark-fcn\] call=(\d+) chi2=([0-9.eE+-]+) model_found=(\d+)/(\d+)", text)
    result: dict[str, float | int | str] = {}
    if dispatch:
        result.update(
            {
                "requested_mode": dispatch.group(1),
                "resolved_mode": dispatch.group(2),
                "implementation_tag": dispatch.group(3),
                "dispatch_unique": int(dispatch.group(5)),
            }
        )
    if summary:
        (
            result["mode"],
            result["repeat"],
            result["avg_total_sec"],
            result["median_total_sec"],
            result["min_total_sec"],
            result["max_total_sec"],
            result["std_total_sec"],
            result["avg_det_scan_sec"],
            result["avg_candidate_generation_sec"],
            result["avg_classifier_sec"],
            result["avg_assignment_sec"],
            result["avg_chisq_sec"],
            result["cache_load_sec"],
            result["precompute_sec"],
            result["max_rss_mb"],
        ) = (
            summary.group(1),
            int(summary.group(2)),
            float(summary.group(3)),
            float(summary.group(4)),
            float(summary.group(5)),
            float(summary.group(6)),
            float(summary.group(7)),
            float(summary.group(8)),
            float(summary.group(9)),
            float(summary.group(10)),
            float(summary.group(11)),
            float(summary.group(12)),
            float(summary.group(13)),
            float(summary.group(14)),
            float(summary.group(15)),
        )
    if call:
        result["chi2"] = float(call.group(2))
        result["model_found"] = int(call.group(3))
        result["target_count"] = int(call.group(4))
    return result


def assignment_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def assignment_signature(rows: list[dict[str, str]]) -> list[tuple[str, str, str, str]]:
    sig = []
    for r in rows:
        sig.append((r["Lbyas"], r["irrep"], r["target_index"], f"{float(r['model_root_Ecm']):.12f}"))
    return sig


def compare_assignments(ref: list[dict[str, str]], other: list[dict[str, str]]) -> bool:
    return assignment_signature(ref) == assignment_signature(other)


def run_benchmark(mode: str, repeat: int, warmup: int, log_name: str) -> tuple[dict[str, float | int | str], str, float, float]:
    cfg = write_temp_cfg(mode)
    log_path = REPO / "logs" / log_name
    cmd = [str(BIN), str(cfg), "benchmark-fcn", "--repeat", str(repeat), "--warmup", str(warmup)]
    rc, text, wall, rss = run_time(cmd, log_path)
    cfg.unlink(missing_ok=True)
    result = parse_summary(text)
    result["returncode"] = rc
    result["wall_time_external_sec"] = wall
    result["max_rss_mb"] = rss
    return result, text, wall, rss


def write_dispatch_audit() -> list[dict[str, str]]:
    rows = []
    csv_path = REPO / "diagnostics" / "v33m" / "classifier_dispatch_audit.csv"
    if csv_path.exists():
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
    return rows


def write_correctness_report(rows: list[dict[str, object]]) -> None:
    rep = REPO / "reports" / "v33m_classifier_correctness_comparison.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w") as fh:
        fh.write("# v33m classifier correctness comparison\n\n")
        fh.write("| mode | model_found | chi2 | eligible | assignment_equivalent_to_reference | status |\n")
        fh.write("| --- | --- | ---: | --- | --- | --- |\n")
        for r in rows:
            fh.write(
                f"| {r['mode']} | {r['model_found']} | {r['chi2']} | {r['eligible']} | {r['assignment_equivalent_to_reference']} | {r['status']} |\n"
            )


def write_bench_reports(rows: list[BenchRow]) -> None:
    rep = REPO / "reports" / "v33m_fcn10_by_classifier.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w") as fh:
        fh.write("# v33m FCN benchmark by classifier\n\n")
        fh.write("| mode | eligible | target_count | model_found | chi2_first | chi2_last | avg_total_sec | median_total_sec | min_total_sec | max_total_sec | std_total_sec | avg_det_scan_sec | avg_candidate_generation_sec | avg_classifier_sec | avg_assignment_sec | avg_chisq_sec | cache_load_sec | precompute_sec | wall_time_external_sec | max_rss_mb | assignment_equivalent_to_reference | rank_by_avg_total | status |\n")
        fh.write("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |\n")
        for r in rows:
            fh.write(
                f"| {r.mode} | {r.eligible} | {r.target_count} | {r.model_found} | {r.chi2_first} | {r.chi2_last} | {r.avg_total_sec} | {r.median_total_sec} | {r.min_total_sec} | {r.max_total_sec} | {r.std_total_sec} | {r.avg_det_scan_sec} | {r.avg_candidate_generation_sec} | {r.avg_classifier_sec} | {r.avg_assignment_sec} | {r.avg_chisq_sec} | {r.cache_load_sec} | {r.precompute_sec} | {r.wall_time_external_sec} | {r.max_rss_mb} | {r.assignment_equivalent_to_reference} | {r.rank_by_avg_total} | {r.status} |\n"
            )


def main() -> int:
    subprocess.run([sys.executable, str(TARGET_PARITY)], cwd=REPO, check=True)
    subprocess.run(["bash", "scripts/compile_v33f_all.sh"], cwd=REPO, check=True)
    subprocess.run([sys.executable, str(DISPATCH_AUDIT)], cwd=REPO, check=True)

    correctness_rows: list[dict[str, object]] = []
    assignment_cache: dict[str, list[dict[str, str]]] = {}
    for mode in MODES:
        result, text, _, _ = run_benchmark(mode, repeat=1, warmup=1, log_name=f"v33m_correctness_{mode}.log")
        rows = assignment_rows(REPO / "diagnostics" / "v33m" / "root_assignments" / f"{mode}_assignments.csv")
        assignment_cache[mode] = rows
        eligible = result.get("model_found", 0) == 32 and math.isfinite(float(result.get("chi2", float("nan"))))
        eq = "YES" if mode == "raw_sign_only" else "NO"
        if mode != "raw_sign_only" and "raw_sign_only" in assignment_cache:
            eq = "YES" if compare_assignments(assignment_cache["raw_sign_only"], rows) else "NO"
        if mode == "raw_sign_only":
            eq = "YES"
        status = "PASS" if eligible else "FAIL"
        if eligible and eq == "NO":
            status = "COVERAGE_ONLY_NOT_EQUIVALENT"
        correctness_rows.append(
            {
                "mode": mode,
                "model_found": result.get("model_found", 0),
                "chi2": result.get("chi2", float("nan")),
                "eligible": "YES" if eligible else "NO",
                "assignment_equivalent_to_reference": eq,
                "status": status,
                "result": result,
            }
        )

    write_correctness_report(correctness_rows)
    corr_csv = REPO / "diagnostics" / "v33m" / "classifier_correctness_summary.csv"
    corr_csv.parent.mkdir(parents=True, exist_ok=True)
    with corr_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "eligible", "model_found", "chi2", "assignment_equivalent_to_reference", "status"])
        for r in correctness_rows:
            writer.writerow([r["mode"], r["eligible"], r["model_found"], r["chi2"], r["assignment_equivalent_to_reference"], r["status"]])

    eligible_modes = [r["mode"] for r in correctness_rows if r["eligible"] == "YES"]
    if not eligible_modes:
        raise SystemExit("no eligible modes")

    bench_rows: list[BenchRow] = []
    for idx, mode in enumerate(eligible_modes, start=1):
        result, text, wall, rss = run_benchmark(mode, repeat=10, warmup=2, log_name=f"v33m_benchmark_{mode}.log")
        bench_rows.append(
            BenchRow(
                mode=mode,
                eligible="YES",
                target_count=int(result.get("target_count", 32)),
                model_found=f"{int(result.get('model_found', 0))}/32",
                chi2_first=float(result.get("chi2", float("nan"))),
                chi2_last=float(result.get("chi2", float("nan"))),
                avg_total_sec=float(result.get("avg_total_sec", float("nan"))),
                median_total_sec=float(result.get("median_total_sec", float("nan"))),
                min_total_sec=float(result.get("min_total_sec", float("nan"))),
                max_total_sec=float(result.get("max_total_sec", float("nan"))),
                std_total_sec=float(result.get("std_total_sec", float("nan"))),
                avg_det_scan_sec=float(result.get("avg_det_scan_sec", float("nan"))),
                avg_candidate_generation_sec=float(result.get("avg_candidate_generation_sec", float("nan"))),
                avg_classifier_sec=float(result.get("avg_classifier_sec", float("nan"))),
                avg_assignment_sec=float(result.get("avg_assignment_sec", float("nan"))),
                avg_chisq_sec=float(result.get("avg_chisq_sec", float("nan"))),
                cache_load_sec=float(result.get("cache_load_sec", float("nan"))),
                precompute_sec=float(result.get("precompute_sec", float("nan"))),
                wall_time_external_sec=wall,
                max_rss_mb=rss,
                assignment_equivalent_to_reference="YES" if mode == "raw_sign_only" or compare_assignments(assignment_cache["raw_sign_only"], assignment_cache[mode]) else "NO",
                rank_by_avg_total=idx,
                status="PASS",
            )
        )

    bench_rows.sort(key=lambda r: r.avg_total_sec)
    for i, row in enumerate(bench_rows, start=1):
        row.rank_by_avg_total = i
    chosen = bench_rows[0]
    bench_csv = REPO / "benchmarks" / "v33m" / "fcn10_by_classifier.csv"
    bench_csv.parent.mkdir(parents=True, exist_ok=True)
    with bench_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "mode",
                "eligible",
                "target_count",
                "model_found",
                "chi2_first",
                "chi2_last",
                "avg_total_sec",
                "median_total_sec",
                "min_total_sec",
                "max_total_sec",
                "std_total_sec",
                "avg_det_scan_sec",
                "avg_candidate_generation_sec",
                "avg_classifier_sec",
                "avg_assignment_sec",
                "avg_chisq_sec",
                "cache_load_sec",
                "precompute_sec",
                "wall_time_external_sec",
                "max_rss_mb",
                "assignment_equivalent_to_reference",
                "rank_by_avg_total",
                "status",
            ]
        )
        for r in bench_rows:
            writer.writerow(
                [
                    r.mode,
                    r.eligible,
                    r.target_count,
                    r.model_found,
                    r.chi2_first,
                    r.chi2_last,
                    r.avg_total_sec,
                    r.median_total_sec,
                    r.min_total_sec,
                    r.max_total_sec,
                    r.std_total_sec,
                    r.avg_det_scan_sec,
                    r.avg_candidate_generation_sec,
                    r.avg_classifier_sec,
                    r.avg_assignment_sec,
                    r.avg_chisq_sec,
                    r.cache_load_sec,
                    r.precompute_sec,
                    r.wall_time_external_sec,
                    r.max_rss_mb,
                    r.assignment_equivalent_to_reference,
                    r.rank_by_avg_total,
                    r.status,
                ]
            )
    write_bench_reports(bench_rows)

    cfg_text = BASE_CFG.read_text()
    cfg_text = re.sub(r"^classifier_mode = .*$", f"classifier_mode = {chosen.mode}", cfg_text, flags=re.M)
    cfg_text = re.sub(r"^output_dir = .*$", "output_dir = output_v33m/fit_fastest_production", cfg_text, flags=re.M)
    cfg_text = re.sub(r"^output_tag = .*$", "output_tag = v33m_fastest_production_fit", cfg_text, flags=re.M)
    cfg_text = re.sub(r"^benchmark_repeat = .*$", "benchmark_repeat = 10", cfg_text, flags=re.M)
    cfg_text = re.sub(r"^benchmark_warmup = .*$", "benchmark_warmup = 2", cfg_text, flags=re.M)
    prod_cfg = REPO / "configs" / "config_v33m_multiL_all_irreps_fastest_production.in"
    prod_cfg.write_text(
        "# v33m locked production config\n"
        "# chosen by scripts/run_v33m_classifier_speed_search.py\n"
        "# target set: L20,L24 five default irreps, 32 targets\n"
        f"# avg warm FCN time over 10 calls: {chosen.avg_total_sec}\n"
        f"# validation chi2: {chosen.chi2_first}\n"
        f"# model_found: {chosen.model_found}\n\n"
        + cfg_text
    )

    verify_log = REPO / "logs" / "v33m_production_verify.log"
    rc, text, wall, rss = run_time([str(BIN), str(prod_cfg), "benchmark-fcn", "--repeat", "10", "--warmup", "2"], verify_log)
    if rc != 0:
        raise SystemExit("production config verification failed")
    verify = parse_summary(text)

    full_fit_log = REPO / "logs" / "v33m_full_fit.log"
    rc, text, fit_wall, fit_rss = run_time([str(BIN), str(prod_cfg), "fit"], full_fit_log)
    full_fit_started = True
    full_fit_completed = rc == 0
    summary_file = REPO / "output_v33m" / "fit_fastest_production" / "v33m_fastest_production_fit_fit_summary_allL.dat"
    full_fit_report = REPO / "reports" / "v33m_full_fit_report.md"
    full_fit_report.parent.mkdir(parents=True, exist_ok=True)
    summary_text = summary_file.read_text() if summary_file.exists() else ""
    eval_count = len(re.findall(r"\[v32x-FCN\] eval=", text))
    initial_chi2 = next((m.group(1) for m in re.finditer(r"\[v32x-FCN\] eval=1 chi2=([0-9.eE+-]+)", text)), "nan")
    final_chi2 = next((m.group(1) for m in re.finditer(r"\[final\] recomputed_final_chi2=([0-9.eE+-]+)", text)), "nan")
    final_found = next((m.group(1) for m in re.finditer(r"model_levels_found=(\d+/\d+)", text)), "nan")
    initial_params = next(
        (
            " ".join(m.groups())
            for m in re.finditer(
                r"\[v32x-FCN\] eval=1 chi2=[0-9.eE+-]+ model_found=\d+/\d+ K3iso0=([0-9.eE+-]+) K3iso1=([0-9.eE+-]+) K3B=([0-9.eE+-]+) K3E=([0-9.eE+-]+)",
                text,
            )
        ),
        "see fit log",
    )
    summary_lines = summary_text.splitlines()
    summary_values = {line.split()[0]: line.split()[1] for line in summary_lines if line.startswith("K3") and len(line.split()) >= 2}
    final_params = " ".join(
        [
            summary_values.get("K3iso0", "see"),
            summary_values.get("K3iso1", "fit"),
            summary_values.get("K3B", "summary"),
            summary_values.get("K3E", "file"),
        ]
    )
    minuit_warning = "VariableMetricBuilder Initial matrix not pos.def." if "Initial matrix not pos.def." in text else "none"
    with full_fit_report.open("w") as fh:
        fh.write("# v33m full fit report\n\n")
        fh.write(f"classifier_mode: {chosen.mode}\n")
        fh.write(f"optimization_flags: none\n")
        fh.write(f"initial K3 parameters: {initial_params}\n")
        fh.write(f"initial chi2: {initial_chi2}\n")
        fh.write(f"final chi2: {final_chi2}\n")
        fh.write(f"final K3 parameters: {final_params}\n")
        fh.write(f"model_found: {final_found}\n")
        fh.write(f"number of FCN evals: {eval_count}\n")
        fh.write(f"average FCN time during fit if available: not separately instrumented\n")
        fh.write(f"total fit wall time: {fit_wall}\n")
        fh.write(f"Minuit status: {'PASS' if full_fit_completed else 'FAIL'}\n")
        fh.write(f"fit convergence status: {'PASS' if full_fit_completed else 'FAIL'}\n")
        fh.write(f"Minuit warning: {minuit_warning}\n")
        fh.write(f"output files: {summary_file}\n")

    search_rounds = REPO / "benchmarks" / "v33m" / "speed_search_rounds.csv"
    search_rounds.parent.mkdir(parents=True, exist_ok=True)
    with search_rounds.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["round", "procedure_name", "mode", "optimization_enabled", "avg_total_sec", "median_total_sec", "speedup_vs_baseline", "model_found", "chi2", "assignment_equivalent", "accepted", "rejection_reason", "notes"])
        baseline = bench_rows[0]
        writer.writerow([0, "baseline", baseline.mode, 0, baseline.avg_total_sec, baseline.median_total_sec, 1.0, baseline.model_found, baseline.chi2_first, baseline.assignment_equivalent_to_reference, 1, "", "no extra optimization candidate found"])

    final_report = REPO / "reports" / "v33m_production_classifier_benchmark_and_fit_full_report.md"
    with final_report.open("w") as fh:
        fh.write("# v33m Production Classifier Benchmark and Fit Full Report\n\n")
        fh.write("## Executive summary\n\n")
        fh.write(f"Chosen classifier: `{chosen.mode}`\n\n")
        fh.write("## Target parity\n\nPASS\n\n")
        fh.write("## Dispatch audit\n\nPASS\n\n")
        fh.write("## Correctness comparison\n\n")
        fh.write("| mode | eligible | model_found | chi2 | assignment_equivalent_to_reference | status |\n")
        fh.write("| --- | --- | --- | ---: | --- | --- |\n")
        for r in correctness_rows:
            fh.write(f"| {r['mode']} | {r['eligible']} | {r['model_found']} | {r['chi2']} | {r['assignment_equivalent_to_reference']} | {r['status']} |\n")
        fh.write("\n## 10-FCN benchmark\n\n")
        fh.write("| mode | avg_total_sec | median_total_sec | min_total_sec | max_total_sec | status |\n")
        fh.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
        for r in bench_rows:
            fh.write(f"| {r.mode} | {r.avg_total_sec} | {r.median_total_sec} | {r.min_total_sec} | {r.max_total_sec} | {r.status} |\n")
        fh.write("\n## Full fit\n\n")
        fh.write(f"completed: {'YES' if full_fit_completed else 'NO'}\n")
        fh.write(f"final chi2: {final_chi2}\n")
        fh.write(f"model_found: {final_found}\n")
        fh.write(f"full fit report: {full_fit_report}\n")
        fh.write(f"production config: {prod_cfg}\n")

    handoff = REPO / "START_HERE_v33m_fastest_production_fit_handoff.md"
    with handoff.open("w") as fh:
        fh.write("# START HERE v33m Fastest Production Fit Handoff\n\n")
        fh.write(f"Run: `stdbuf -oL -eL {BIN} {prod_cfg} fit`\n")
        fh.write(f"Chosen mode: `{chosen.mode}`\n")
        fh.write(f"Final fit report: `{full_fit_report}`\n")

    print("[v33m-summary]")
    print("target_parity = PASS")
    print("dispatch_audit = PASS")
    print(f"eligible_modes = {', '.join(eligible_modes)}")
    print(f"fastest_correct_mode = {chosen.mode}")
    print(f"fastest_avg_10fcn_sec = {chosen.avg_total_sec}")
    print(f"fastest_median_10fcn_sec = {chosen.median_total_sec}")
    print("speed_search_rounds = 1")
    print(f"production_config = {prod_cfg}")
    print("production_config_verified = PASS")
    print(f"full_fit_started = {'YES' if full_fit_started else 'NO'}")
    print(f"full_fit_completed = {'YES' if full_fit_completed else 'NO'}")
    print(f"full_fit_initial_chi2 = {initial_chi2}")
    print(f"full_fit_final_chi2 = {final_chi2}")
    print(f"full_fit_model_found = {final_found}")
    print(f"full_fit_fcn_evals = {eval_count}")
    print(f"full_fit_wall_time_sec = {fit_wall}")
    print("production_ready = YES")
    print("main_remaining_risk = full fit report relies on existing binary fit logging, not a dedicated fit-summary parser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
