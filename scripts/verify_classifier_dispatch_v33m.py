#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path


MODES = [
    "raw_sign_only",
    "raw_sign_clustered",
    "algo_v6_branch_count_final_select",
    "algo_v7_eigenbranch",
    "algo_v7_hybrid_det_eigenbranch",
    "digonto_v3_window",
    "digonto_v4_window",
]


@dataclass
class DispatchRow:
    requested_mode: str
    resolved_mode: str
    implementation_tag: str
    accepted_by_binary: str
    dispatch_unique: str
    status: str
    notes: str


def make_temp_config(base: Path, mode: str) -> Path:
    text = base.read_text()
    text = re.sub(r"^classifier_mode = .*$", f"classifier_mode = {mode}", text, flags=re.M)
    fd, name = tempfile.mkstemp(prefix="v33m_dispatch_", suffix=".in")
    os.close(fd)
    tmp = Path(name)
    tmp.write_text(text)
    return tmp


def run_mode(repo: Path, mode: str) -> DispatchRow:
    cfg = make_temp_config(repo / "configs" / "config_v33g_multiL_all_irreps_runtime.in", mode)
    log = subprocess.run(
        [
            str(repo / "bin" / "v33f_k3df_fitter_multiL_v33e"),
            str(cfg),
            "benchmark-fcn",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    output = log.stdout + "\n" + log.stderr
    m = re.search(
        r"\[classifier-dispatch\] requested_mode=(\S+) resolved_mode=(\S+) implementation=(\S+) version_tag=(\S+) dispatch_unique=(\d+)",
        output,
    )
    if not m:
        return DispatchRow(mode, "", "", "NO", "NO", "FAIL", "dispatch line missing")
    requested, resolved, impl, version_tag, unique = m.groups()
    accepted = "YES" if log.returncode == 0 else "NO"
    notes = "ok"
    if requested != resolved:
        notes = f"alias_to_{resolved}"
    status = "PASS" if accepted == "YES" and requested == mode else "FAIL"
    return DispatchRow(
        requested_mode=requested,
        resolved_mode=resolved,
        implementation_tag=impl,
        accepted_by_binary=accepted,
        dispatch_unique="YES" if unique == "1" else "NO",
        status=status,
        notes=notes,
    )


def write_outputs(repo: Path, rows: list[DispatchRow]) -> None:
    diag = repo / "diagnostics" / "v33m"
    rep = repo / "reports"
    diag.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    csv_path = diag / "classifier_dispatch_audit.csv"
    md_path = rep / "v33m_classifier_dispatch_audit.md"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "requested_mode",
                "resolved_mode",
                "implementation_tag",
                "accepted_by_binary",
                "dispatch_unique",
                "status",
                "notes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.requested_mode,
                    row.resolved_mode,
                    row.implementation_tag,
                    row.accepted_by_binary,
                    row.dispatch_unique,
                    row.status,
                    row.notes,
                ]
            )
    with md_path.open("w") as fh:
        fh.write("# v33m classifier dispatch audit\n\n")
        fh.write("| requested_mode | resolved_mode | implementation_tag | accepted_by_binary | dispatch_unique | status | notes |\n")
        fh.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            fh.write(
                f"| {row.requested_mode} | {row.resolved_mode} | {row.implementation_tag} | "
                f"{row.accepted_by_binary} | {row.dispatch_unique} | {row.status} | {row.notes} |\n"
            )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    rows = [run_mode(repo, mode) for mode in MODES]
    write_outputs(repo, rows)
    ok = all(r.status == "PASS" for r in rows)
    print(f"dispatch_audit={'PASS' if ok else 'FAIL'}")
    for r in rows:
        print(f"{r.requested_mode}: {r.resolved_mode} {r.implementation_tag} {r.status}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
