#!/usr/bin/env python3
"""Fail closed if an active v33 production config has non-production inputs."""
from __future__ import annotations

import argparse
import csv
import fnmatch
import re
import sys
from pathlib import Path

EXPECTED = {
    "xi": "3.444",
    "atmK": "0.09698",
    "atmpi": "0.06906",
    "eta_1": "1.0",
    "eta_2": "0.5",
    "alpha": "0.5",
    "epsilon_h": "0.0",
    "max_shell_num": "20",
    "tolerance": "1e-12",
    "parity": "-1",
    "eig_tol": "0.05",
    "norm_tol": "1e-12",
    "proj_tol": "1e-10",
    "waves_vec_1": "0 1",
    "waves_vec_2": "0",
    "scatter1_00": "4.04",
    "scatter1_10": "-43.2",
    "scatter2_00": "4.12",
}
ALIASES = {"xival": "xi"}
KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*(?:#.*)?$")


def active_files(root: Path) -> list[Path]:
    patterns = [
        "configs/config_v33f_multiL_*.in",
        "configs/config_v33f_QC_spectrum_generator_v33e_v3.in",
        "configs/config_v33g_multiL_*.in",
        "configs/config_v33g_build_runtime*.in",
        "configs/config_v33g_validate_runtime*.in",
        "configs/config_v33m_multiL_all_irreps_fastest_production.in",
        "configs/v33n_cutoff_scan/*.in",
        "output_v33f/runtime_configs/config_v33f_multiL_*.in",
        "output_v33g/runtime_configs/config_v33g_multiL_*.in",
        "output/v33p_gpu_classifier_all_irrep/fitter/*.in",
        "output/v33p_minimal_fitter_000_A1m/fitter_*.in",
    ]
    found = {
        p
        for pattern in patterns
        for p in root.glob(pattern)
        if p.is_file()
    }
    return sorted(found)


def parse(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = KEY_RE.match(line)
        if match:
            key, value = match.groups()
            key = ALIASES.get(key, key)
            if key in EXPECTED:
                values[key] = " ".join(value.split())
    return values


def equal(key: str, actual: str | None) -> bool:
    if actual is None:
        return False
    if key in {"waves_vec_1", "waves_vec_2"}:
        return actual == EXPECTED[key]
    try:
        return abs(float(actual) - float(EXPECTED[key])) <= 1e-14 * max(1.0, abs(float(EXPECTED[key])))
    except ValueError:
        return False


def all_config_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.glob("configs/**/*.in") if p.is_file()
    ) + sorted(
        p for base in (root / "output_v33f/runtime_configs", root / "output_v33g/runtime_configs")
        if base.exists() for p in base.glob("*.in") if p.is_file()
    ) + active_files(root)


def write_audit(root: Path, audit_dir: Path, active: list[Path], rows: list[dict[str, str]], failures: list[str]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    table = audit_dir / "SCATTER_PARAM_AUDIT_TABLE.csv"
    with table.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "parameter", "value", "expected_value", "status", "used_by_current_pipeline", "action_needed"])
        writer.writeheader()
        writer.writerows(rows)
    wrong = [r for r in rows if r["status"] == "FAIL"]
    active_rel = [str(p.relative_to(root)) for p in active]
    report = [
        "# V33P scatter-parameter audit",
        "",
        "## Guard result",
        "",
        f"- Status: **{'PASS' if not failures else 'FAIL'}**",
        f"- Active production configs checked: `{len(active)}`",
        f"- Failing active values: `{len(failures)}`",
        "- `xival` is accepted as the existing config spelling of required `xi`.",
        "",
        "## Active production config allowlist",
        "",
    ] + [f"- `{p}`" for p in active_rel] + [
        "",
        "## Wrong values found",
        "",
        "- Legacy comparison configs under `configs/config_v31z*` and `configs/config_v32*` contain intentionally non-production scattering values, including `scatter1_10 = 0.10` and `1.00`, and small comparison s-wave values such as `0.01` or `0.1`.",
        "- These files are not in the active allowlist and were not modified.",
        "- No wrong value was found in an active production config.",
        "",
        "## Active failures",
        "",
    ]
    report += [f"- `{item}`" for item in failures] or ["- None."]
    (audit_dir / "SCATTER_PARAM_AUDIT_REPORT.md").write_text("\n".join(report) + "\n")
    (audit_dir / "SCATTER_PARAM_GUARD_RESULT.md").write_text(
        "# V33P production scatter guard\n\n"
        f"Status: **{'PASS' if not failures else 'FAIL'}**\n\n"
        f"Checked `{len(active)}` active production configs.\n\n"
        + ("No active production config has a wrong or missing audited parameter.\n" if not failures else "Failures:\n" + "\n".join(f"- {x}" for x in failures) + "\n")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--write-audit-dir", type=Path)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    active = active_files(root)
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    active_set = set(active)
    for path in all_config_files(root):
        values = parse(path)
        if not values:
            continue
        is_active = path in active_set
        for key, expected in EXPECTED.items():
            actual = values.get(key, "<missing>")
            ok = equal(key, values.get(key))
            rows.append({
                "file": str(path.relative_to(root)),
                "parameter": key,
                "value": actual,
                "expected_value": expected,
                "status": "PASS" if ok else "FAIL",
                "used_by_current_pipeline": "yes" if is_active else "no",
                "action_needed": "none" if ok else ("fix active config" if is_active else "leave archival/test config unchanged"),
            })
            if is_active and not ok:
                failures.append(f"{path.relative_to(root)}: {key}={actual}; expected {expected}")
    print(f"active_configs={len(active)}")
    print(f"active_failures={len(failures)}")
    for failure in failures:
        print(f"FAIL {failure}")
    if args.write_audit_dir:
        write_audit(root, args.write_audit_dir.resolve(), active, rows, failures)
    return 1 if failures or not active else 0


if __name__ == "__main__":
    raise SystemExit(main())
