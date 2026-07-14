#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {
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


def write_report(path: Path, rows: list[dict[str, object]], total: int, ok: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# v33k target loader parity\n\n")
        fh.write(f"status: {'PASS' if ok else 'FAIL'}\n")
        fh.write(f"expected_total: {sum(EXPECTED.values())}\n")
        fh.write(f"observed_total: {total}\n\n")
        fh.write("| Lbyas | irrep | expected | observed | status |\n")
        fh.write("| --- | --- | ---: | ---: | --- |\n")
        for row in rows:
            fh.write(f"| {row['Lbyas']} | {row['irrep']} | {row['expected']} | {row['observed']} | {row['status']} |\n")
        fh.write("\n## Alias check\n\n")
        alias_rows = [r for r in rows if r["aliases"]]
        if alias_rows:
            fh.write("| Lbyas | irrep | aliases |\n| --- | --- | --- |\n")
            for row in alias_rows:
                fh.write(f"| {row['Lbyas']} | {row['irrep']} | {', '.join(row['aliases'])} |\n")
        else:
            fh.write("No alias differences were present in the dumped target list.\n")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / "configs" / "config_v33g_multiL_all_irreps_runtime.in"
    bin_path = repo / "bin" / "v33f_k3df_fitter_multiL_v33e"
    if not bin_path.exists():
        print(f"missing binary: {bin_path}", file=sys.stderr)
        return 2

    subprocess.run(
        [str(bin_path), str(cfg), "dump-targets"],
        cwd=repo,
        check=True,
    )

    json_path = repo / "diagnostics" / "v33k" / "fitter_targets" / "fitter_targets.json"
    if not json_path.exists():
        print(f"missing dump output: {json_path}", file=sys.stderr)
        return 2

    data = json.loads(json_path.read_text())
    counts: Counter[tuple[float, str]] = Counter()
    aliases: dict[tuple[float, str], list[str]] = defaultdict(list)
    for row in data:
        key = (float(row["Lbyas"]), str(row["canonical_momentum_label"]))
        counts[key] += 1
        if row["momentum_label"] != row["canonical_momentum_label"]:
            aliases[key].append(str(row["momentum_label"]))

    rows: list[dict[str, object]] = []
    ok = True
    for key, expected in sorted(EXPECTED.items()):
        observed = counts.get(key, 0)
        status = "PASS" if observed == expected else "FAIL"
        if status != "PASS":
            ok = False
        rows.append(
            {
                "Lbyas": key[0],
                "irrep": key[1],
                "expected": expected,
                "observed": observed,
                "status": status,
                "aliases": sorted(set(aliases.get(key, []))),
            }
        )

    total = len(data)
    if total != sum(EXPECTED.values()):
        ok = False

    report_path = repo / "reports" / "v33k_target_loader_parity.md"
    write_report(report_path, rows, total, ok)

    print(f"expected_total={sum(EXPECTED.values())} observed_total={total}")
    for row in rows:
        print(f"L{int(row['Lbyas'])} {row['irrep']}: {row['observed']} / {row['expected']} {row['status']}")
    print(f"report={report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
