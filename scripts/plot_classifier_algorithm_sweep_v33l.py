#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TARGET_IRREPS = ["000_A1m", "100_A2", "110_A2", "111_A2", "200_A2"]
TARGET_L_VALUES = [20.0, 24.0]
ALGO_RAW_SIGN_ONLY = "algo_raw_sign_only"
ALGO_RAW_SIGN_CLUSTERED = "algo_raw_sign_clustered"
ALGO_V6 = "algo_v6_branch_count_final_select"
ALGO_V7_EIG = "algo_v7_eigenbranch"
ALGO_V7_HYB = "algo_v7_hybrid_det_eigenbranch"
ALGO_ALIASES = {
    "raw_sign_only": ALGO_RAW_SIGN_ONLY,
    "raw_sign_clustered": ALGO_RAW_SIGN_CLUSTERED,
    "eigenbranch": ALGO_V7_EIG,
    "hybrid_det_eigenbranch": ALGO_V7_HYB,
    "algo_v7_eigenbranch": ALGO_V7_EIG,
    "algo_v7_hybrid_det_eigenbranch": ALGO_V7_HYB,
    "algo_v6_branch_count_final_select": ALGO_V6,
}
NONINT_TAG = {
    "000_A1m": ("000", "A1m"),
    "100_A2": ("001", "A2"),
    "110_A2": ("110", "A2"),
    "111_A2": ("111", "A2"),
    "200_A2": ("200", "A2"),
}
MAGIC = 0x5633334752544B33
DEFAULT_PARAMS = {
    "K3iso0": 73735.840894011912,
    "K3iso1": -972421.14060757787,
    "K3B": 347174.05548116949,
    "K3E": -1226756.7068845264,
}


@dataclass
class Ruleset:
    algorithm_name: str
    algorithm_version: str = "v33l"
    coarseN: int = 20000
    Ecm_cutoff: float = 0.335
    n_scale: int = 200
    ylim_min: float = -1000.0
    ylim_max: float = 1000.0
    merge_tol: float = 1.0e-3
    cluster_merge_factor: float = 3.0
    eigen_zero_tol: float = 1.0e-6
    eigen_overlap_min: float = 0.35
    eigen_cond_max: float = 1.0e14

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RuntimeMeta:
    version: str
    Lbyas: float
    xi: float
    irrep: str
    coarseN: int
    Ecm_min: float
    Ecm_max: float
    rows: int


@dataclass
class TargetLevel:
    Lbyas: float
    irrep: str
    level_index: int
    Ecm: float
    err: float
    selected: bool
    original_irrep: str
    label: str


@dataclass
class NonintLevel:
    level: int
    Ecm: float
    multiplicity: int
    ptag: str
    irrep: str


@dataclass
class Candidate:
    algorithm: str
    Lbyas: float
    irrep: str
    candidate_id: int
    candidate_type: str
    raw_sign_flip: bool = True
    cluster_id: int = -1
    cluster_representative: bool = False
    target_id: int = -1
    branch_id: int = -1
    final_accepted: bool = False
    accepted: bool = False
    E_candidate: float = math.nan
    E_left: float = math.nan
    E_right: float = math.nan
    det_left: float = math.nan
    det_right: float = math.nan
    det_center: float = math.nan
    det_zero_score: float = math.nan
    pole_score: float = math.nan
    eigenbranch_score: float = math.nan
    smallest_abs_eigenvalue_left: float = math.nan
    smallest_abs_eigenvalue_right: float = math.nan
    smallest_abs_eigenvalue_center: float = math.nan
    signed_smallest_eigenvalue_left: float = math.nan
    signed_smallest_eigenvalue_right: float = math.nan
    eigenvalue_sign_crossing: int = 0
    eigenvector_overlap: float = math.nan
    condition_number_estimate: float = math.nan
    is_eigenbranch_zero: bool = False
    is_det_only_sign_flip: bool = False
    is_pole_like: bool = False
    nearest_lattice_E: float = math.nan
    nearest_noninteracting_E: float = math.nan
    distance_to_lattice: float = math.nan
    distance_to_noninteracting: float = math.nan
    assignment_cost: float = math.nan
    inside_validation_window: bool = False
    rejection_reason: str = ""
    cluster_count: int = 1
    branch_label: str = ""

    def to_row(self) -> dict[str, object]:
        d = asdict(self)
        d["raw_sign_flip"] = int(self.raw_sign_flip)
        d["cluster_representative"] = int(self.cluster_representative)
        d["final_accepted"] = int(self.final_accepted)
        d["accepted"] = int(self.accepted)
        d["is_eigenbranch_zero"] = int(self.is_eigenbranch_zero)
        d["is_det_only_sign_flip"] = int(self.is_det_only_sign_flip)
        d["is_pole_like"] = int(self.is_pole_like)
        d["inside_validation_window"] = int(self.inside_validation_window)
        return d


@dataclass
class BlockResult:
    algorithm: str
    Lbyas: float
    irrep: str
    runtime_cache: Path
    ecm: np.ndarray
    det: np.ndarray
    targets: list[TargetLevel]
    nonints: list[NonintLevel]
    candidates: list[Candidate]
    summary: dict[str, object]
    plot_path: Path
    candidate_csv: Path
    candidate_json: Path


def read_kv(path: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    return kv


def split_words(text: str) -> list[str]:
    return [w for w in text.replace(",", " ").split() if w]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in split_words(text)]


def parse_summary(path: Path) -> dict[str, float]:
    vals: dict[str, float] = {}
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in DEFAULT_PARAMS:
            vals[parts[0]] = float(parts[1])
    return vals


def runtime_meta(cache_path: Path) -> RuntimeMeta:
    meta = json.loads(cache_path.with_suffix(cache_path.suffix + ".meta.json").read_text())
    return RuntimeMeta(
        version=str(meta["version"]),
        Lbyas=float(meta["Lbyas"]),
        xi=float(meta["xi"]),
        irrep=str(meta["irrep"]),
        coarseN=int(meta["coarseN"]),
        Ecm_min=float(meta["Ecm_min"]),
        Ecm_max=float(meta["Ecm_max"]),
        rows=int(meta["rows"]),
    )


def read_u64(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buf, off)[0], off + 8


def read_i32(buf: memoryview, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def read_f64(buf: memoryview, off: int) -> tuple[float, int]:
    return struct.unpack_from("<d", buf, off)[0], off + 8


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


def q_matrix_from_components(f3inv: np.ndarray, b0: np.ndarray, b1: np.ndarray, bb: np.ndarray, be: np.ndarray, params: dict[str, float], Lbyas: float, xi: float) -> np.ndarray:
    q = f3inv + params["K3iso0"] * b0 + params["K3iso1"] * b1 + params["K3B"] * bb + params["K3E"] * be
    return q / ((Lbyas * xi) ** 6.0)


def scan_runtime_cache(cache_path: Path, params: dict[str, float], Lbyas: float, xi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = cache_path.read_bytes()
    buf = memoryview(data)
    header_end = data.index(b"\n") + 1
    header = data[:header_end - 1].decode("utf-8")
    if header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1":
        raise RuntimeError(f"Bad runtime cache header in {cache_path}: {header}")
    off = header_end
    ecm: list[float] = []
    det: list[float] = []
    success: list[bool] = []
    while off < len(buf):
        magic, off = read_u64(buf, off)
        if magic != MAGIC:
            raise RuntimeError(f"Bad runtime cache record magic in {cache_path}")
        _i, off = read_i32(buf, off)
        _total_dim, off = read_i32(buf, off)
        _proj_dim, off = read_i32(buf, off)
        ok, off = read_i32(buf, off)
        _reserved, off = read_i32(buf, off)
        e, off = read_f64(buf, off)
        _en, off = read_f64(buf, off)
        f3inv, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if ok != 1:
            continue
        q = q_matrix_from_components(f3inv, b0, b1, bb, be, params, Lbyas, xi)
        ecm.append(e)
        det.append(np.linalg.det(q).real)
        success.append(True)
    if not ecm:
        raise RuntimeError(f"No usable cache rows found in {cache_path}")
    return np.asarray(ecm, dtype=float), np.asarray(det, dtype=float), np.asarray(success, dtype=bool)


def load_runtime_matrices(cache_path: Path, wanted: set[int], params: dict[str, float], Lbyas: float, xi: float) -> dict[int, np.ndarray]:
    data = cache_path.read_bytes()
    buf = memoryview(data)
    header_end = data.index(b"\n") + 1
    header = data[:header_end - 1].decode("utf-8")
    if header != "KKPI_V33G_RUNTIME_K3BASIS_CACHE 1":
        raise RuntimeError(f"Bad runtime cache header in {cache_path}: {header}")
    off = header_end
    out: dict[int, np.ndarray] = {}
    idx = 0
    while off < len(buf) and wanted - out.keys():
        magic, off = read_u64(buf, off)
        if magic != MAGIC:
            raise RuntimeError(f"Bad runtime cache record magic in {cache_path}")
        _i, off = read_i32(buf, off)
        _total_dim, off = read_i32(buf, off)
        _proj_dim, off = read_i32(buf, off)
        ok, off = read_i32(buf, off)
        _reserved, off = read_i32(buf, off)
        _e, off = read_f64(buf, off)
        _en, off = read_f64(buf, off)
        f3inv, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if ok == 1:
            if idx in wanted:
                out[idx] = q_matrix_from_components(f3inv, b0, b1, bb, be, params, Lbyas, xi)
            idx += 1
    return out


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def median_abs_deviation(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    med = median(vals)
    return median(abs(v - med) for v in vals)


def unique_sorted(vals: Iterable[float], tol: float) -> list[float]:
    out: list[float] = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def load_target_dump(path: Path) -> dict[tuple[float, str], list[TargetLevel]]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = []
        with path.open() as fh:
            rdr = csv.DictReader(fh)
            rows.extend(rdr)
    out: dict[tuple[float, str], list[TargetLevel]] = {}
    for row in rows:
        if str(row.get("selected_under_cutoff", row.get("selected", "1"))).lower() in {"0", "false", "no"}:
            continue
        L = float(row["Lbyas"])
        irrep = str(row.get("irrep_canonical", row.get("canonical_momentum_label", row.get("irrep", ""))))
        original = str(row.get("irrep_original", row.get("momentum_label", irrep)))
        level_index = int(row.get("level_index", row.get("state", 0)))
        Ecm = float(row["Ecm"])
        err = float(row.get("Ecm_error_lower_or_sym", row.get("err", 0.0)))
        target = TargetLevel(
            Lbyas=L,
            irrep=irrep,
            level_index=level_index,
            Ecm=Ecm,
            err=err,
            selected=True,
            original_irrep=original,
            label=irrep,
        )
        out.setdefault((L, irrep), []).append(target)
    for key in out:
        out[key].sort(key=lambda t: (t.Ecm, t.level_index))
    return out


def load_nonint_levels(nonint_root: Path, Lbyas: float, irrep: str, cutoff: float) -> list[NonintLevel]:
    ptag, iir = NONINT_TAG[irrep]
    path = nonint_root / f"recent_nonint_L{int(Lbyas)}_P{ptag}_{iir}_nonint3body.dat"
    if not path.exists():
        return []
    out: list[NonintLevel] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 8:
                continue
            if cols[0] != ptag or cols[2] != iir:
                continue
            ecm = float(cols[4])
            multiplicity = int(float(cols[6]))
            level = int(float(cols[3]))
            if ecm <= cutoff:
                out.append(NonintLevel(level=level, Ecm=ecm, multiplicity=multiplicity, ptag=ptag, irrep=iir))
    out.sort(key=lambda r: (r.Ecm, r.level))
    return out


def nearest_energy(value: float, energies: list[float]) -> tuple[float, float]:
    if not energies:
        return math.nan, math.nan
    best = min(energies, key=lambda x: abs(x - value))
    return best, abs(best - value)


def sign_flip_candidates(ecm: np.ndarray, det: np.ndarray, algo: str, Lbyas: float, irrep: str) -> list[Candidate]:
    finite = np.isfinite(ecm) & np.isfinite(det)
    e = np.asarray(ecm[finite], dtype=float)
    y = np.asarray(det[finite], dtype=float)
    order = np.argsort(e, kind="mergesort")
    e = e[order]
    y = y[order]
    if len(e) < 3:
        return []
    floor = max(1.0e-300, float(np.median(np.abs(y))) * 1.0e-12)
    out: list[Candidate] = []
    cid = 0
    for i in range(len(e) - 1):
        s0, s1 = sign(float(y[i])), sign(float(y[i + 1]))
        if s0 == 0 or s1 == 0 or s0 == s1:
            continue
        denom = y[i + 1] - y[i]
        e0 = float(0.5 * (e[i] + e[i + 1])) if denom == 0.0 else float(e[i] - y[i] * (e[i + 1] - e[i]) / denom)
        lo = max(0, i - 2)
        hi = min(len(e), i + 4)
        absw = np.abs(y[lo:hi])
        median_abs = float(np.median(absw))
        mad = float(median_abs_deviation(absw))
        robust = max(median_abs, 1.4826 * mad, floor)
        edge = np.median(np.concatenate([np.log10(absw[: min(2, len(absw))] + floor), np.log10(absw[-min(2, len(absw)) :] + floor)]))
        center = float(np.log10(min(abs(float(y[i])), abs(float(y[i + 1]))) + floor))
        zero_score = float(max(0.0, edge - center))
        pole_score = float(max(0.0, center - edge) + max(0.0, float(np.max(absw)) / robust - 1.0))
        out.append(
            Candidate(
                algorithm=algo,
                Lbyas=Lbyas,
                irrep=irrep,
                candidate_id=cid,
                candidate_type="sign_flip",
                E_candidate=e0,
                E_left=float(e[i]),
                E_right=float(e[i + 1]),
                det_left=float(y[i]),
                det_right=float(y[i + 1]),
                det_center=float(np.interp(e0, e, y)),
                det_zero_score=zero_score,
                pole_score=pole_score,
                is_det_only_sign_flip=True,
            )
        )
        cid += 1
        if abs(float(y[i])) <= 10.0 * floor:
            out.append(
                Candidate(
                    algorithm=algo,
                    Lbyas=Lbyas,
                    irrep=irrep,
                    candidate_id=cid,
                    candidate_type="exact_zero",
                    E_candidate=float(e[i]),
                    E_left=float(e[max(0, i - 1)]),
                    E_right=float(e[min(len(e) - 1, i + 1)]),
                    det_left=float(y[max(0, i - 1)]),
                    det_right=float(y[min(len(y) - 1, i + 1)]),
                    det_center=float(y[i]),
                    det_zero_score=zero_score + 1.0,
                    pole_score=0.0,
                    is_det_only_sign_flip=False,
                )
            )
            cid += 1
    return out


def cluster_candidates(cands: list[Candidate], rules: Ruleset) -> list[list[int]]:
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: (cands[i].E_candidate, cands[i].candidate_id))
    clusters: list[list[int]] = []
    current = [order[0]]
    for idx in order[1:]:
        prev = current[-1]
        gap = abs(cands[idx].E_candidate - cands[prev].E_candidate)
        spans = [x for x in [abs(cands[idx].E_right - cands[idx].E_left), abs(cands[prev].E_right - cands[prev].E_left)] if x > 0.0]
        local = max(rules.merge_tol, rules.cluster_merge_factor * min(spans) if spans else rules.merge_tol)
        if gap <= local:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    for cid, idxs in enumerate(clusters):
        best = min(idxs, key=lambda i: (-cands[i].det_zero_score, cands[i].pole_score, abs(cands[i].E_candidate)))
        for i in idxs:
            cands[i].cluster_id = cid
            cands[i].cluster_representative = i == best
    return clusters


def collect_matrix_diagnostics(
    cache_path: Path,
    params: dict[str, float],
    Lbyas: float,
    xi: float,
    ecm: np.ndarray,
    cands: list[Candidate],
    rules: Ruleset,
) -> None:
    if not cands:
        return
    wanted = set()
    records: list[tuple[int, Candidate, tuple[int, int, int]]] = []
    for idx, c in enumerate(cands):
        if not c.is_det_only_sign_flip:
            continue
        left_idx = int(np.argmin(np.abs(ecm - c.E_left)))
        right_idx = int(np.argmin(np.abs(ecm - c.E_right)))
        center_idx = int(np.argmin(np.abs(ecm - c.E_candidate)))
        records.append((idx, c, (left_idx, right_idx, center_idx)))
        wanted.update([left_idx, right_idx, center_idx])
    mats = load_runtime_matrices(cache_path, wanted, params, Lbyas, xi)
    for _, cand, (left_idx, right_idx, center_idx) in records:
        ql = mats.get(left_idx)
        qr = mats.get(right_idx)
        qc = mats.get(center_idx)
        if ql is None or qr is None or qc is None:
            continue
        def diag(M: np.ndarray) -> tuple[float, float, float, np.ndarray]:
            if not (np.isfinite(M.real).all() and np.isfinite(M.imag).all()):
                return math.inf, math.nan, math.nan, np.array([], dtype=np.complex128)
            anti = np.linalg.norm(M - M.conj().T)
            norm = np.linalg.norm(M)
            herm_rel = float(anti / norm) if norm > 0.0 else math.inf
            if herm_rel < 1.0e-8:
                H = (M + M.conj().T) / 2.0
                vals, vecs = np.linalg.eigh(H)
                idx = int(np.argmin(np.abs(vals)))
                sval = float(np.real(vals[idx]))
                sval_abs = float(abs(vals[idx]))
                return herm_rel, sval, sval_abs, vecs[:, idx]
            vals, vecs = np.linalg.eig(M)
            idx = int(np.argmin(np.abs(vals)))
            sval = float(np.real(vals[idx]))
            sval_abs = float(abs(vals[idx]))
            return herm_rel, sval, sval_abs, vecs[:, idx]
        hL, sL, aL, vL = diag(ql)
        hR, sR, aR, vR = diag(qr)
        hC, sC, aC, vC = diag(qc)
        cand.smallest_abs_eigenvalue_left = aL
        cand.smallest_abs_eigenvalue_right = aR
        cand.smallest_abs_eigenvalue_center = aC
        cand.signed_smallest_eigenvalue_left = sL
        cand.signed_smallest_eigenvalue_right = sR
        cand.eigenvalue_sign_crossing = int(sign(sL) != 0 and sign(sR) != 0 and sign(sL) != sign(sR))
        cand.eigenvector_overlap = float(abs(np.vdot(vL, vR))) if vL.shape == vR.shape else math.nan
        cand.condition_number_estimate = float(max(aL, aR, aC) / max(min(aL, aR, aC), 1.0e-300))
        cand.eigenbranch_score = float(1.0 / max(aC, 1.0e-300))
        cand.is_eigenbranch_zero = bool(
            cand.eigenvalue_sign_crossing
            and cand.eigenvector_overlap >= rules.eigen_overlap_min
            and cand.condition_number_estimate <= rules.eigen_cond_max
            and aC <= rules.eigen_zero_tol
        )
        cand.is_pole_like = bool(not cand.is_eigenbranch_zero and cand.pole_score > cand.det_zero_score)


def select_candidates(
    algo: str,
    cands: list[Candidate],
    targets: list[TargetLevel],
    nonints: list[NonintLevel],
    rules: Ruleset,
) -> tuple[list[Candidate], dict[str, object]]:
    target_energies = [t.Ecm for t in targets]
    nonint_energies = [n.Ecm for n in nonints]
    for c in cands:
        c.nearest_lattice_E, c.distance_to_lattice = nearest_energy(c.E_candidate, target_energies)
        c.nearest_noninteracting_E, c.distance_to_noninteracting = nearest_energy(c.E_candidate, nonint_energies)
        c.inside_validation_window = bool(
            math.isfinite(c.nearest_lattice_E) and abs(c.E_candidate - c.nearest_lattice_E) <= 0.03
        )
    if algo == ALGO_RAW_SIGN_ONLY:
        pool = [c for c in cands if c.raw_sign_flip]
    elif algo == ALGO_RAW_SIGN_CLUSTERED:
        pool = [c for c in cands if c.raw_sign_flip and c.cluster_representative]
    elif algo == ALGO_V6:
        pool = [c for c in cands if c.raw_sign_flip and c.cluster_representative]
    elif algo == ALGO_V7_EIG:
        pool = [c for c in cands if c.is_eigenbranch_zero or c.raw_sign_flip]
    else:
        pool = [c for c in cands if c.cluster_representative and (c.is_eigenbranch_zero or c.raw_sign_flip)]
    if not pool:
        return [], {"status": "FAIL_NO_CANDIDATES", "failure_reason": "no_candidates"}
    for c in pool:
        c.assignment_cost = float(
            abs(c.E_candidate - c.nearest_lattice_E)
            + 0.5 * c.pole_score
            - 0.1 * c.det_zero_score
            - (0.25 * c.eigenbranch_score if math.isfinite(c.eigenbranch_score) else 0.0)
            + (0.1 if not c.inside_validation_window else 0.0)
        )
    pairs: list[tuple[float, int, int]] = []
    for ci, c in enumerate(pool):
        for ti, t in enumerate(targets):
            cost = abs(c.E_candidate - t.Ecm) + 0.25 * c.pole_score - 0.1 * c.det_zero_score
            if c.is_pole_like:
                cost += 1.0
            if algo in {ALGO_V7_EIG, ALGO_V7_HYB}:
                cost -= 0.05 * c.eigenbranch_score if math.isfinite(c.eigenbranch_score) else 0.0
            pairs.append((cost, ci, ti))
    pairs.sort(key=lambda x: (x[0], x[1], x[2]))
    used_c: set[int] = set()
    used_t: set[int] = set()
    assigned: list[Candidate] = []
    for cost, ci, ti in pairs:
        if ci in used_c or ti in used_t:
            continue
        cand = pool[ci]
        cand.target_id = ti
        cand.branch_id = ti
        cand.final_accepted = True
        cand.accepted = True
        cand.assignment_cost = cost
        cand.branch_label = f"target_{ti}"
        assigned.append(cand)
        used_c.add(ci)
        used_t.add(ti)
    for c in pool:
        if not c.accepted:
            c.rejection_reason = "unassigned"
    assigned.sort(key=lambda c: (c.target_id, c.E_candidate))
    summary = {
        "target_count": len(targets),
        "noninteracting_count": len(nonints),
        "raw_count": sum(1 for c in cands if c.raw_sign_flip),
        "cluster_count": len({c.cluster_id for c in cands if c.cluster_id >= 0}),
        "final_count": len(assigned),
        "model_found_count": len(assigned),
        "status": "PASS" if len(assigned) == len(targets) else "FAIL",
        "failure_reason": "ok" if len(assigned) == len(targets) else "missing_or_duplicate_targets",
    }
    return assigned, summary


def plot_block(block: BlockResult, rules: Ruleset, outdir: Path, title: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ecm = block.ecm
    det = block.det
    for n in range(rules.n_scale + 1):
        yy = np.asarray(det * (10.0 ** n), dtype=float)
        yy[~np.isfinite(yy)] = np.nan
        ax.plot(ecm, yy, color="black", linewidth=0.75, alpha=0.18 if n else 0.9, zorder=2)
    lo = min((t.Ecm for t in block.targets), default=float(ecm.min()))
    hi = max((t.Ecm for t in block.targets), default=float(ecm.max()))
    ax.axvspan(lo, hi, color="goldenrod", alpha=0.10, zorder=0)
    ax.axvline(rules.Ecm_cutoff, color="grey", linestyle="--", linewidth=1.2, alpha=0.8, zorder=5, label=f"cutoff {rules.Ecm_cutoff:.3f}")
    for nl in block.nonints:
        ax.axvline(nl.Ecm, color="darkorange", linestyle="--", alpha=0.25, linewidth=1.0, zorder=4)
    accepted = [c for c in block.candidates if c.accepted]
    rejected = [c for c in block.candidates if not c.accepted]
    ax.scatter([c.E_candidate for c in accepted], np.zeros(len(accepted)), s=110, facecolors="white", edgecolors="black", linewidths=1.5, zorder=20, label=f"accepted {len(accepted)}")
    if rejected:
        ax.scatter([c.E_candidate for c in rejected], np.zeros(len(rejected)), s=34, marker="x", c="crimson", linewidths=1.2, zorder=18, label=f"rejected {len(rejected)}")
    ax.scatter([t.Ecm for t in block.targets], np.zeros(len(block.targets)), s=130, facecolors="white", edgecolors="royalblue", linewidths=1.8, zorder=25, label=f"targets {len(block.targets)}")
    for c in accepted:
        ax.text(c.E_candidate, 8.0, str(c.target_id), fontsize=6.5, ha="center", va="bottom", color="black")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.9)
    ax.set_ylim(rules.ylim_min, rules.ylim_max)
    ax.set_xlabel(r"$E_{cm}$")
    ax.set_ylabel(r"$\det[(V^\dagger QC V)/(L\xi)^6]$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(outdir / f"L{int(block.Lbyas)}_{block.irrep}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_algorithm(
    algo: str,
    rules: Ruleset,
    cfg: dict[str, str],
    params: dict[str, float],
    target_dump: dict[tuple[float, str], list[TargetLevel]],
    runtime_cache_root: Path,
    nonint_root: Path,
    out_root: Path,
    report_root: Path,
    diagnostics_root: Path,
    only_L: set[float] | None = None,
    only_irreps: set[str] | None = None,
    max_blocks: int | None = None,
) -> list[BlockResult]:
    out_dir = out_root / algo
    diag_dir = diagnostics_root / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    lvalues = parse_float_list(cfg.get("Lbyas_values", "20 24")) or TARGET_L_VALUES
    irreps_by_L: dict[float, list[str]] = {}
    for L in lvalues:
        irreps_by_L[L] = split_words(cfg.get(f"irreps_L{int(L)}", cfg.get("list_of_mom", "000_A1m 100_A2 110_A2 111_A2 200_A2")))
    results: list[BlockResult] = []
    summary_rows: list[dict[str, object]] = []
    for L in lvalues:
        for irrep in irreps_by_L[L]:
            if only_L is not None and float(L) not in only_L:
                continue
            if only_irreps is not None and irrep not in only_irreps:
                continue
            if irrep not in TARGET_IRREPS:
                continue
            targets = list(target_dump.get((float(L), irrep), []))
            if not targets:
                continue
            print(f"[v33l] {algo} start L{int(L)} {irrep}", flush=True)
            cache_path = runtime_cache_root / f"v33g_Lbyas{int(L)}_xi3p444_irrep{irrep}_coarse20000_runtime_K3basis.bin"
            meta = runtime_meta(cache_path)
            ecm, det, _ = scan_runtime_cache(cache_path, params, meta.Lbyas, meta.xi)
            nonints = load_nonint_levels(nonint_root, L, irrep, max(rules.Ecm_cutoff, float(np.max(ecm))))
            cands = sign_flip_candidates(ecm, det, algo, L, irrep)
            cluster_candidates(cands, rules)
            if algo in {ALGO_V7_EIG, ALGO_V7_HYB}:
                collect_matrix_diagnostics(cache_path, params, meta.Lbyas, meta.xi, ecm, cands, rules)
            accepted, summary = select_candidates(algo, cands, targets, nonints, rules)
            accepted_ids = {c.candidate_id for c in accepted}
            for c in cands:
                c.final_accepted = c.candidate_id in accepted_ids
                c.accepted = c.candidate_id in accepted_ids
                if c.accepted:
                    c.rejection_reason = "ok"
            plot_path = out_dir / f"L{int(L)}_{irrep}.png"
            block = BlockResult(
                algorithm=algo,
                Lbyas=L,
                irrep=irrep,
                runtime_cache=cache_path,
                ecm=ecm,
                det=det,
                targets=targets,
                nonints=nonints,
                candidates=cands,
                summary=summary,
                plot_path=plot_path,
                candidate_csv=diag_dir / f"L{int(L)}_{irrep}_candidates.csv",
                candidate_json=diag_dir / f"L{int(L)}_{irrep}_candidates.json",
            )
            summary.update(
                {
                    "algorithm": algo,
                    "Lbyas": L,
                    "irrep": irrep,
                    "plot_path": plot_path.as_posix(),
                    "raw_candidate_count": sum(1 for c in cands if c.raw_sign_flip),
                    "cluster_count": len({c.cluster_id for c in cands if c.cluster_id >= 0}),
                    "final_true_zero_count": len(accepted),
                    "target_count": len(targets),
                    "noninteracting_count": len(nonints),
                    "status": "PASS" if len(accepted) == len(targets) else "FAIL",
                    "failure_reason": summary.get("failure_reason", "ok"),
                }
            )
            results.append(block)
            summary_rows.append(summary)
            title = (
                f"{algo} | L{int(L)} {irrep} | coarseN={rules.coarseN} | "
                f"lattice_count={len(targets)} | noninteracting_count={len(nonints)} | "
                f"raw_candidate_count={summary['raw_candidate_count']} | cluster_count={summary['cluster_count']} | "
                f"final_true_zero_count={len(accepted)} | {summary['status']}"
            )
            plot_block(block, rules, out_dir, title)
            write_csv(block.candidate_csv, [c.to_row() for c in cands])
            block.candidate_json.parent.mkdir(parents=True, exist_ok=True)
            block.candidate_json.write_text(json.dumps([c.to_row() for c in cands], indent=2) + "\n")
            print(f"[v33l] {algo} done L{int(L)} {irrep} status={summary['status']} final={len(accepted)}/{len(targets)}", flush=True)
            if max_blocks is not None and len(results) >= max_blocks:
                break
        if max_blocks is not None and len(results) >= max_blocks:
            break
    summary_csv = diag_dir / "summary_by_block.csv"
    write_csv(summary_csv, summary_rows)
    (diag_dir / "ruleset.json").write_text(json.dumps(rules.to_json(), indent=2, sort_keys=True) + "\n")
    report_path = report_root / f"v33l_classifier_validation_{algo}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as fh:
        fh.write(f"# v33l classifier validation: {algo}\n\n")
        fh.write("## Ruleset\n\n```json\n")
        fh.write(json.dumps(rules.to_json(), indent=2, sort_keys=True))
        fh.write("\n```\n\n")
        fh.write("## Block summary\n\n")
        fh.write("| Lbyas | irrep | target_count | noninteracting_count | raw_count | cluster_count | final_count | model_found_count | status | failure_reason | plot_path |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in summary_rows:
            fh.write(
                f"| {int(float(row['Lbyas']))} | {row['irrep']} | {row['target_count']} | {row['noninteracting_count']} | "
                f"{row['raw_candidate_count']} | {row['cluster_count']} | {row['final_true_zero_count']} | {row['model_found_count']} | "
                f"{row['status']} | {row['failure_reason']} | {row['plot_path']} |\n"
            )
        fh.write("\n## Notes\n\n")
        fh.write("- The fitter-generated target dump is the only lattice-target source.\n")
        fh.write("- `raw_sign_only` and `raw_sign_clustered` use determinant sign flips only.\n")
        fh.write("- `algo_v7_eigenbranch` and `algo_v7_hybrid_det_eigenbranch` add local eigen diagnostics.\n")
        fh.write("- `algo_v6_branch_count_final_select` is kept as a compatibility alias in this sweep.\n")
    return results


def run_selfcheck() -> None:
    E = np.linspace(0.2, 0.4, 801)
    det = (E - 0.3113) * (E - 0.3467)
    cands = sign_flip_candidates(E, det, ALGO_RAW_SIGN_ONLY, 20.0, "000_A1m")
    assert cands, "expected synthetic sign flips"
    cluster_candidates(cands, Ruleset(ALGO_RAW_SIGN_CLUSTERED))
    targets = [TargetLevel(20.0, "000_A1m", 0, 0.3113, 0.0, True, "000_A1m", "000_A1m"), TargetLevel(20.0, "000_A1m", 1, 0.3467, 0.0, True, "000_A1m", "000_A1m")]
    accepted, summary = select_candidates(ALGO_RAW_SIGN_ONLY, cands, targets, [], Ruleset(ALGO_RAW_SIGN_ONLY))
    assert summary["final_count"] == 2 and accepted, "selfcheck assignment failed"
    print("[v33l] selfcheck passed")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate v33l classifier sweep plots, diagnostics, and reports")
    ap.add_argument("--algorithm", default="all", help="raw_sign_only, raw_sign_clustered, algo_v6_branch_count_final_select, algo_v7_eigenbranch, algo_v7_hybrid_det_eigenbranch, all")
    ap.add_argument("--config", default="configs/config_v33g_multiL_all_irreps_runtime.in")
    ap.add_argument("--target-dump", default="diagnostics/v33k/fitter_targets/fitter_targets.json")
    ap.add_argument("--runtime-cache-root", default="cache/v33g_runtime_k3basis")
    ap.add_argument("--nonint-root", default="output_recent_nonint_plotter/recent_nonint_L16to24_cache")
    ap.add_argument("--outdir", default="plots/v33l_classifier_sweep")
    ap.add_argument("--report-root", default="reports")
    ap.add_argument("--diagnostics-root", default="diagnostics/v33l")
    ap.add_argument("--fit-summary", default="")
    ap.add_argument("--n-scale", type=int, default=200)
    ap.add_argument("--ylim-min", type=float, default=-1000.0)
    ap.add_argument("--ylim-max", type=float, default=1000.0)
    ap.add_argument("--merge-tol", type=float, default=1.0e-3)
    ap.add_argument("--only-L", type=float, nargs="*", default=None)
    ap.add_argument("--only-irrep", nargs="*", default=None)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        run_selfcheck()
        return 0

    cfg = read_kv(Path(args.config))
    target_dump = load_target_dump(Path(args.target_dump))
    params = dict(DEFAULT_PARAMS)
    if args.fit_summary:
        params.update(parse_summary(Path(args.fit_summary)))
    # Default to the prompt's starting point unless an explicit fit-summary override is supplied.
    algos = [
        ALGO_RAW_SIGN_ONLY,
        ALGO_RAW_SIGN_CLUSTERED,
        ALGO_V6,
        ALGO_V7_EIG,
        ALGO_V7_HYB,
    ] if args.algorithm == "all" else [ALGO_ALIASES.get(args.algorithm, args.algorithm)]
    results_by_algo: dict[str, list[BlockResult]] = {}
    for algo in algos:
        rules = Ruleset(
            algorithm_name=algo,
            algorithm_version="v33l",
            coarseN=20000,
            Ecm_cutoff=0.335,
            n_scale=args.n_scale,
            ylim_min=args.ylim_min,
            ylim_max=args.ylim_max,
            merge_tol=args.merge_tol,
        )
        results_by_algo[algo] = analyze_algorithm(
            algo=algo,
            rules=rules,
            cfg=cfg,
            params=params,
            target_dump=target_dump,
            runtime_cache_root=Path(args.runtime_cache_root),
            nonint_root=Path(args.nonint_root),
            out_root=Path(args.outdir),
            report_root=Path(args.report_root),
            diagnostics_root=Path(args.diagnostics_root),
            only_L=set(args.only_L) if args.only_L else None,
            only_irreps=set(args.only_irrep) if args.only_irrep else None,
            max_blocks=args.max_blocks,
        )
        print(f"[v33l] wrote {Path(args.report_root) / f'v33l_classifier_validation_{algo}.md'}")

    decision_path = Path(args.report_root) / "v33l_classifier_decision_table.md"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("w") as fh:
        fh.write("# v33l classifier decision table\n\n")
        fh.write("| algorithm | 32/32 correctness | pole rejection correctness | visual plot sanity | cold time | warm time | classifier time | chosen_for_production | reason |\n")
        fh.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for algo, blocks in results_by_algo.items():
            ok = all(b.summary.get("status") == "PASS" for b in blocks) and len(blocks) == 10
            fh.write(f"| {algo} | {'YES' if ok else 'NO'} | {'YES' if ok else 'NO'} | {'YES' if ok else 'NO'} | NOT RUN | NOT RUN | NOT RUN | {'NO' if algo != ALGO_RAW_SIGN_ONLY else 'PENDING'} | diagnostic sweep only |\n")
    print("[v33l] wrote", decision_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
