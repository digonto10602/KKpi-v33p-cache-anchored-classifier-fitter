#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


TARGET_IRREPS = ["000_A1m", "100_A2", "110_A2", "111_A2", "200_A2"]
TARGET_L_VALUES = [20.0, 24.0]
ALGO_V6 = "algo_v6_branch_count_final_select"
NONINT_TAG = {
    "000_A1m": ("000", "A1m"),
    "100_A2": ("001", "A2"),
    "110_A2": ("110", "A2"),
    "111_A2": ("111", "A2"),
    "200_A2": ("200", "A2"),
}
MAGIC = 0x5633334752544B33


@dataclass
class Ruleset:
    algorithm_name: str
    algorithm_version: str = "v33j"
    coarseN: int = 20000
    Ecm_cutoff: float = 0.335
    n_scale: int = 200
    ylim_min: float = -1000.0
    ylim_max: float = 1000.0
    merge_tol: float = 1.0e-3
    initial_soft_margin: float = 5.0e-4
    expansion_steps: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01)
    max_extension_above_lattice_cutoff: float = 0.01
    points_each_side_choices: tuple[int, ...] = (7, 9, 11, 15, 21, 31)
    local_zero_floor_rel: float = 1.0e-12
    zero_accept_margin: float = 0.15
    pole_accept_margin: float = 0.15
    cluster_merge_factor: float = 3.0
    branch_distance_weight: float = 1.0
    zero_score_weight: float = 1.0
    pole_score_weight: float = 1.0
    ambiguity_penalty: float = 0.5
    blowup_threshold: float = 25.0

    def to_json(self) -> dict[str, object]:
        out = asdict(self)
        out["expansion_steps"] = list(self.expansion_steps)
        out["points_each_side_choices"] = list(self.points_each_side_choices)
        return out


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
class LatticeLevel:
    index: int
    state: int
    Ecm: float
    err: float
    selected: bool
    path: Path


@dataclass
class NonintLevel:
    level: int
    Ecm: float
    multiplicity: int
    ptag: str
    irrep: str


@dataclass
class Candidate:
    algorithm_name: str
    algorithm_version: str
    ruleset_hash: str
    Lbyas: float
    irrep: str
    candidate_id: int
    candidate_type: str
    raw_candidate: bool = True
    shape_pass: bool = False
    cluster_id: int = -1
    cluster_representative: bool = False
    branch_id: int = -1
    assigned_to_branch: bool = False
    final_accepted: bool = False
    E_estimate: float = math.nan
    E_left: float = math.nan
    E_right: float = math.nan
    y_left: float = math.nan
    y_right: float = math.nan
    y_at_estimate_if_available: float = math.nan
    sign_flip: int = 0
    local_grid_spacing: float = math.nan
    window_points_each_side: int = 0
    median_abs_y: float = math.nan
    MAD_abs_y: float = math.nan
    robust_scale: float = math.nan
    min_abs_y_window: float = math.nan
    candidate_log_abs: float = math.nan
    median_edge_log_abs: float = math.nan
    valley_depth: float = math.nan
    peak_height: float = math.nan
    left_inward_fraction: float = math.nan
    right_inward_fraction: float = math.nan
    left_inward_growth_fraction: float = math.nan
    right_inward_growth_fraction: float = math.nan
    slope_left: float = math.nan
    slope_right: float = math.nan
    slope_ratio: float = math.nan
    reciprocal_zero_score: float = math.nan
    blowup_score: float = math.nan
    zero_score: float = math.nan
    pole_score: float = math.nan
    noise_score: float = math.nan
    discontinuity_score: float = math.nan
    duplicate_score: float = 0.0
    branch_assignment_cost: float = math.nan
    nearest_noninteracting_energy: float = math.nan
    distance_to_nearest_noninteracting: float = math.nan
    nearest_lattice_energy: float = math.nan
    distance_to_nearest_lattice: float = math.nan
    block_lattice_cutoff_Ecm: float = math.nan
    E_count_hi_initial: float = math.nan
    E_count_hi_final: float = math.nan
    E_window_min: float = math.nan
    E_window_upper_initial: float = math.nan
    E_window_upper_final: float = math.nan
    E_window_upper_hard: float = math.nan
    inside_count_window_initial: int = 0
    inside_count_window_final: int = 0
    inside_initial_window: int = 0
    inside_final_window: int = 0
    inside_hard_window: int = 0
    window_expansion_used: float = 0.0
    count_window_status: str = ""
    final_status: str = ""
    rejection_reason: str = ""
    accepted: bool = False
    energy: float = math.nan
    score: float = math.nan

    def to_row(self) -> dict[str, object]:
        d = asdict(self)
        d["accepted"] = int(self.accepted)
        return d


@dataclass
class BlockResult:
    Lbyas: float
    irrep: str
    runtime_cache: Path
    ecm: np.ndarray
    det: np.ndarray
    lattice_levels: list[LatticeLevel]
    nonint_levels: list[NonintLevel]
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


def parse_summary(path: Path) -> dict[str, float]:
    vals: dict[str, float] = {}
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {"K3iso0", "K3iso1", "K3B", "K3E"}:
            vals[parts[0]] = float(parts[1])
    return vals


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in split_words(text)]


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


def load_runtime_cache(cache_path: Path, params: dict[str, float], Lbyas: float, xi: float) -> tuple[np.ndarray, np.ndarray]:
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
        _i, off = read_i32(buf, off)
        _total_dim, off = read_i32(buf, off)
        _proj_dim, off = read_i32(buf, off)
        success, off = read_i32(buf, off)
        _reserved, off = read_i32(buf, off)
        e, off = read_f64(buf, off)
        _en, off = read_f64(buf, off)
        f3inv_proj, off = read_cpx_matrix(buf, off)
        b0, off = read_cpx_matrix(buf, off)
        b1, off = read_cpx_matrix(buf, off)
        bb, off = read_cpx_matrix(buf, off)
        be, off = read_cpx_matrix(buf, off)
        if success != 1:
            continue
        q = f3inv_proj + params["K3iso0"] * b0 + params["K3iso1"] * b1 + params["K3B"] * bb + params["K3E"] * be
        q = q / ((Lbyas * xi) ** 6.0)
        ecm.append(e)
        det_scaled.append(np.linalg.det(q).real)
    if not ecm:
        raise RuntimeError(f"No usable cache rows found in {cache_path}")
    return np.asarray(ecm, dtype=float), np.asarray(det_scaled, dtype=float)


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


def parse_jack_path(path: Path) -> tuple[float, str, int]:
    m = re.match(r"^(?P<L>\d+)_([^_]+_[^_]+)_n(?P<n>\d+)\.jack$", path.name)
    if not m:
        raise ValueError(f"Unrecognized jack filename: {path.name}")
    L = float(m.group("L"))
    irrep = m.group(2)
    state = int(m.group("n"))
    return L, irrep, state


def read_jack_values(path: Path, skip: int, col1based: int) -> list[float]:
    vals: list[float] = []
    with path.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            if lineno <= skip:
                continue
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < col1based:
                continue
            vals.append(float(cols[col1based - 1]))
    if not vals:
        raise RuntimeError(f"No jackknife samples found in {path}")
    return vals


def jackknife_resampling(data: list[float]) -> list[float]:
    if len(data) < 2:
        raise RuntimeError("jackknife_resampling requires at least 2 data points")
    total = sum(data)
    denom = float(len(data) - 1)
    return [(total - x) / denom for x in data]


def jackknife_average(vals: list[float]) -> float:
    return float(sum(vals) / len(vals))


def jackknife_error(vals: list[float]) -> float:
    avg = jackknife_average(vals)
    n = len(vals)
    if n < 2:
        raise RuntimeError("jackknife_error requires at least 2 samples")
    return math.sqrt(((n - 1) / n) * sum((x - avg) ** 2 for x in vals))


def nP_from_irrep(irrep: str) -> tuple[int, int, int]:
    if irrep == "000_A1m":
        return 0, 0, 0
    if irrep == "100_A2":
        return 0, 0, 1
    if irrep == "110_A2":
        return 1, 1, 0
    if irrep == "111_A2":
        return 1, 1, 1
    if irrep == "200_A2":
        return 0, 0, 2
    raise ValueError(f"Unsupported irrep: {irrep}")


def convert_samples_to_ecm(raw: list[float], etype: str, Lbyas: float, xi: float, irrep: str) -> list[float]:
    nx, ny, nz = nP_from_irrep(irrep)
    p = 2.0 * math.pi * math.sqrt(nx * nx + ny * ny + nz * nz) / (xi * Lbyas)
    out: list[float] = []
    for e in raw:
        if etype in {"En_lab", "Elab", "E_lab"}:
            arg = e * e - p * p
            if arg <= 0.0:
                raise RuntimeError(f"Nonpositive Ecm^2 after lab conversion for L={Lbyas} irrep={irrep}")
            out.append(math.sqrt(arg))
        else:
            out.append(e)
    return out


def load_lattice_levels(jack_dir: Path, ensemble: str, Lbyas: float, irrep: str, xi: float, etype: str, skip: int, col1based: int, cutoff: float) -> list[LatticeLevel]:
    prefix = f"{int(Lbyas)}_{irrep}_n"
    files = sorted(jack_dir.glob(prefix + "*.jack"))
    if not files and ensemble:
        files = sorted(jack_dir.glob(f"{ensemble}_{irrep}_n*.jack"))
    out: list[LatticeLevel] = []
    for idx, path in enumerate(files):
        m = re.search(r"_n(?P<n>\d+)\.jack$", path.name)
        if not m:
            raise ValueError(f"Unrecognized jack filename: {path.name}")
        state = int(m.group("n"))
        raw = read_jack_values(path, skip, col1based)
        resampled = jackknife_resampling(raw)
        ecm = convert_samples_to_ecm(resampled, etype, Lbyas, xi, irrep)
        mean = jackknife_average(ecm)
        err = jackknife_error(ecm)
        out.append(LatticeLevel(index=idx, state=state, Ecm=mean, err=err, selected=(mean <= cutoff), path=path))
    out.sort(key=lambda r: (r.Ecm, r.state, r.path.name))
    return out


def load_nonint_levels(nonint_root: Path, Lbyas: float, irrep: str, cutoff: float) -> list[NonintLevel]:
    ptag, iir = NONINT_TAG[irrep]
    path = nonint_root / f"recent_nonint_L{int(Lbyas)}_all_irreps_nonint3body.dat"
    out: list[NonintLevel] = []
    if not path.exists():
        return out
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


def local_window_indices(n: int, i: int, window: int) -> tuple[int, int]:
    lo = max(0, i - window)
    hi = min(n - 1, i + window + 1)
    return lo, hi


def slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return math.nan
    x0 = xs - xs.mean()
    y0 = ys - ys.mean()
    denom = float(np.dot(x0, x0))
    if denom == 0.0:
        return math.nan
    return float(np.dot(x0, y0) / denom)


def local_noise_score(yw: np.ndarray, robust: float) -> float:
    if len(yw) < 3:
        return 0.0
    diffs = np.diff(yw)
    if len(diffs) < 2:
        return 0.0
    osc = np.sum(np.sign(diffs[:-1]) != np.sign(diffs[1:]))
    return float(max(0.0, osc / max(len(diffs) - 1, 1)))


def local_discontinuity_score(yw: np.ndarray, robust: float) -> float:
    if len(yw) < 2:
        return 0.0
    jump = float(np.max(np.abs(np.diff(yw))))
    return float(max(0.0, jump / max(robust, 1.0e-300) - 1.0))


def build_candidates(ecm: np.ndarray, det: np.ndarray, rules: Ruleset, algo: str, ruleset_hash: str, Lbyas: float, irrep: str) -> list[Candidate]:
    finite = np.isfinite(ecm) & np.isfinite(det)
    e = ecm[finite]
    y = det[finite]
    order = np.argsort(e, kind="mergesort")
    e = e[order]
    y = y[order]
    if len(e) < 3:
        return []

    grid_spacing = float(np.median(np.diff(e))) if len(e) > 1 else math.nan
    floor = max(1.0e-300, float(np.median(np.abs(y))) * rules.local_zero_floor_rel)
    candidates: list[Candidate] = []
    cid = 0
    for i in range(len(e) - 1):
        s0, s1 = sign(float(y[i])), sign(float(y[i + 1]))
        if s0 == 0 or s1 == 0 or s0 == s1:
            continue
        lo, hi = local_window_indices(len(e), i, 31)
        idxs = np.arange(lo, hi)
        ew = e[idxs]
        yw = y[idxs]
        absw = np.abs(yw)
        logabs = np.log10(absw + floor)
        median_abs = float(np.median(absw))
        mad_abs = float(median_abs_deviation(absw))
        robust = max(median_abs, 1.4826 * mad_abs, floor)
        scaled_abs = absw / robust
        edge_log = float(np.median(np.concatenate([logabs[: min(2, len(logabs))], logabs[-min(2, len(logabs)) :]])))
        candidate_log_abs = float(np.log10(min(abs(float(y[i])), abs(float(y[i + 1]))) + floor))
        valley_depth = float(edge_log - candidate_log_abs)
        peak_height = float(candidate_log_abs - edge_log)
        left_inward_fraction = float(np.mean(np.diff(np.abs(y[max(0, i - 3) : i + 1])) < 0.0)) if i >= 1 else 0.0
        right_inward_fraction = float(np.mean(np.diff(np.abs(y[i + 1 : min(len(y), i + 5)])) < 0.0)) if i + 2 < len(y) else 0.0
        left_inward_growth_fraction = 1.0 - left_inward_fraction if not math.isnan(left_inward_fraction) else math.nan
        right_inward_growth_fraction = 1.0 - right_inward_fraction if not math.isnan(right_inward_fraction) else math.nan
        xs_left = e[max(0, i - 3) : i + 1]
        ys_left = y[max(0, i - 3) : i + 1]
        xs_right = e[i + 1 : min(len(y), i + 5)]
        ys_right = y[i + 1 : min(len(y), i + 5)]
        slope_left = slope(xs_left, ys_left)
        slope_right = slope(xs_right, ys_right)
        slope_ratio = math.nan
        if math.isfinite(slope_left) and math.isfinite(slope_right):
            denom = max(min(abs(slope_left), abs(slope_right)), 1.0e-300)
            slope_ratio = max(abs(slope_left), abs(slope_right)) / denom
        y_est = float(y[i] - y[i] * (e[i + 1] - e[i]) / (y[i + 1] - y[i])) if y[i + 1] != y[i] else 0.5 * (e[i] + e[i + 1])
        reciprocal_zero_score = 0.0
        if y[i] != 0.0 and y[i + 1] != 0.0:
            reciprocal_zero_score = 1.0 if sign(1.0 / float(y[i])) != sign(1.0 / float(y[i + 1])) else 0.0
        blowup = float(max(absw) / robust)
        noise_score = local_noise_score(yw, robust)
        discontinuity_score = local_discontinuity_score(yw, robust)
        zero_score = float(
            max(0.0, left_inward_fraction) + max(0.0, right_inward_fraction) + max(0.0, valley_depth)
        )
        pole_score = float(
            max(0.0, left_inward_growth_fraction) + max(0.0, right_inward_growth_fraction) + max(0.0, peak_height) + reciprocal_zero_score + 0.1 * max(0.0, blowup - 1.0)
        )
        cand = Candidate(
            algorithm_name=algo,
            algorithm_version=rules.algorithm_version,
            ruleset_hash=ruleset_hash,
            Lbyas=Lbyas,
            irrep=irrep,
            candidate_id=cid,
            candidate_type="sign_flip",
            E_estimate=float(y_est if math.isfinite(y_est) else 0.5 * (e[i] + e[i + 1])),
            E_left=float(e[i]),
            E_right=float(e[i + 1]),
            y_left=float(y[i]),
            y_right=float(y[i + 1]),
            y_at_estimate_if_available=float(np.interp(0.5 * (e[i] + e[i + 1]), e, y)),
            sign_flip=1,
            local_grid_spacing=grid_spacing,
            window_points_each_side=31,
            median_abs_y=median_abs,
            MAD_abs_y=mad_abs,
            robust_scale=robust,
            min_abs_y_window=float(np.min(absw)),
            candidate_log_abs=candidate_log_abs,
            median_edge_log_abs=edge_log,
            valley_depth=valley_depth,
            peak_height=peak_height,
            left_inward_fraction=left_inward_fraction,
            right_inward_fraction=right_inward_fraction,
            left_inward_growth_fraction=left_inward_growth_fraction,
            right_inward_growth_fraction=right_inward_growth_fraction,
            slope_left=slope_left,
            slope_right=slope_right,
            slope_ratio=slope_ratio,
            reciprocal_zero_score=reciprocal_zero_score,
            blowup_score=blowup,
            zero_score=zero_score,
            pole_score=pole_score,
            noise_score=noise_score,
            discontinuity_score=discontinuity_score,
            nearest_lattice_energy=float("nan"),
            distance_to_nearest_lattice=float("nan"),
            nearest_noninteracting_energy=float("nan"),
            distance_to_nearest_noninteracting=float("nan"),
            score=zero_score - pole_score,
            shape_pass=True,
        )
        candidates.append(cand)
        cid += 1

        if abs(float(y[i])) <= 10.0 * floor:
            exact_cand = Candidate(
                algorithm_name=algo,
                algorithm_version=rules.algorithm_version,
                ruleset_hash=ruleset_hash,
                Lbyas=Lbyas,
                irrep=irrep,
                candidate_id=cid,
                candidate_type="exact_zero",
                E_estimate=float(e[i]),
                E_left=float(e[max(0, i - 1)]),
                E_right=float(e[min(len(e) - 1, i + 1)]),
                y_left=float(y[max(0, i - 1)]),
                y_right=float(y[min(len(y) - 1, i + 1)]),
                y_at_estimate_if_available=float(y[i]),
                sign_flip=0,
                local_grid_spacing=grid_spacing,
                window_points_each_side=31,
                median_abs_y=median_abs,
                MAD_abs_y=mad_abs,
                robust_scale=robust,
                min_abs_y_window=float(np.min(absw)),
                candidate_log_abs=float(np.log10(abs(float(y[i])) + floor)),
                median_edge_log_abs=edge_log,
                valley_depth=float(edge_log - np.log10(abs(float(y[i])) + floor)),
                peak_height=float(np.log10(abs(float(y[i])) + floor) - edge_log),
                zero_score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor)) + 1.0),
                pole_score=0.0,
                noise_score=noise_score,
                discontinuity_score=discontinuity_score,
                score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor)) + 1.0),
                shape_pass=True,
            )
            candidates.append(exact_cand)
            cid += 1

    # Add local minimum candidates without a sign flip.
    for i in range(1, len(e) - 1):
        if not np.isfinite(y[i - 1]) or not np.isfinite(y[i]) or not np.isfinite(y[i + 1]):
            continue
        if abs(y[i]) <= min(abs(y[i - 1]), abs(y[i + 1])):
            lo, hi = local_window_indices(len(e), i, 11)
            idxs = np.arange(lo, hi)
            absw = np.abs(y[idxs])
            median_abs = float(np.median(absw))
            mad_abs = float(median_abs_deviation(absw))
            robust = max(median_abs, 1.4826 * mad_abs, floor)
            logabs = np.log10(absw + floor)
            edge_log = float(np.median(np.concatenate([logabs[: min(2, len(logabs))], logabs[-min(2, len(logabs)) :]])))
            cand = Candidate(
                algorithm_name=algo,
                algorithm_version=rules.algorithm_version,
                ruleset_hash=ruleset_hash,
                Lbyas=Lbyas,
                irrep=irrep,
                candidate_id=cid,
                candidate_type="local_minimum",
                E_estimate=float(e[i]),
                E_left=float(e[i - 1]),
                E_right=float(e[i + 1]),
                y_left=float(y[i - 1]),
                y_right=float(y[i + 1]),
                y_at_estimate_if_available=float(y[i]),
                sign_flip=0,
                local_grid_spacing=grid_spacing,
                window_points_each_side=11,
                median_abs_y=median_abs,
                MAD_abs_y=mad_abs,
                robust_scale=robust,
                min_abs_y_window=float(np.min(absw)),
                candidate_log_abs=float(np.log10(abs(float(y[i])) + floor)),
                median_edge_log_abs=edge_log,
                valley_depth=float(edge_log - np.log10(abs(float(y[i])) + floor)),
                peak_height=float(np.log10(abs(float(y[i])) + floor) - edge_log),
                noise_score=0.0,
                discontinuity_score=float(max(0.0, np.max(absw) / robust - 1.0)),
                zero_score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor))),
                pole_score=0.0,
                score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor))),
                shape_pass=True,
            )
            candidates.append(cand)
            cid += 1
    return candidates


def cluster_candidates(cands: list[Candidate], rules: Ruleset) -> list[list[int]]:
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: (cands[i].E_estimate, cands[i].candidate_id))
    clusters: list[list[int]] = []
    current = [order[0]]
    for idx in order[1:]:
        prev = current[-1]
        gap = abs(cands[idx].E_estimate - cands[prev].E_estimate)
        local = max(
            3.0 * min(
                x
                for x in [cands[idx].local_grid_spacing, cands[prev].local_grid_spacing]
                if math.isfinite(x) and x > 0.0
            )
            if any(math.isfinite(x) and x > 0.0 for x in [cands[idx].local_grid_spacing, cands[prev].local_grid_spacing])
            else 0.0,
            rules.merge_tol,
        )
        if gap <= local:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    return clusters


def nearest_energy(value: float, energies: list[float]) -> tuple[float, float]:
    if not energies:
        return math.nan, math.nan
    best = min(energies, key=lambda x: abs(x - value))
    return best, abs(best - value)


def load_nonint_candidates(nonints: list[NonintLevel]) -> tuple[list[float], dict[float, int]]:
    uniq: list[float] = []
    counts: dict[float, int] = {}
    for nl in nonints:
        found = None
        for e in uniq:
            if abs(e - nl.Ecm) <= 1.0e-10:
                found = e
                break
        if found is None:
            uniq.append(nl.Ecm)
            counts[nl.Ecm] = nl.multiplicity
        else:
            counts[found] += nl.multiplicity
    uniq.sort()
    return uniq, counts


def branch_windows(unique_energies: list[float]) -> list[tuple[float, float]]:
    if not unique_energies:
        return []
    if len(unique_energies) == 1:
        return [(-math.inf, math.inf)]
    mids = [(unique_energies[i] + unique_energies[i + 1]) / 2.0 for i in range(len(unique_energies) - 1)]
    windows: list[tuple[float, float]] = [(-math.inf, mids[0])]
    for i in range(1, len(unique_energies) - 1):
        windows.append((mids[i - 1], mids[i]))
    windows.append((mids[-1], math.inf))
    return windows


def adjacent_spacing(values: list[float], default: float = 0.005) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if len(vals) < 2:
        return default
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    return float(median(diffs)) if diffs else default


def build_branch_anchors(lattice: list[float], nonint: list[float]) -> tuple[list[float], list[tuple[float, float]], bool]:
    lattice = sorted(lattice)
    nonint = sorted(nonint)
    mismatch = len(lattice) != len(nonint)
    anchors: list[float] = []
    pairs = min(len(lattice), len(nonint))
    for i in range(pairs):
        anchors.append(0.6 * nonint[i] + 0.4 * lattice[i])
    if len(lattice) > pairs:
        anchors.extend(lattice[pairs:])
    if len(nonint) > pairs:
        anchors.extend(nonint[pairs:])
    anchors = sorted(anchors)
    return anchors, branch_windows(anchors), mismatch


def window_bounds_from_anchors(anchors: list[float], e_min: float, e_max: float, delta_grid: float, branch_spacing: float) -> list[tuple[float, float]]:
    if not anchors:
        return []
    pad = max(5.0 * delta_grid, 0.05 * branch_spacing, 2.0e-4)
    if len(anchors) == 1:
        return [(e_min, e_max)]
    bounds: list[tuple[float, float]] = []
    mids = [(anchors[i] + anchors[i + 1]) / 2.0 for i in range(len(anchors) - 1)]
    for i, a in enumerate(anchors):
        lo = e_min if i == 0 else mids[i - 1]
        hi = e_max if i == len(anchors) - 1 else mids[i]
        lo = max(e_min, lo - pad)
        hi = min(e_max, hi + pad)
        bounds.append((lo, hi))
    return bounds


def branch_assignment_cost(cand: Candidate, branch_E_NI: float, branch_E_latt: float, branch_window: tuple[float, float], delta_grid: float, branch_spacing: float) -> float:
    scale = max(branch_spacing, 10.0 * delta_grid, 1.0e-6)
    dist_ni = abs(cand.E_estimate - branch_E_NI) / scale
    dist_latt = abs(cand.E_estimate - branch_E_latt) / scale
    lo, hi = branch_window
    outside = 0.0
    if cand.E_estimate < lo:
        outside = (lo - cand.E_estimate) / scale
    elif cand.E_estimate > hi:
        outside = (cand.E_estimate - hi) / scale
    zero = max(0.0, float(cand.zero_score))
    pole = max(0.0, float(cand.pole_score))
    noise = max(0.0, float(cand.noise_score if math.isfinite(cand.noise_score) else 0.0))
    return float(1.0 * dist_ni + 0.6 * dist_latt - 1.5 * zero + 2.5 * pole + 1.5 * noise + 3.0 * outside)


def classify_block_v6(
    ecm: np.ndarray,
    det: np.ndarray,
    lattice_levels: list[LatticeLevel],
    nonint_levels: list[NonintLevel],
    rules: Ruleset,
    algo: str,
    ruleset_hash: str,
    Lbyas: float,
    irrep: str,
) -> tuple[list[Candidate], dict[str, object]]:
    candidates = build_candidates(ecm, det, rules, algo, ruleset_hash, Lbyas, irrep)
    if not candidates:
        return [], {
            "algorithm_name": algo,
            "Lbyas": Lbyas,
            "irrep": irrep,
            "PASS_FAIL": "FAIL_NO_CANDIDATES",
            "blocks_status": "FAIL_NO_CANDIDATES",
            "failure_reason": "no_candidates",
            "lattice_count": 0,
            "noninteracting_count": 0,
            "raw_candidate_count": 0,
            "shape_pass_count": 0,
            "cluster_count": 0,
            "final_true_zero_count": 0,
        }

    finite_ecm = np.asarray(ecm[np.isfinite(ecm)], dtype=float)

    e_cache_min = float(np.min(finite_ecm)) if len(finite_ecm) else math.nan
    e_cache_max = float(np.max(finite_ecm)) if len(finite_ecm) else math.nan
    delta_grid = float(np.median(np.diff(finite_ecm))) if len(finite_ecm) > 1 else 0.0

    lattice_selected = sorted(l.Ecm for l in lattice_levels if l.selected)
    lattice_all = sorted(l.Ecm for l in lattice_levels)
    lattice_count = len(lattice_selected)
    e_latt_max = max(lattice_selected) if lattice_selected else (max(lattice_all) if lattice_all else math.nan)
    nominal_cutoff = rules.Ecm_cutoff

    unique_nonint, nonint_mult = load_nonint_candidates(nonint_levels)
    branch_ref_nonint = [e for e in unique_nonint if e <= max(nominal_cutoff, e_latt_max if math.isfinite(e_latt_max) else nominal_cutoff) + 0.1]
    delta_branch = adjacent_spacing(branch_ref_nonint, default=adjacent_spacing(lattice_selected, default=0.005))
    if delta_branch <= 0.0 or not math.isfinite(delta_branch):
        delta_branch = 0.005

    base_upper = max(nominal_cutoff, e_latt_max if math.isfinite(e_latt_max) else nominal_cutoff)
    small_margin = max(10.0 * delta_grid, 0.10 * delta_branch, 5.0e-4)
    hard_margin = max(50.0 * delta_grid, 0.35 * delta_branch, 0.005)
    e_window_min = e_cache_min
    e_window_upper_initial = min(e_cache_max, base_upper + small_margin)
    e_window_upper_hard = min(e_cache_max, base_upper + hard_margin)
    if not math.isfinite(e_window_upper_initial):
        e_window_upper_initial = e_cache_max
    if not math.isfinite(e_window_upper_hard):
        e_window_upper_hard = e_cache_max

    clusters = cluster_candidates(candidates, rules)
    cluster_rep_idx: dict[int, int] = {}
    for cid, idxs in enumerate(clusters):
        best = min(
            idxs,
            key=lambda i: (
                -float(candidates[i].zero_score if math.isfinite(candidates[i].zero_score) else 0.0),
                float(candidates[i].pole_score if math.isfinite(candidates[i].pole_score) else 0.0),
                float(candidates[i].discontinuity_score if math.isfinite(candidates[i].discontinuity_score) else 0.0),
                float(candidates[i].noise_score if math.isfinite(candidates[i].noise_score) else 0.0),
                abs(candidates[i].E_estimate),
            ),
        )
        cluster_rep_idx[cid] = best
        for i in idxs:
            candidates[i].cluster_id = cid
            candidates[i].cluster_representative = i == best

    def update_window_flags(upper: float) -> None:
        for cand in candidates:
            cand.E_window_min = e_window_min
            cand.E_window_upper_initial = e_window_upper_initial
            cand.E_window_upper_final = upper
            cand.E_window_upper_hard = e_window_upper_hard
            cand.inside_initial_window = int(e_window_min <= cand.E_estimate <= e_window_upper_initial)
            cand.inside_final_window = int(e_window_min <= cand.E_estimate <= upper)
            cand.inside_hard_window = int(e_window_min <= cand.E_estimate <= e_window_upper_hard)
            cand.inside_count_window_initial = cand.inside_initial_window
            cand.inside_count_window_final = cand.inside_final_window
            cand.nearest_lattice_energy, cand.distance_to_nearest_lattice = nearest_energy(cand.E_estimate, lattice_selected or lattice_all)
            cand.nearest_noninteracting_energy, cand.distance_to_nearest_noninteracting = nearest_energy(cand.E_estimate, unique_nonint)

    def select_for_upper(upper: float) -> tuple[list[int], dict[str, object]]:
        update_window_flags(upper)
        nonint_in_window = [e for e in unique_nonint if e <= upper]
        anchors, windows, mismatch = build_branch_anchors(lattice_selected or lattice_all, nonint_in_window)
        branch_spacing = adjacent_spacing(anchors, default=delta_branch)
        branch_bounds = window_bounds_from_anchors(anchors, e_window_min, upper, delta_grid, branch_spacing)
        if not branch_bounds and lattice_count:
            branch_bounds = [(e_window_min, upper)] * lattice_count
            anchors = list(lattice_selected or lattice_all)
        if not branch_bounds:
            return [], {
                "branch_windows": [],
                "mismatch": mismatch,
                "nonint_in_window": nonint_in_window,
                "lattice_in_window": lattice_selected,
                "branch_spacing": branch_spacing,
            }

        rep_indices = [cluster_rep_idx[cid] for cid in sorted(cluster_rep_idx)]
        branch_edges: list[tuple[float, int, int]] = []
        rep_branch_choice: dict[int, tuple[int, float]] = {}
        for idx in rep_indices:
            cand = candidates[idx]
            cand.assigned_to_branch = False
            cand.branch_id = -1
            cand.branch_assignment_cost = math.nan
            if cand.E_estimate > upper or cand.E_estimate < e_window_min:
                continue
            for b, (lo, hi) in enumerate(branch_bounds):
                if b >= len(anchors):
                    break
                branch_E_latt = lattice_selected[b] if b < len(lattice_selected) else (lattice_selected[-1] if lattice_selected else anchors[b])
                branch_E_NI = nonint_in_window[b] if b < len(nonint_in_window) else (nonint_in_window[-1] if nonint_in_window else anchors[b])
                cost = branch_assignment_cost(cand, branch_E_NI, branch_E_latt, (lo, hi), delta_grid, branch_spacing)
                branch_edges.append((cost, b, idx))
        branch_edges.sort(key=lambda x: (x[0], x[1], x[2]))
        used_candidates: set[int] = set()
        for cost, b, idx in branch_edges:
            if b in rep_branch_choice or idx in used_candidates:
                continue
            rep_branch_choice[b] = (idx, cost)
            used_candidates.add(idx)
        for b, (idx, cost) in rep_branch_choice.items():
            cand = candidates[idx]
            cand.branch_id = b
            cand.branch_assignment_cost = cost
            cand.assigned_to_branch = True

        selected_idx = sorted(idx for idx, _ in rep_branch_choice.values())
        for idx in rep_indices:
            cand = candidates[idx]
            if idx not in selected_idx:
                if cand.branch_id < 0 and branch_edges:
                    cand.branch_id = min(
                        range(len(branch_bounds)),
                        key=lambda b: branch_assignment_cost(
                            cand,
                            nonint_in_window[b] if b < len(nonint_in_window) else (nonint_in_window[-1] if nonint_in_window else anchors[min(b, len(anchors) - 1)]),
                            lattice_selected[b] if b < len(lattice_selected) else (lattice_selected[-1] if lattice_selected else anchors[min(b, len(anchors) - 1)]),
                            branch_bounds[min(b, len(branch_bounds) - 1)],
                            delta_grid,
                            branch_spacing,
                        ),
                    )
                cand.assigned_to_branch = cand.branch_id >= 0
                if cand.branch_id >= 0 and not math.isfinite(cand.branch_assignment_cost):
                    cand.branch_assignment_cost = branch_assignment_cost(
                        cand,
                        nonint_in_window[min(cand.branch_id, len(nonint_in_window) - 1)] if nonint_in_window else anchors[min(cand.branch_id, len(anchors) - 1)],
                        lattice_selected[min(cand.branch_id, len(lattice_selected) - 1)] if lattice_selected else anchors[min(cand.branch_id, len(anchors) - 1)],
                        branch_bounds[min(cand.branch_id, len(branch_bounds) - 1)],
                        delta_grid,
                        branch_spacing,
                    )
        return selected_idx, {
            "branch_windows": branch_bounds,
            "mismatch": mismatch,
            "nonint_in_window": nonint_in_window,
            "lattice_in_window": lattice_selected,
            "branch_spacing": branch_spacing,
            "anchors": anchors,
        }

    final_upper = e_window_upper_initial
    selection: list[int] = []
    meta: dict[str, object] = {}
    for upper in [e_window_upper_initial] + [min(e_window_upper_hard, base_upper + step) for step in rules.expansion_steps]:
        selection, meta = select_for_upper(upper)
        if len(selection) >= lattice_count:
            final_upper = upper
            break
        final_upper = upper
    if not selection and lattice_count:
        selection, meta = select_for_upper(e_window_upper_hard)
        final_upper = e_window_upper_hard

    update_window_flags(final_upper)
    branch_windows_final = meta.get("branch_windows", [])
    mismatch = bool(meta.get("mismatch", False))
    nonint_in_final_window = list(meta.get("nonint_in_window", []))
    lattice_in_final_window = list(meta.get("lattice_in_window", lattice_selected))

    accepted_idx = set(selection)
    for cand in candidates:
        if cand.cluster_id >= 0 and cand.cluster_representative:
            cand.final_accepted = cand.candidate_id in {candidates[i].candidate_id for i in accepted_idx}
        else:
            cand.final_accepted = False
        if cand.candidate_id in {candidates[i].candidate_id for i in accepted_idx}:
            cand.accepted = True
            cand.final_status = "accepted_true_zero"
            cand.rejection_reason = "ok"
        else:
            cand.accepted = False

    for cand in candidates:
        if cand.accepted and (not math.isfinite(cand.pole_score) or cand.pole_score > cand.zero_score + rules.pole_accept_margin):
            cand.accepted = False
            cand.final_accepted = False
            cand.final_status = "rejected_pole_like"
            cand.rejection_reason = "pole_like"
        elif cand.candidate_id not in {candidates[i].candidate_id for i in accepted_idx}:
            if cand.cluster_representative:
                cand.final_status = "rejected_worse_assignment_cost"
                cand.rejection_reason = "worse_assignment_cost"
            else:
                cand.final_status = "rejected_duplicate_same_branch"
                cand.rejection_reason = "duplicate_same_branch"

    final_true_zeros = [cand for cand in candidates if cand.accepted and cand.inside_final_window]
    final_count = len(final_true_zeros)
    raw_count = len(candidates)
    shape_pass_count = sum(1 for c in candidates if c.shape_pass)
    cluster_count = len(clusters)
    lattice_count_final = len([e for e in lattice_selected if e <= final_upper])
    nonint_count_final = len([e for e in unique_nonint if e <= final_upper])
    pole_like_present = any(c.pole_score > c.zero_score + rules.pole_accept_margin for c in candidates)

    if mismatch:
        blocks_status = "COUNT_PRIOR_MISMATCH"
        failure_reason = "count_prior_mismatch"
    elif pole_like_present:
        blocks_status = "FAIL_POLE_LIKE"
        failure_reason = "pole_like"
    elif final_count > lattice_count:
        blocks_status = "FAIL_CLASSIFIER_OVERCOUNT_IN_LATTICE_WINDOW"
        failure_reason = "overcount"
    elif final_count < lattice_count:
        blocks_status = "MISSING_BRANCH_ROOT"
        failure_reason = "missing_branch_root"
    else:
        blocks_status = "PASS"
        failure_reason = "ok"

    for cand in candidates:
        cand.E_count_hi_initial = e_window_upper_initial
        cand.E_count_hi_final = final_upper
        cand.window_expansion_used = max(0.0, final_upper - e_window_upper_initial)
        cand.count_window_status = blocks_status
        if cand.accepted and cand.inside_final_window:
            cand.final_status = "accepted_true_zero"
            cand.rejection_reason = "ok"
        elif cand.E_estimate > final_upper:
            cand.final_status = "rejected_outside_validation_window"
            cand.rejection_reason = "outside_validation_window"
        elif cand.pole_score > cand.zero_score + rules.pole_accept_margin:
            cand.final_status = "rejected_pole_like"
            cand.rejection_reason = "pole_like"

    summary = {
        "algorithm_name": algo,
        "Lbyas": Lbyas,
        "irrep": irrep,
        "coarseN": rules.coarseN,
        "Ecm_min": e_cache_min,
        "Ecm_max": e_cache_max,
        "Ecm_cutoff": rules.Ecm_cutoff,
        "E_latt_max": e_latt_max,
        "E_window_min": e_window_min,
        "E_window_upper_initial": e_window_upper_initial,
        "E_window_upper_final": final_upper,
        "E_window_upper_hard": e_window_upper_hard,
        "lattice_count": lattice_count,
        "noninteracting_count": len(nonint_in_final_window),
        "raw_candidate_count": raw_count,
        "shape_pass_count": shape_pass_count,
        "cluster_count": cluster_count,
        "final_true_zero_count": final_count,
        "blocks_status": blocks_status,
        "failure_reason": failure_reason,
        "plot_path": "",
        "count_prior_mismatch": int(mismatch),
        "accepted_branch_count": len(selection),
        "final_window_count": final_count,
        "branch_windows": branch_windows_final,
        "lattice_levels_in_window": lattice_count_final,
        "noninteracting_levels_in_window": nonint_count_final,
        "final_status": blocks_status,
        "PASS_FAIL": blocks_status,
    }
    return candidates, summary


def classify_block(
    ecm: np.ndarray,
    det: np.ndarray,
    lattice_levels: list[LatticeLevel],
    nonint_levels: list[NonintLevel],
    rules: Ruleset,
    algo: str,
    ruleset_hash: str,
    Lbyas: float,
    irrep: str,
) -> tuple[list[Candidate], dict[str, object]]:
    if algo == ALGO_V6:
        return classify_block_v6(ecm, det, lattice_levels, nonint_levels, rules, algo, ruleset_hash, Lbyas, irrep)
    candidates = build_candidates(ecm, det, rules, algo, ruleset_hash, Lbyas, irrep)
    if not candidates:
        return [], {
            "status": "FAIL_NEEDS_VISUAL_INSPECTION",
            "reason": "no_candidates",
            "lattice_selected_count": int(sum(1 for l in lattice_levels if l.selected)),
            "noninteracting_count": int(len(nonint_levels)),
        }

    clusters = cluster_candidates(candidates, rules)
    cluster_id_by_idx: dict[int, int] = {}
    for cid, idxs in enumerate(clusters):
        for idx in idxs:
            cluster_id_by_idx[idx] = cid
            candidates[idx].cluster_id = cid

    unique_nonint, nonint_mult = load_nonint_candidates(nonint_levels)
    windows = branch_windows(unique_nonint)
    block_cutoff = max((l.Ecm for l in lattice_levels if l.selected), default=max((l.Ecm for l in lattice_levels), default=math.nan))
    selected_count = sum(1 for l in lattice_levels if l.selected)
    if not math.isfinite(block_cutoff):
        block_cutoff = max((l.Ecm for l in lattice_levels), default=math.nan)
    spacing = float(np.median(np.diff(sorted(l.Ecm for l in lattice_levels)))) if len(lattice_levels) > 1 else 0.0
    initial_soft_margin = max(2.0 * spacing, rules.initial_soft_margin)
    e_hi = min(float(ecm.max()), block_cutoff + initial_soft_margin)
    hi_initial = e_hi
    hi_final = e_hi
    expansion_used = 0.0
    lattice_selected_energies = [l.Ecm for l in lattice_levels if l.selected]
    if lattice_selected_energies:
        selected_lo = min(lattice_selected_energies)
    else:
        selected_lo = float(ecm.min())

    for cand in candidates:
        cand.block_lattice_cutoff_Ecm = block_cutoff
        cand.E_count_hi_initial = hi_initial
        cand.E_count_hi_final = hi_final
        cand.nearest_lattice_energy, cand.distance_to_nearest_lattice = nearest_energy(cand.E_estimate, [l.Ecm for l in lattice_levels])
        cand.nearest_noninteracting_energy, cand.distance_to_nearest_noninteracting = nearest_energy(cand.E_estimate, unique_nonint)

    for cand in candidates:
        cand.inside_count_window_initial = int(selected_lo <= cand.E_estimate <= hi_initial)

    def assign_branches(cands: list[Candidate], upper: float) -> None:
        for cand in cands:
            cand.final_status = ""
            cand.rejection_reason = ""
            cand.branch_id = -1
            cand.count_window_status = "outside"
            if not (selected_lo <= cand.E_estimate <= upper):
                cand.final_status = "rejected_outside_count_window"
                cand.rejection_reason = "outside_count_window"
                cand.accepted = False
                continue
            cand.count_window_status = "inside"

            if cand.zero_score <= 0.0 and cand.candidate_type != "local_minimum":
                cand.final_status = "rejected_low_zero_score"
                cand.rejection_reason = "low_zero_score"
                cand.accepted = False
                continue
            if cand.pole_score > cand.zero_score + rules.pole_accept_margin:
                cand.final_status = "rejected_pole"
                cand.rejection_reason = "pole_like"
                cand.accepted = False
                continue

            if unique_nonint:
                branch = -1
                branch_cost = math.inf
                for b, (lo, hi) in enumerate(windows):
                    if lo <= cand.E_estimate < hi or (b == len(windows) - 1 and cand.E_estimate <= hi):
                        branch = b
                        branch_cost = abs(cand.E_estimate - unique_nonint[b])
                        break
                if branch < 0:
                    best, dist = nearest_energy(cand.E_estimate, unique_nonint)
                    cand.final_status = "rejected_off_branch"
                    cand.rejection_reason = "off_branch"
                    cand.branch_assignment_cost = dist
                    cand.nearest_noninteracting_energy = best
                    cand.distance_to_nearest_noninteracting = dist
                    cand.accepted = False
                    continue
                cand.branch_id = branch
                cand.branch_assignment_cost = branch_cost

            cand.accepted = True
            cand.final_status = "accepted_true_zero"

    assign_branches(candidates, hi_initial)

    if algo in {"algo_v5b_shape_plus_cluster", "algo_v5c_shape_cluster_NI", "algo_v5d_shape_cluster_NI_lattice_window"}:
        for cluster in clusters:
            accepted = [candidates[i] for i in cluster if candidates[i].accepted]
            if len(accepted) <= 1:
                continue
            accepted.sort(key=lambda c: (-c.zero_score, c.pole_score, abs(c.E_estimate)))
            keep = accepted[0]
            for loser in accepted[1:]:
                loser.accepted = False
                loser.duplicate_score = 1.0
                loser.final_status = "rejected_duplicate_same_branch"
                loser.rejection_reason = "duplicate_same_branch"
            keep.duplicate_score = 0.0

    if algo in {"algo_v5c_shape_cluster_NI", "algo_v5d_shape_cluster_NI_lattice_window"} and unique_nonint:
        # Allow at most the multiplicity available for each non-interacting branch.
        branch_keep: dict[int, list[Candidate]] = {}
        for cand in candidates:
            if cand.accepted and cand.branch_id >= 0:
                branch_keep.setdefault(cand.branch_id, []).append(cand)
        for branch, branch_cands in branch_keep.items():
            branch_cands.sort(key=lambda c: (-c.zero_score, c.pole_score, abs(c.E_estimate)))
            allowed = nonint_mult.get(unique_nonint[branch], 1)
            for loser in branch_cands[allowed:]:
                loser.accepted = False
                loser.final_status = "rejected_not_used_by_lattice_window"
                loser.rejection_reason = "branch_multiplicity_exhausted"

    selected_count = sum(1 for l in lattice_levels if l.selected)
    initial_accepted = sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= hi_initial)
    final_hi = hi_initial
    window_status = "PASS"
    final_reason = "ok"
    if algo == "algo_v5d_shape_cluster_NI_lattice_window":
        if initial_accepted > selected_count:
            window_status = "FAIL_CLASSIFIER_OVERCOUNT_IN_LATTICE_WINDOW"
            final_reason = "overcount"
        elif initial_accepted < selected_count:
            final_reason = "undercount"
            for step in rules.expansion_steps:
                trial = min(float(ecm.max()), block_cutoff + min(rules.max_extension_above_lattice_cutoff, step))
                assign_branches(candidates, trial)
                if algo in {"algo_v5b_shape_plus_cluster", "algo_v5c_shape_cluster_NI", "algo_v5d_shape_cluster_NI_lattice_window"}:
                    for cluster in clusters:
                        accepted = [candidates[i] for i in cluster if candidates[i].accepted]
                        if len(accepted) <= 1:
                            continue
                        accepted.sort(key=lambda c: (-c.zero_score, c.pole_score, abs(c.E_estimate)))
                        for loser in accepted[1:]:
                            loser.accepted = False
                            loser.final_status = "rejected_duplicate_same_branch"
                            loser.rejection_reason = "duplicate_same_branch"
                if algo in {"algo_v5c_shape_cluster_NI", "algo_v5d_shape_cluster_NI_lattice_window"} and unique_nonint:
                    branch_keep = {}
                    for cand in candidates:
                        if cand.accepted and cand.branch_id >= 0:
                            branch_keep.setdefault(cand.branch_id, []).append(cand)
                    for branch, branch_cands in branch_keep.items():
                        branch_cands.sort(key=lambda c: (-c.zero_score, c.pole_score, abs(c.E_estimate)))
                        allowed = nonint_mult.get(unique_nonint[branch], 1)
                        for loser in branch_cands[allowed:]:
                            loser.accepted = False
                            loser.final_status = "rejected_not_used_by_lattice_window"
                            loser.rejection_reason = "branch_multiplicity_exhausted"
                count_trial = sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= trial)
                if count_trial == selected_count:
                    final_hi = trial
                    expansion_used = final_hi - hi_initial
                    window_status = "PASS"
                    final_reason = "expanded_to_match"
                    break
                if count_trial > selected_count:
                    final_hi = trial
                    expansion_used = final_hi - hi_initial
                    window_status = "FAIL_OVERCOUNT_AFTER_WINDOW_EXPANSION"
                    final_reason = "overcount_after_expansion"
                    break
            else:
                final_hi = min(float(ecm.max()), block_cutoff + rules.max_extension_above_lattice_cutoff)
                expansion_used = final_hi - hi_initial
                if sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= final_hi) < selected_count:
                    window_status = "FAIL_MISSED_TRUE_ZERO_OR_TOO_STRICT_WINDOW"
                    final_reason = "missed_true_zero"
    else:
        final_hi = hi_initial
        expansion_used = 0.0

    # Final pass: assign statuses after the count-window decision.
    for cand in candidates:
        cand.E_count_hi_final = final_hi
        cand.window_expansion_used = expansion_used
        cand.inside_count_window_final = int(selected_lo <= cand.E_estimate <= final_hi)
        if cand.accepted and cand.E_estimate > final_hi:
            cand.final_status = "rejected_outside_count_window"
            cand.rejection_reason = "outside_count_window"
            cand.accepted = False

    if algo == "algo_v5d_shape_cluster_NI_lattice_window":
        accepted_inside = sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= final_hi)
        if accepted_inside != selected_count:
            if accepted_inside > selected_count and window_status == "PASS":
                window_status = "FAIL_CLASSIFIER_OVERCOUNT_IN_LATTICE_WINDOW"
                final_reason = "overcount"
            elif accepted_inside < selected_count and window_status == "PASS":
                window_status = "FAIL_MISSED_TRUE_ZERO_OR_TOO_STRICT_WINDOW"
                final_reason = "missed_true_zero"

    for cand in candidates:
        if cand.accepted and selected_lo <= cand.E_estimate <= final_hi:
            cand.final_status = "accepted_true_zero"
        elif cand.final_status == "":
            cand.final_status = "rejected_outside_count_window"
            cand.rejection_reason = "outside_count_window"

    if window_status != "PASS":
        for cand in candidates:
            if cand.accepted and selected_lo <= cand.E_estimate <= final_hi:
                cand.final_status = window_status
                cand.rejection_reason = final_reason

    summary = {
        "algorithm_name": algo,
        "Lbyas": Lbyas,
        "irrep": irrep,
        "coarseN": rules.coarseN,
        "Ecm_min": float(ecm.min()),
        "Ecm_max": float(ecm.max()),
        "Ecm_cutoff": rules.Ecm_cutoff,
        "block_lattice_cutoff_Ecm": block_cutoff,
        "E_count_hi_initial": hi_initial,
        "E_count_hi_final": final_hi,
        "window_expansion_used": expansion_used,
        "lattice_selected_count": selected_count,
        "noninteracting_count_in_window": sum(1 for e in unique_nonint if selected_lo <= e <= final_hi),
        "classifier_count_in_window_initial": initial_accepted,
        "classifier_count_in_window_final": sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= final_hi),
        "candidate_count_total": len(candidates),
        "accepted_true_zero_count": sum(1 for c in candidates if c.accepted),
        "rejected_pole_count": sum(1 for c in candidates if c.final_status == "rejected_pole"),
        "rejected_duplicate_count": sum(1 for c in candidates if c.final_status == "rejected_duplicate_same_branch"),
        "rejected_off_branch_count": sum(1 for c in candidates if c.final_status == "rejected_off_branch"),
        "ambiguous_count": sum(1 for c in candidates if c.final_status == "ambiguous_needs_visual_inspection"),
        "lattice_level_count": selected_count,
        "noninteracting_count": len(nonint_levels),
        "count_match_lattice": int(sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= final_hi) == selected_count),
        "count_match_noninteracting": int(
            sum(1 for c in candidates if c.accepted and selected_lo <= c.E_estimate <= final_hi)
            == sum(1 for e in unique_nonint if selected_lo <= e <= final_hi)
        ),
        "PASS_FAIL": "PASS" if window_status == "PASS" else window_status,
        "count_window_status": window_status,
        "final_status": window_status,
        "rejection_reason": final_reason,
        "selected_lo": selected_lo,
        "final_hi": final_hi,
    }
    return candidates, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def plot_block(block: BlockResult, rules: Ruleset, outdir: Path, params: dict[str, float], algo: str, full_title: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6.8), constrained_layout=True)
    ecm = block.ecm
    det = block.det
    for n in range(rules.n_scale + 1):
        yy = det * (10.0 ** n)
        yy = np.asarray(yy, dtype=float)
        yy[~np.isfinite(yy)] = np.nan
        ax.plot(ecm, yy, color="black", linewidth=0.75, alpha=0.18 if n else 0.9, zorder=2)

    block_cutoff = block.summary.get("E_latt_max", block.summary.get("block_lattice_cutoff_Ecm", math.nan))
    hi_initial = block.summary.get("E_window_upper_initial", block.summary.get("E_count_hi_initial", math.nan))
    hi_final = block.summary.get("E_window_upper_final", block.summary.get("E_count_hi_final", math.nan))
    hi_hard = block.summary.get("E_window_upper_hard", hi_final)
    lo = block.summary.get("E_window_min", min((l.Ecm for l in block.lattice_levels if l.selected), default=float(ecm.min())))
    ax.axvspan(lo, hi_final, color="goldenrod", alpha=0.12, zorder=0)
    ax.axvline(rules.Ecm_cutoff, color="grey", linestyle="--", linewidth=1.5, alpha=0.8, zorder=5, label=f"global cutoff {rules.Ecm_cutoff:.3f}")
    if math.isfinite(block_cutoff):
        ax.axvline(block_cutoff, color="darkred", linestyle="-.", linewidth=1.6, alpha=0.9, zorder=5, label=f"block cutoff {block_cutoff:.6f}")
    if math.isfinite(hi_final):
        ax.axvline(hi_final, color="seagreen", linestyle=":", linewidth=1.8, alpha=0.95, zorder=5, label=f"E_window_upper_final {hi_final:.6f}")
    if math.isfinite(hi_initial):
        ax.axvline(hi_initial, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8, zorder=5, label=f"E_window_upper_initial {hi_initial:.6f}")
    if math.isfinite(hi_hard):
        ax.axvline(hi_hard, color="slateblue", linestyle="--", linewidth=1.1, alpha=0.55, zorder=5, label=f"E_window_upper_hard {hi_hard:.6f}")

    for i, pair in enumerate(block.summary.get("branch_windows", []) or []):
        try:
            lo_b, hi_b = pair
        except Exception:
            continue
        ax.axvspan(lo_b, hi_b, color="lightgrey", alpha=0.03 + 0.01 * (i % 3), zorder=0)

    accepted_in = [c.E_estimate for c in block.candidates if c.accepted and lo <= c.E_estimate <= hi_final]
    accepted_out = [c.E_estimate for c in block.candidates if c.accepted and not (lo <= c.E_estimate <= hi_final)]
    rejected_pole = [c.E_estimate for c in block.candidates if c.final_status == "rejected_pole"]
    rejected_dup = [c.E_estimate for c in block.candidates if c.final_status == "rejected_duplicate_same_branch"]
    rejected_out = [c.E_estimate for c in block.candidates if c.final_status == "rejected_outside_count_window"]
    ambiguous = [c.E_estimate for c in block.candidates if c.final_status == "ambiguous_needs_visual_inspection"]

    ax.scatter(accepted_in, np.zeros(len(accepted_in)), s=92, facecolors="white", edgecolors="black", linewidths=1.5, zorder=20, label=f"accepted in window: {len(accepted_in)}")
    if accepted_out:
        ax.scatter(accepted_out, np.zeros(len(accepted_out)), s=92, facecolors="none", edgecolors="black", linewidths=1.5, zorder=19, label=f"accepted outside window: {len(accepted_out)}")
    if rejected_pole:
        ax.scatter(rejected_pole, np.zeros(len(rejected_pole)), s=42, marker="x", c="crimson", linewidths=1.3, zorder=18, label=f"rejected pole: {len(rejected_pole)}")
    if rejected_dup:
        ax.scatter(rejected_dup, np.zeros(len(rejected_dup)), s=42, marker="x", c="dimgray", linewidths=1.3, zorder=18, label=f"rejected duplicate: {len(rejected_dup)}")
    if rejected_out:
        ax.scatter(rejected_out, np.zeros(len(rejected_out)), s=30, marker="v", c="teal", linewidths=1.2, zorder=18, label=f"rejected outside window: {len(rejected_out)}")
    if ambiguous:
        ax.scatter(ambiguous, np.zeros(len(ambiguous)), s=38, marker="^", c="purple", linewidths=1.2, zorder=18, label=f"ambiguous: {len(ambiguous)}")

    lattice_in = [l.Ecm for l in block.lattice_levels if l.selected]
    lattice_out = [l.Ecm for l in block.lattice_levels if not l.selected]
    ax.scatter(lattice_in, np.zeros(len(lattice_in)), s=120, facecolors="white", edgecolors="royalblue", linewidths=1.8, zorder=25, label=f"lattice in window: {len(lattice_in)}")
    if lattice_out:
        ax.scatter(lattice_out, np.zeros(len(lattice_out)), s=70, facecolors="white", edgecolors="darkorange", linewidths=1.2, zorder=24, label=f"lattice outside window: {len(lattice_out)}")

    for nl in block.nonint_levels:
        ax.axvline(nl.Ecm, color="darkorange", linestyle="--", alpha=0.35, linewidth=1.0, zorder=6)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.9)
    ax.set_ylim(rules.ylim_min, rules.ylim_max)
    ax.set_xlabel(r"$E_{cm}$")
    ax.set_ylabel(r"$\det[(V^\dagger QC V)/(L\xi)^6]$")
    ax.set_title(full_title, fontsize=11)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="best", ncol=2)
    outpath = outdir / f"L{int(block.Lbyas)}_{block.irrep}.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def analyze_algorithm(
    algo: str,
    rules: Ruleset,
    cfg: dict[str, str],
    fit_summary_path: Path,
    runtime_cache_root: Path,
    nonint_root: Path,
    out_root: Path,
    report_root: Path,
    diagnostics_root: Path,
) -> list[BlockResult]:
    ruleset_hash = hashlib.sha1(json.dumps(rules.to_json(), sort_keys=True).encode()).hexdigest()[:12]
    params = parse_summary(fit_summary_path)
    if not params:
        raise RuntimeError(f"Missing fit summary parameters: {fit_summary_path}")
    jack_dir = Path(cfg.get("lattice_jackknife_dir", "/home/digonto/Codes/KKpi_I2/spectrum/Ecm_data/data"))
    ensemble = cfg.get("ensemble", "szscl21_20_128_b1p50_t_x4p300_um0p0840_sm0p0743_n1p265")
    etype = cfg.get("lattice_jack_energy_type", "En_lab")
    skip = int(cfg.get("jack_skip_header_lines", "1"))
    col1based = int(cfg.get("jack_energy_column", "2"))
    lvalues = parse_float_list(cfg.get("Lbyas_values", "20 24")) or TARGET_L_VALUES
    irreps_by_L: dict[float, list[str]] = {}
    for L in lvalues:
        irreps_by_L[L] = split_words(cfg.get(f"irreps_L{int(L)}", cfg.get("list_of_mom", "000_A1m 100_A2 110_A2 111_A2 200_A2")))

    out_dir = out_root / algo
    diag_dir = diagnostics_root / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    results: list[BlockResult] = []
    summary_rows: list[dict[str, object]] = []
    for L in lvalues:
        for irrep in irreps_by_L[L]:
            if irrep not in TARGET_IRREPS:
                continue
            cache_path = runtime_cache_root / f"v33g_Lbyas{int(L)}_xi3p444_irrep{irrep}_coarse20000_runtime_K3basis.bin"
            meta = runtime_meta(cache_path)
            ecm, det = load_runtime_cache(cache_path, params, meta.Lbyas, meta.xi)
            lattice_levels = load_lattice_levels(jack_dir, ensemble, L, irrep, float(cfg.get("xival", "3.444")), etype, skip, col1based, rules.Ecm_cutoff)
            nonint_cutoff = float(np.max(ecm)) if algo == ALGO_V6 else rules.Ecm_cutoff
            nonints = load_nonint_levels(nonint_root, L, irrep, nonint_cutoff)
            cands, summary = classify_block(ecm, det, lattice_levels, nonints, rules, algo, ruleset_hash, L, irrep)
            block = BlockResult(
                Lbyas=L,
                irrep=irrep,
                runtime_cache=cache_path,
                ecm=ecm,
                det=det,
                lattice_levels=lattice_levels,
                nonint_levels=nonints,
                candidates=cands,
                summary=summary,
                plot_path=out_dir / f"L{int(L)}_{irrep}.png",
                candidate_csv=diag_dir / f"L{int(L)}_{irrep}_candidates.csv",
                candidate_json=diag_dir / f"L{int(L)}_{irrep}_candidates.json",
            )
            summary["plot_path"] = block.plot_path.as_posix()
            summary.setdefault("candidate_count_total", summary.get("raw_candidate_count", len(cands)))
            summary.setdefault("accepted_true_zero_count", summary.get("final_true_zero_count", 0))
            summary.setdefault("lattice_level_count", summary.get("lattice_count", 0))
            summary.setdefault("noninteracting_count_in_window", summary.get("noninteracting_count", 0))
            summary.setdefault("classifier_count_in_window_final", summary.get("final_true_zero_count", 0))
            results.append(block)
            summary_rows.append(summary)
            plot_title = (
                f"{algo} | L{int(L)} {irrep} | coarseN={rules.coarseN} | "
                f"lattice={summary.get('lattice_count', summary.get('lattice_level_count', 0))} | "
                f"nonint={summary.get('noninteracting_count', summary.get('noninteracting_count_in_window', 0))} | "
                f"raw={summary.get('raw_candidate_count', summary.get('candidate_count_total', 0))} | "
                f"cluster={summary.get('cluster_count', 0)} | "
                f"final={summary.get('final_true_zero_count', summary.get('classifier_count_in_window_final', 0))} | "
                f"{summary.get('blocks_status', summary.get('PASS_FAIL', 'UNKNOWN'))}"
            )
            plot_block(block, rules, out_dir, params, algo, plot_title)
            write_csv(block.candidate_csv, [cand.to_row() for cand in cands])
            with block.candidate_json.open("w") as fh:
                json.dump([cand.to_row() for cand in cands], fh, indent=2)

    summary_csv = diag_dir / "summary.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(diag_dir / "summary_by_block.csv", summary_rows)
    ruleset_json = diag_dir / "ruleset.json"
    ruleset_json.write_text(json.dumps(rules.to_json(), indent=2, sort_keys=True) + "\n")

    report_path = report_root / f"v33j_classifier_visual_report_{algo}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as fh:
        fh.write(f"# v33j visual report: {algo}\n\n")
        fh.write("## Commands\n\n")
        fh.write("```bash\n")
        fh.write(f"python3 scripts/plot_classifier_algorithm_sweep_v33j.py --algorithm {algo}\n")
        fh.write("```\n\n")
        fh.write("## Ruleset\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(rules.to_json(), indent=2, sort_keys=True))
        fh.write("\n```\n\n")
        fh.write("## Block summary\n\n")
        fh.write("| Lbyas | irrep | lattice | nonint | raw | cluster | final | status | plot |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in summary_rows:
            plot_rel = (out_dir / f"L{int(float(row['Lbyas']))}_{row['irrep']}.png").as_posix()
            fh.write(
                f"| {int(float(row['Lbyas']))} | {row['irrep']} | {row.get('lattice_count', row.get('lattice_level_count', 0))} | "
                f"{row.get('noninteracting_count', row.get('noninteracting_count_in_window', 0))} | "
                f"{row.get('raw_candidate_count', row.get('candidate_count_total', 0))} | "
                f"{row.get('cluster_count', 0)} | {row.get('final_true_zero_count', row.get('accepted_true_zero_count', 0))} | "
                f"{row.get('blocks_status', row.get('PASS_FAIL', 'UNKNOWN'))} | {plot_rel} |\n"
            )
        fh.write("\n## Notes\n\n")
        fh.write("- `algo_v5a_shape_only` keeps sign-flip shape scoring only.\n")
        fh.write("- `algo_v5b_shape_plus_cluster` adds cluster de-duplication.\n")
        fh.write("- `algo_v5c_shape_cluster_NI` adds non-interacting branch multiplicity.\n")
        fh.write("- `algo_v5d_shape_cluster_NI_lattice_window` adds the per-block count window and expansion check.\n")
        fh.write(f"- `algo_v6_branch_count_final_select` adds branch-count final selection and controlled expansion.\n")
        fh.write("- This sweep is diagnostic-first and does not modify the fitter binary.\n")

    if algo == ALGO_V6:
        validation_path = report_root / "v33j_algo_v6_block_validation.md"
        with validation_path.open("w") as fh:
            fh.write("# v33j algo_v6 block validation\n\n")
            fh.write("## Summary\n\n")
            fh.write("| Lbyas | irrep | lattice | nonint | raw | cluster | final | status | reason |\n")
            fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")
            for row in summary_rows:
                fh.write(
                    f"| {int(float(row['Lbyas']))} | {row['irrep']} | {row.get('lattice_count', 0)} | "
                    f"{row.get('noninteracting_count', 0)} | {row.get('raw_candidate_count', 0)} | "
                    f"{row.get('cluster_count', 0)} | {row.get('final_true_zero_count', 0)} | "
                    f"{row.get('blocks_status', 'UNKNOWN')} | {row.get('failure_reason', '')} |\n"
                )
            fh.write("\n## Commands\n\n")
            fh.write("```bash\n")
            fh.write("python3 scripts/plot_classifier_algorithm_sweep_v33j.py --algorithm algo_v6_branch_count_final_select\n")
            fh.write("```\n")

    return results


def selfcheck() -> None:
    def mk_levels(lattice_es: list[float], nonint_es: list[float]) -> tuple[list[LatticeLevel], list[NonintLevel]]:
        latt = [
            LatticeLevel(index=i, state=i + 1, Ecm=e, err=0.0, selected=True, path=Path(f"synthetic_lat_{i}.jack"))
            for i, e in enumerate(lattice_es)
        ]
        nonints = [
            NonintLevel(level=i + 1, Ecm=e, multiplicity=1, ptag="000", irrep="A1m")
            for i, e in enumerate(nonint_es)
        ]
        return latt, nonints

    def run_case(name: str, det_fn, lattice_es: list[float], nonint_es: list[float], expected_count: int, expected_status: str, expected_reason: str = "") -> None:
        E = np.linspace(0.2, 0.4, 801)
        det = det_fn(E)
        latt, nonints = mk_levels(lattice_es, nonint_es)
        rules = Ruleset(ALGO_V6)
        cands, summary = classify_block_v6(E, det, latt, nonints, rules, ALGO_V6, "selfcheck", 20.0, "000_A1m")
        got_count = int(summary.get("final_true_zero_count", 0))
        got_status = str(summary.get("blocks_status", ""))
        got_reason = str(summary.get("failure_reason", ""))
        ok = got_count == expected_count and got_status == expected_status and (not expected_reason or expected_reason == got_reason)
        print(f"[selfcheck] {name}: {'PASS' if ok else 'FAIL'} accepted={got_count} expected={expected_count} status={got_status} reason={got_reason}")
        assert ok, f"{name}: got count={got_count} status={got_status} reason={got_reason}"
        assert cands, f"{name}: no candidates returned"

    run_case(
        "one_branch_many_raw",
        lambda E: (E - 0.31) * (1.0 + 0.02 * np.sin(2000.0 * E)),
        [0.31],
        [0.31],
        1,
        "PASS",
    )
    run_case(
        "two_branches_dup_first",
        lambda E: (E - 0.29) * (E - 0.345) * (1.0 + 0.01 * np.sin(1500.0 * E)),
        [0.29, 0.345],
        [0.295, 0.34],
        2,
        "PASS",
    )
    run_case(
        "pole_sign_flip",
        lambda E: 1.0 / (E - 0.3337),
        [0.3337],
        [0.3337],
        0,
        "FAIL_POLE_LIKE",
        "pole_like",
    )
    run_case(
        "zero_above_cutoff_hard",
        lambda E: (E - 0.3362) * (1.0 + 0.005 * np.sin(1200.0 * E)),
        [0.3348],
        [0.3360],
        1,
        "PASS",
    )
    run_case(
        "extra_root_outside_window",
        lambda E: (E - 0.31) * (E - 0.349) * (1.0 + 0.005 * np.sin(1000.0 * E)),
        [0.31],
        [0.31],
        1,
        "PASS",
    )
    run_case(
        "missing_branch",
        lambda E: (E - 0.31) * (1.0 + 0.01 * np.sin(900.0 * E)),
        [0.31, 0.344],
        [0.31, 0.344],
        1,
        "MISSING_BRANCH_ROOT",
        "missing_branch_root",
    )
    run_case(
        "count_mismatch",
        lambda E: (E - 0.31) * (1.0 + 0.01 * np.sin(700.0 * E)),
        [0.31],
        [0.31, 0.318],
        1,
        "COUNT_PRIOR_MISMATCH",
        "count_prior_mismatch",
    )
    run_case(
        "storm_pole",
        lambda E: np.sin(3000.0 * (E - 0.3337)) / (E - 0.3337),
        [0.3337],
        [0.3337],
        1,
        "FAIL_POLE_LIKE",
        "pole_like",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate v33j classifier sweep plots, diagnostics, and reports")
    ap.add_argument("--algorithm", default="all", help="algo_v5a_shape_only, algo_v5b_shape_plus_cluster, algo_v5c_shape_cluster_NI, algo_v5d_shape_cluster_NI_lattice_window, algo_v6_branch_count_final_select, or all")
    ap.add_argument("--config", default="configs/config_v33g_multiL_all_irreps_runtime.in")
    ap.add_argument("--fit-summary", default="output_v33g/fit_all_irreps_runtime_ecm0335/debug_v33g_all_irreps_runtime_ecm0335_fit_summary_allL.dat")
    ap.add_argument("--runtime-cache-root", default="cache/v33g_runtime_k3basis")
    ap.add_argument("--nonint-root", default="output_recent_nonint_plotter/recent_nonint_L16to24_cache")
    ap.add_argument("--outdir", default="plots/v33j_classifier_sweep")
    ap.add_argument("--report-root", default="reports")
    ap.add_argument("--diagnostics-root", default="diagnostics/v33j")
    ap.add_argument("--n-scale", type=int, default=200)
    ap.add_argument("--ylim-min", type=float, default=-1000.0)
    ap.add_argument("--ylim-max", type=float, default=1000.0)
    ap.add_argument("--merge-tol", type=float, default=1.0e-3)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        print("[v33j] selfcheck passed")
        return 0

    cfg = read_kv(Path(args.config))
    algos = [
        "algo_v5a_shape_only",
        "algo_v5b_shape_plus_cluster",
        "algo_v5c_shape_cluster_NI",
        "algo_v5d_shape_cluster_NI_lattice_window",
        ALGO_V6,
    ] if args.algorithm == "all" else [args.algorithm]

    for algo in algos:
        rules = Ruleset(
            algorithm_name=algo,
            n_scale=args.n_scale,
            ylim_min=args.ylim_min,
            ylim_max=args.ylim_max,
            merge_tol=args.merge_tol,
        )
        analyze_algorithm(
            algo=algo,
            rules=rules,
            cfg=cfg,
            fit_summary_path=Path(args.fit_summary),
            runtime_cache_root=Path(args.runtime_cache_root),
            nonint_root=Path(args.nonint_root),
            out_root=Path(args.outdir),
            report_root=Path(args.report_root),
            diagnostics_root=Path(args.diagnostics_root),
        )
        print(f"[v33j] wrote {Path(args.report_root) / f'v33j_classifier_visual_report_{algo}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
