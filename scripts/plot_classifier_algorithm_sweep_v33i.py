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
    algorithm_version: str = "v33i"
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
    cluster_id: int = -1
    branch_id: int = -1
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
    duplicate_score: float = 0.0
    branch_assignment_cost: float = math.nan
    nearest_noninteracting_energy: float = math.nan
    distance_to_nearest_noninteracting: float = math.nan
    nearest_lattice_energy: float = math.nan
    distance_to_nearest_lattice: float = math.nan
    block_lattice_cutoff_Ecm: float = math.nan
    E_count_hi_initial: float = math.nan
    E_count_hi_final: float = math.nan
    inside_count_window_initial: int = 0
    inside_count_window_final: int = 0
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
            nearest_lattice_energy=float("nan"),
            distance_to_nearest_lattice=float("nan"),
            nearest_noninteracting_energy=float("nan"),
            distance_to_nearest_noninteracting=float("nan"),
            score=zero_score - pole_score,
        )
        candidates.append(cand)
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
                zero_score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor))),
                pole_score=0.0,
                score=float(max(0.0, edge_log - np.log10(abs(float(y[i])) + floor))),
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

    block_cutoff = block.summary["block_lattice_cutoff_Ecm"]
    hi_initial = block.summary["E_count_hi_initial"]
    hi_final = block.summary["E_count_hi_final"]
    lo = min((l.Ecm for l in block.lattice_levels if l.selected), default=float(ecm.min()))
    ax.axvspan(lo, hi_final, color="goldenrod", alpha=0.12, zorder=0)
    ax.axvline(rules.Ecm_cutoff, color="grey", linestyle="--", linewidth=1.5, alpha=0.8, zorder=5, label=f"global cutoff {rules.Ecm_cutoff:.3f}")
    ax.axvline(block_cutoff, color="darkred", linestyle="-.", linewidth=1.6, alpha=0.9, zorder=5, label=f"block cutoff {block_cutoff:.6f}")
    ax.axvline(hi_final, color="seagreen", linestyle=":", linewidth=1.8, alpha=0.95, zorder=5, label=f"E_count_hi_final {hi_final:.6f}")
    ax.axvline(hi_initial, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8, zorder=5, label=f"E_count_hi_initial {hi_initial:.6f}")

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
            nonints = load_nonint_levels(nonint_root, L, irrep, rules.Ecm_cutoff)
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
            results.append(block)
            summary_rows.append(summary)
            plot_title = (
                f"{algo} | L{int(L)} {irrep} | coarseN={rules.coarseN} | "
                f"Ecm=[{float(ecm.min()):.6f},{float(ecm.max()):.6f}] | "
                f"global_cutoff={rules.Ecm_cutoff:.3f} | "
                f"block_cutoff={summary['block_lattice_cutoff_Ecm']:.6f} | "
                f"final_hi={summary['E_count_hi_final']:.6f} | "
                f"expansion={summary['window_expansion_used']:.6f} | "
                f"counts={summary['classifier_count_in_window_final']}/{summary['lattice_level_count']}/{summary['noninteracting_count_in_window']} | "
                f"{summary['PASS_FAIL']}"
            )
            plot_block(block, rules, out_dir, params, algo, plot_title)
            write_csv(block.candidate_csv, [cand.to_row() for cand in cands])
            with block.candidate_json.open("w") as fh:
                json.dump([cand.to_row() for cand in cands], fh, indent=2)

    summary_csv = diag_dir / "summary.csv"
    write_csv(summary_csv, summary_rows)
    ruleset_json = diag_dir / "ruleset.json"
    ruleset_json.write_text(json.dumps(rules.to_json(), indent=2, sort_keys=True) + "\n")

    report_path = report_root / f"v33i_classifier_visual_report_{algo}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as fh:
        fh.write(f"# v33i visual report: {algo}\n\n")
        fh.write("## Commands\n\n")
        fh.write("```bash\n")
        fh.write(f"python3 scripts/plot_classifier_algorithm_sweep_v33i.py --algorithm {algo}\n")
        fh.write("```\n\n")
        fh.write("## Ruleset\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(rules.to_json(), indent=2, sort_keys=True))
        fh.write("\n```\n\n")
        fh.write("## Block summary\n\n")
        fh.write("| Lbyas | irrep | lattice | nonint | cand | accepted | status | plot |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n")
        for row in summary_rows:
            plot_rel = (out_dir / f"L{int(float(row['Lbyas']))}_{row['irrep']}.png").as_posix()
            fh.write(
                f"| {int(float(row['Lbyas']))} | {row['irrep']} | {row['lattice_level_count']} | "
                f"{row['noninteracting_count_in_window']} | {row['candidate_count_total']} | "
                f"{row['accepted_true_zero_count']} | {row['PASS_FAIL']} | {plot_rel} |\n"
            )
        fh.write("\n## Notes\n\n")
        fh.write("- `algo_v5a_shape_only` keeps sign-flip shape scoring only.\n")
        fh.write("- `algo_v5b_shape_plus_cluster` adds cluster de-duplication.\n")
        fh.write("- `algo_v5c_shape_cluster_NI` adds non-interacting branch multiplicity.\n")
        fh.write("- `algo_v5d_shape_cluster_NI_lattice_window` adds the per-block count window and expansion check.\n")
        fh.write("- This sweep is diagnostic-first and does not modify the fitter binary.\n")

    return results


def selfcheck() -> None:
    def f1(E: np.ndarray, E0: float) -> np.ndarray:
        return E - E0

    def f2(E: np.ndarray, E0: float) -> np.ndarray:
        return (E - E0) ** 3

    def f3(E: np.ndarray, Ep: float) -> np.ndarray:
        return 1.0 / (E - Ep)

    E = np.linspace(0.2, 0.4, 401)
    rules = Ruleset("selfcheck")
    e0 = 0.31
    ep = 0.3337
    c1 = build_candidates(E, f1(E, e0), rules, "algo_v5a_shape_only", "hash", 20.0, "000_A1m")
    c2 = build_candidates(E, f2(E, e0), rules, "algo_v5a_shape_only", "hash", 20.0, "000_A1m")
    c3 = build_candidates(E, f3(E, ep), rules, "algo_v5a_shape_only", "hash", 20.0, "000_A1m")
    assert any(abs(c.E_estimate - e0) < 1e-3 for c in c1), "simple zero not detected"
    assert any(abs(c.E_estimate - e0) < 1e-3 for c in c2), "cubic zero not detected"
    assert any(c.pole_score > c.zero_score for c in c3), "pole not distinguished"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate v33i classifier sweep plots, diagnostics, and reports")
    ap.add_argument("--algorithm", default="all", help="algo_v5a_shape_only, algo_v5b_shape_plus_cluster, algo_v5c_shape_cluster_NI, algo_v5d_shape_cluster_NI_lattice_window, or all")
    ap.add_argument("--config", default="configs/config_v33g_multiL_all_irreps_runtime.in")
    ap.add_argument("--fit-summary", default="output_v33g/fit_all_irreps_runtime_ecm0335/debug_v33g_all_irreps_runtime_ecm0335_fit_summary_allL.dat")
    ap.add_argument("--runtime-cache-root", default="cache/v33g_runtime_k3basis")
    ap.add_argument("--nonint-root", default="output_recent_nonint_plotter/recent_nonint_L16to24_cache")
    ap.add_argument("--outdir", default="plots/v33i_classifier_sweeps")
    ap.add_argument("--report-root", default="reports")
    ap.add_argument("--diagnostics-root", default="diagnostics/v33i_classifier")
    ap.add_argument("--n-scale", type=int, default=200)
    ap.add_argument("--ylim-min", type=float, default=-1000.0)
    ap.add_argument("--ylim-max", type=float, default=1000.0)
    ap.add_argument("--merge-tol", type=float, default=1.0e-3)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        print("[v33i] selfcheck passed")
        return 0

    cfg = read_kv(Path(args.config))
    algos = [
        "algo_v5a_shape_only",
        "algo_v5b_shape_plus_cluster",
        "algo_v5c_shape_cluster_NI",
        "algo_v5d_shape_cluster_NI_lattice_window",
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
        print(f"[v33i] wrote {Path(args.report_root) / f'v33i_classifier_visual_report_{algo}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
