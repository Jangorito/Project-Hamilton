"""Physics-based minimum-curvature racing line for Hirochi Raceway.

Pipeline
--------
1. compute_track_boundaries  — interpolate DecalRoad widths onto centreline points,
                               compute left/right boundary XY positions.
2. find_racing_line          — L-BFGS-B minimisation of total squared curvature
                               over a scalar offset field α[i] ∈ [−hw, +hw].
3. compute_speed_profile     — forward-backward traction-circle integration.
4. build_physics_raceline    — orchestrator; returns point-dict list matching the
                               ai_raceline JSON format so it can be loaded with
                               write_raceline_json / TrackCentreline.from_json.

Run as a script to regenerate data/hirochi_track/physics_raceline.json::

    python src/beamng_rl/track/physics_raceline.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from beamng_rl.track.geometry import resample_polyline
from beamng_rl.track.query_utils import TrackCentreline
from beamng_rl.track.raceline_loader import build_sequential_chain, load_items_lines

# ---------------------------------------------------------------------------
# Physical constants / defaults
# ---------------------------------------------------------------------------

_G = 9.81  # m/s²
DEFAULT_MU = 1.1  # tyre–road friction coefficient (track-spec etkc)
DEFAULT_ENGINE_ACCEL = 8.0  # m/s²  — longitudinal engine cap (≈0.82g)
DEFAULT_MAX_SPEED = 80.0  # m/s  — hard cap (~288 km/h; physically unreachable)
DEFAULT_MARGIN_M = 0.5  # metres clearance kept from each wall
DEFAULT_RESAMPLE_M = 2.0  # racing-line output point spacing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_raw_path_with_widths(
    items_file: Path,
    highlight_ids: list[int],
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Parse DecalRoad chain and return XYZ + per-node road widths."""
    decal_roads = load_items_lines(items_file)
    chain_slices = build_sequential_chain(decal_roads, highlight_ids)

    raw_xyz: list[tuple[float, float, float]] = []
    raw_widths: list[float] = []

    for road_idx in highlight_ids:
        start_i, end_i = chain_slices[road_idx]
        nodes = decal_roads[road_idx]["nodes"]
        for node in nodes[start_i : end_i + 1]:
            x, y = float(node[0]), float(node[1])
            z = float(node[2]) if len(node) > 2 else 0.0
            w = float(node[3]) if len(node) > 3 else 7.0
            raw_xyz.append((x, y, z))
            raw_widths.append(w)

    return raw_xyz, raw_widths


def _cumulative_arc_lengths_2d(pts: np.ndarray) -> np.ndarray:
    """Arc-lengths along a polyline using XY only."""
    dists = np.zeros(len(pts), dtype=float)
    for i in range(1, len(pts)):
        dists[i] = dists[i - 1] + float(np.linalg.norm(pts[i, :2] - pts[i - 1, :2]))
    return dists


def _compute_curvatures(path_xy: np.ndarray) -> np.ndarray:
    """Circumcircle curvature at every point of a closed-loop polyline.

    Fully vectorised: no Python loop.  Returns 1/radius (1/m).
    """
    A = np.roll(path_xy, 1, axis=0)   # previous points
    B = path_xy
    C = np.roll(path_xy, -1, axis=0)  # next points

    ab = B - A
    ac = C - A
    bc = C - B

    sa = np.linalg.norm(ab, axis=1)
    sb = np.linalg.norm(bc, axis=1)
    sc = np.linalg.norm(ac, axis=1)

    twice_area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    denom = sa * sb * sc

    valid = (sa > 1e-9) & (sb > 1e-9) & (sc > 1e-9) & (twice_area > 1e-9) & (denom > 1e-9)
    kappas = np.zeros(len(path_xy), dtype=float)
    kappas[valid] = 2.0 * twice_area[valid] / denom[valid]
    return kappas


def _unit_left_normals(pts: np.ndarray) -> np.ndarray:
    """Unit normal vectors pointing 90° left of the path direction (closed loop)."""
    n = len(pts)
    normals = np.zeros((n, 2), dtype=float)
    for i in range(n):
        prev_pt = pts[(i - 1) % n, :2]
        next_pt = pts[(i + 1) % n, :2]
        t = next_pt - prev_pt
        length = float(np.linalg.norm(t))
        if length > 1e-9:
            t /= length
        # left normal: rotate 90° CCW → (−ty, tx)
        normals[i] = [-t[1], t[0]]
    return normals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_track_boundaries(
    centreline: TrackCentreline,
    items_file: Path,
    highlight_ids: list[int],
    *,
    margin_m: float = DEFAULT_MARGIN_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute left/right boundary positions for each centreline point.

    Returns
    -------
    left_boundary  : (N, 3) float array — left wall positions
    right_boundary : (N, 3) float array — right wall positions
    half_widths    : (N,)   float array — usable half-width after margin
    normals        : (N, 2) float array — unit left-normal at each centreline point
    """
    raw_xyz, raw_widths = _derive_raw_path_with_widths(items_file, highlight_ids)

    raw_arr = np.array(raw_xyz, dtype=float)
    raw_dists = _cumulative_arc_lengths_2d(raw_arr)
    raw_total = raw_dists[-1]

    raw_widths_arr = np.array(raw_widths, dtype=float)

    # Scale raw arc-lengths to match centreline total length so both share
    # the same [0, total_lap_length] domain for interpolation.
    cl_total = centreline.total_lap_length
    raw_dists_scaled = raw_dists * (cl_total / raw_total)

    cl_arc = np.array(centreline.cumulative_arc_lengths[:-1], dtype=float)  # N values, not N+1
    widths_at_cl = np.interp(cl_arc, raw_dists_scaled, raw_widths_arr)

    half_widths = np.clip(widths_at_cl / 2.0 - margin_m, 0.1, None)

    pts = np.array(centreline.points_xyz, dtype=float)
    normals = _unit_left_normals(pts)

    left_boundary = pts.copy()
    right_boundary = pts.copy()
    left_boundary[:, :2] += normals * half_widths[:, None]
    right_boundary[:, :2] -= normals * half_widths[:, None]

    return left_boundary, right_boundary, half_widths, normals


def find_racing_line(
    centreline: TrackCentreline,
    normals: np.ndarray,
    half_widths: np.ndarray,
    *,
    max_iter: int = 1000,
    verbose: bool = True,
) -> np.ndarray:
    """Minimum-curvature racing line via L-BFGS-B.

    Decision variable α[i] ∈ [−hw[i], +hw[i]].
    Path: p[i] = centreline[i] + α[i] * normals[i]
    Objective: minimise Σ κ[i]²

    Returns
    -------
    path_xyz : (N, 3) float array
    """
    pts = np.array(centreline.points_xyz, dtype=float)
    n = len(pts)
    pts_xy = pts[:, :2].copy()

    def path_xy_from_alpha(alpha: np.ndarray) -> np.ndarray:
        return pts_xy + alpha[:, None] * normals

    def objective(alpha: np.ndarray) -> float:
        p_xy = path_xy_from_alpha(alpha)
        kappas = _compute_curvatures(p_xy)
        return float(np.sum(kappas**2))

    alpha0 = np.zeros(n, dtype=float)
    bounds = [(-hw, hw) for hw in half_widths]

    if verbose:
        print(f"  Optimising racing line over {n} path points …")

    call_count = [0]

    def cb(_):
        call_count[0] += 1
        if verbose and call_count[0] % 50 == 0:
            print(f"    iteration {call_count[0]} …")

    result = minimize(
        objective,
        alpha0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=cb,
        options={
            "maxiter": max_iter,
            "maxfun": max_iter * 600,
            "ftol": 1e-14,
            "gtol": 1e-8,
            "eps": 1e-5,  # finite-difference step
        },
    )

    if verbose:
        print(
            f"  Done: converged={result.success}  "
            f"f={result.fun:.6e}  nit={result.nit}  nfev={result.nfev}"
        )

    alpha_opt = result.x
    p_xy = path_xy_from_alpha(alpha_opt)
    path_xyz = np.column_stack([p_xy, pts[:, 2]])
    return path_xyz


def compute_speed_profile(
    path_xyz: np.ndarray,
    *,
    mu: float = DEFAULT_MU,
    engine_accel_mps2: float = DEFAULT_ENGINE_ACCEL,
    max_speed_mps: float = DEFAULT_MAX_SPEED,
    min_corner_radius_m: float = 8.0,
    n_passes: int = 4,
) -> np.ndarray:
    """Forward-backward traction-circle velocity profile on a closed-loop path.

    Constraints applied at every point:
      * Cornering limit  v ≤ sqrt(μg / κ)
      * Braking limit    v[i] ≤ sqrt(v[i+1]² + 2 · a_brake · ds)
      * Acceleration cap v[i+1] ≤ sqrt(v[i]² + 2 · a_accel · ds)

    where a_brake / a_accel are computed from the traction circle and the
    lateral acceleration already consumed by the corner radius.

    Returns
    -------
    speeds : (N,) float array in m/s
    """
    n = len(path_xyz)
    mu_g = mu * _G

    # Segment lengths (XY only)
    ds = np.zeros(n, dtype=float)
    for i in range(n):
        d = float(np.linalg.norm(path_xyz[(i + 1) % n, :2] - path_xyz[i, :2]))
        ds[i] = max(d, 1e-6)

    kappas = _compute_curvatures(path_xyz[:, :2])
    # Clamp curvature: very high values are centreline-data artifacts (kinks at
    # DecalRoad segment joins), not real corners.  min_corner_radius_m gives a
    # physically meaningful floor on what the track actually demands.
    kappa_max = 1.0 / max(min_corner_radius_m, 1e-3)
    kappas = np.clip(kappas, 1e-6, kappa_max)

    # Cornering speed limit
    v = np.minimum(np.sqrt(mu_g / kappas), max_speed_mps)

    for _ in range(n_passes):
        # Backward pass — braking
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            a_lat_j = min(v[j] ** 2 * float(kappas[j]), mu_g)
            a_brake = math.sqrt(max(0.0, mu_g**2 - a_lat_j**2))
            v_limit = math.sqrt(v[j] ** 2 + 2.0 * a_brake * float(ds[i]))
            v[i] = min(v[i], v_limit, max_speed_mps)

        # Forward pass — acceleration
        for i in range(n):
            j = (i + 1) % n
            a_lat_i = min(v[i] ** 2 * float(kappas[i]), mu_g)
            a_traction = math.sqrt(max(0.0, mu_g**2 - a_lat_i**2))
            a_accel = min(engine_accel_mps2, a_traction)
            v_limit = math.sqrt(v[i] ** 2 + 2.0 * a_accel * float(ds[i]))
            v[j] = min(v[j], v_limit, max_speed_mps)

    return v


def build_physics_raceline(
    centreline_json: Path,
    items_json: Path,
    highlight_ids: list[int],
    *,
    mu: float = DEFAULT_MU,
    margin_m: float = DEFAULT_MARGIN_M,
    engine_accel_mps2: float = DEFAULT_ENGINE_ACCEL,
    max_speed_mps: float = DEFAULT_MAX_SPEED,
    output_spacing_m: float = DEFAULT_RESAMPLE_M,
    verbose: bool = True,
) -> list[dict]:
    """Full pipeline: centreline → boundaries → racing line → speed profile.

    Returns a list of point dicts compatible with write_raceline_json / from_json.
    """
    track = TrackCentreline.from_json(centreline_json)

    if verbose:
        print("Computing track boundaries …")
    left_bd, right_bd, half_widths, normals = compute_track_boundaries(
        track, items_json, highlight_ids, margin_m=margin_m
    )

    if verbose:
        print(f"  Width range: {half_widths.min() * 2:.1f} – {half_widths.max() * 2:.1f} m usable")

    if verbose:
        print("Finding minimum-curvature racing line …")
    path_xyz_raw = find_racing_line(track, normals, half_widths, verbose=verbose)

    # Resample the optimised path to uniform spacing for a clean output
    if output_spacing_m > 0:
        path_list = [(float(p[0]), float(p[1]), float(p[2])) for p in path_xyz_raw]
        path_list = resample_polyline(path_list, spacing=output_spacing_m)
        path_xyz = np.array(path_list, dtype=float)
    else:
        path_xyz = path_xyz_raw

    if verbose:
        print("Computing speed profile …")
    speeds = compute_speed_profile(
        path_xyz,
        mu=mu,
        engine_accel_mps2=engine_accel_mps2,
        max_speed_mps=max_speed_mps,
    )

    pts_cl = np.array(track.points_xyz, dtype=float)
    normals_resampled = _unit_left_normals(path_xyz)

    # Arc-length progress along the racing line
    racing_dists = _cumulative_arc_lengths_2d(path_xyz)

    n = len(path_xyz)
    output: list[dict] = []
    for i in range(n):
        prev_p = path_xyz[(i - 1) % n]
        next_p = path_xyz[(i + 1) % n]
        dx = float(next_p[0]) - float(prev_p[0])
        dy = float(next_p[1]) - float(prev_p[1])
        heading = math.atan2(dy, dx) if abs(dx) > 1e-9 or abs(dy) > 1e-9 else 0.0

        # Lateral deviation from centreline — project onto centreline normal
        # (positive = left of centreline when looking forward)
        q = track.query(tuple(path_xyz[i]))
        lateral_err = float(q.signed_lateral_error)

        output.append(
            {
                "x": float(path_xyz[i, 0]),
                "y": float(path_xyz[i, 1]),
                "z": float(path_xyz[i, 2]),
                "progress_m": float(racing_dists[i]),
                "speed_mps": float(speeds[i]),
                "heading_rad": heading,
                "static_lateral_error_m": lateral_err,
            }
        )

    return output


# ---------------------------------------------------------------------------
# CLI / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _repo = Path(__file__).resolve().parents[3]
    _data = _repo / "data" / "hirochi_track"
    _HIGHLIGHT_IDS = [30, 12, 10, 17]

    parser = argparse.ArgumentParser(description="Compute physics-based racing line")
    parser.add_argument("--centreline", type=Path, default=_data / "centreline_resampled_2_0m.json")
    parser.add_argument("--items", type=Path, default=_data / "items.level.json")
    parser.add_argument("--output", type=Path, default=_data / "physics_raceline.json")
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_M)
    parser.add_argument("--engine-accel", type=float, default=DEFAULT_ENGINE_ACCEL)
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED)
    args = parser.parse_args()

    print(f"Centreline : {args.centreline}")
    print(f"Items      : {args.items}")
    print(f"Output     : {args.output}")
    print(f"mu={args.mu}  margin={args.margin}m  engine_accel={args.engine_accel}m/s^2")
    print()

    points = build_physics_raceline(
        args.centreline,
        args.items,
        _HIGHLIGHT_IDS,
        mu=args.mu,
        margin_m=args.margin,
        engine_accel_mps2=args.engine_accel,
        max_speed_mps=args.max_speed,
    )

    # Estimated lap time
    lap_time_s = 0.0
    for i, pt in enumerate(points):
        j = (i + 1) % len(points)
        dx = points[j]["x"] - pt["x"]
        dy = points[j]["y"] - pt["y"]
        ds = math.sqrt(dx * dx + dy * dy)
        v_avg = (pt["speed_mps"] + points[j]["speed_mps"]) / 2.0
        if v_avg > 0:
            lap_time_s += ds / v_avg

    speeds = [pt["speed_mps"] for pt in points]
    lateral_errs = [pt["static_lateral_error_m"] for pt in points]

    print()
    print(f"Points          : {len(points)}")
    print(f"Estimated lap   : {lap_time_s:.1f} s  ({lap_time_s / 60:.2f} min)")
    print(f"Speed avg       : {sum(speeds) / len(speeds) * 3.6:.1f} km/h")
    print(f"Speed max       : {max(speeds) * 3.6:.1f} km/h")
    print(f"Speed min       : {min(speeds) * 3.6:.1f} km/h")
    print(f"Max lateral off : {max(abs(e) for e in lateral_errs):.2f} m from centreline")
    print()

    from beamng_rl.track.ai_raceline import write_raceline_json

    write_raceline_json(
        args.output,
        track="hirochi_raceway",
        source=f"physics_raceline_mu{args.mu}_margin{args.margin}",
        points=points,
        metadata={
            "mu": args.mu,
            "margin_m": args.margin,
            "engine_accel_mps2": args.engine_accel,
            "estimated_lap_time_s": round(lap_time_s, 2),
            "avg_speed_mps": round(sum(speeds) / len(speeds), 3),
            "max_speed_mps": round(max(speeds), 3),
        },
    )
    print(f"Written: {args.output}")
