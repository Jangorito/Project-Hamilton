"""Helpers for turning BeamNG AI laps into raceline reference files."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from beamng_rl.track.geometry import resample_polyline
from beamng_rl.track.query_utils import TrackCentreline

Float3 = tuple[float, float, float]


@dataclass(frozen=True)
class TelemetrySample:
    """One sampled state from a live BeamNG AI reference run."""

    time_s: float
    pos: Float3
    velocity: Float3
    speed_mps: float
    heading_rad: float
    raw_progress_m: float
    unwrapped_progress_m: float
    lateral_error_m: float
    damage: float


@dataclass(frozen=True)
class LapCandidate:
    """A projected lap interval and the quality checks used to accept it."""

    lap_number: int
    sample_count: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    average_speed_mps: float
    damage_delta: float
    max_abs_lateral_error_m: float
    max_abs_progress_delta_m: float
    max_stall_s: float
    reverse_distance_m: float
    clean: bool
    rejection_reasons: tuple[str, ...]


def speed_from_velocity(velocity: Sequence[float]) -> float:
    """Return 3D speed in m/s from a velocity vector."""

    arr = np.asarray(velocity, dtype=float).reshape(-1)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(arr[:3]))


def heading_from_direction(direction: Sequence[float]) -> float:
    """Return XY heading angle from a BeamNG direction vector."""

    arr = np.asarray(direction, dtype=float).reshape(-1)
    if arr.shape[0] < 2:
        return 0.0
    if float(np.linalg.norm(arr[:2])) <= 1.0e-9:
        return 0.0
    return float(math.atan2(float(arr[1]), float(arr[0])))


def unwrap_progress(
    *,
    raw_progress_m: float,
    previous_raw_progress_m: float | None,
    previous_unwrapped_progress_m: float | None,
    total_lap_length_m: float,
) -> float:
    """Convert closed-loop raw progress into a monotonically continuing value."""

    raw_progress = float(raw_progress_m)
    if previous_raw_progress_m is None or previous_unwrapped_progress_m is None:
        return raw_progress

    delta = raw_progress - float(previous_raw_progress_m)
    half_lap = 0.5 * float(total_lap_length_m)
    if delta < -half_lap:
        delta += float(total_lap_length_m)
    elif delta > half_lap:
        delta -= float(total_lap_length_m)

    return float(previous_unwrapped_progress_m) + delta


def static_track_points(track: TrackCentreline) -> list[dict[str, float]]:
    """Return track samples as JSON point dictionaries with progress metadata."""

    points: list[dict[str, float]] = []
    for index, xyz in enumerate(track.points_xyz):
        progress_m = float(track.cumulative_arc_lengths[index])
        query = track.query(xyz)
        points.append(
            {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "progress_m": progress_m,
                "heading_rad": float(query.track_heading_rad),
                "static_lateral_error_m": 0.0,
            }
        )
    return points


def build_static_speed_raceline(
    track: TrackCentreline,
    lap_samples: Sequence[TelemetrySample],
) -> list[dict[str, float]]:
    """Attach the selected AI lap's speed profile to the static centreline."""

    points = static_track_points(track)
    speed_by_progress = _interpolate_lap_values(
        lap_samples,
        [point["progress_m"] for point in points],
        "speed_mps",
        track.total_lap_length,
    )

    for point, speed_mps in zip(points, speed_by_progress):
        point["speed_mps"] = float(speed_mps)

    return points


def build_driven_speed_raceline(
    track: TrackCentreline,
    lap_samples: Sequence[TelemetrySample],
    *,
    spacing_m: float = 2.0,
) -> list[dict[str, float]]:
    """Resample the selected AI lap's actual trajectory and attach speed data."""

    if len(lap_samples) < 2:
        return []

    raw_points = [sample.pos for sample in lap_samples]
    resampled_points = resample_polyline(raw_points, spacing=spacing_m)
    if len(resampled_points) < 2:
        return []

    sample_distances = _cumulative_distances(raw_points)
    resampled_distances = _cumulative_distances(resampled_points)
    speeds = np.asarray([sample.speed_mps for sample in lap_samples], dtype=float)
    interp_speeds = np.interp(resampled_distances, sample_distances, speeds)

    output: list[dict[str, float]] = []
    for index, (point, speed_mps) in enumerate(zip(resampled_points, interp_speeds)):
        heading_rad = _heading_from_resampled_point(resampled_points, index)
        query = track.query(point, vehicle_heading_rad=heading_rad)
        output.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "progress_m": float(query.lap_progress),
                "speed_mps": float(speed_mps),
                "heading_rad": float(heading_rad),
                "static_lateral_error_m": float(query.signed_lateral_error),
            }
        )

    return output


def segment_laps(
    samples: Sequence[TelemetrySample],
    total_lap_length_m: float,
    *,
    warmup_laps: int = 1,
    max_damage_delta: float = 1.0,
    max_abs_lateral_error_m: float = 10.0,
    max_abs_progress_delta_m: float = 50.0,
    max_stall_s: float = 3.0,
    min_progress_delta_m: float = 0.05,
    max_reverse_distance_m: float = 5.0,
) -> list[LapCandidate]:
    """Split telemetry into completed projected laps and mark clean candidates."""

    if len(samples) < 2:
        return []

    total = float(total_lap_length_m)
    first_full_lap = int(math.ceil(samples[0].unwrapped_progress_m / total))
    last_full_lap = int(math.floor(samples[-1].unwrapped_progress_m / total)) - 1
    candidates: list[LapCandidate] = []

    for lap_number in range(first_full_lap, last_full_lap + 1):
        start_progress = lap_number * total
        end_progress = (lap_number + 1) * total
        lap_samples = [
            sample
            for sample in samples
            if start_progress <= sample.unwrapped_progress_m <= end_progress
        ]
        if len(lap_samples) < 3:
            continue

        start_time = _interpolate_by_unwrapped_progress(samples, start_progress, "time_s")
        end_time = _interpolate_by_unwrapped_progress(samples, end_progress, "time_s")
        start_damage = _interpolate_by_unwrapped_progress(samples, start_progress, "damage")
        end_damage = _interpolate_by_unwrapped_progress(samples, end_progress, "damage")
        duration = float(end_time - start_time)

        progress_deltas: list[float] = []
        stall_run_s = 0.0
        max_stall_run_s = 0.0
        reverse_distance = 0.0
        for prev, curr in zip(lap_samples, lap_samples[1:]):
            delta_progress = curr.unwrapped_progress_m - prev.unwrapped_progress_m
            delta_time = max(0.0, curr.time_s - prev.time_s)
            progress_deltas.append(delta_progress)
            if delta_progress < min_progress_delta_m:
                stall_run_s += delta_time
            else:
                stall_run_s = 0.0
            max_stall_run_s = max(max_stall_run_s, stall_run_s)
            if delta_progress < 0.0:
                reverse_distance += -delta_progress

        max_lateral = max(abs(sample.lateral_error_m) for sample in lap_samples)
        max_delta = max(abs(delta) for delta in progress_deltas) if progress_deltas else 0.0
        damage_delta = float(end_damage - start_damage)
        avg_speed = total / duration if duration > 0.0 else 0.0

        rejection_reasons: list[str] = []
        if lap_number < first_full_lap + int(warmup_laps):
            rejection_reasons.append("warmup_lap")
        if duration <= 0.0 or not math.isfinite(duration):
            rejection_reasons.append("invalid_duration")
        if damage_delta > max_damage_delta:
            rejection_reasons.append("damage")
        if max_lateral > max_abs_lateral_error_m:
            rejection_reasons.append("lateral_deviation")
        if max_delta > max_abs_progress_delta_m:
            rejection_reasons.append("progress_jump")
        if max_stall_run_s > max_stall_s:
            rejection_reasons.append("stalled")
        if reverse_distance > max_reverse_distance_m:
            rejection_reasons.append("reverse_progress")

        candidates.append(
            LapCandidate(
                lap_number=lap_number,
                sample_count=len(lap_samples),
                start_time_s=float(start_time),
                end_time_s=float(end_time),
                duration_s=float(duration),
                average_speed_mps=float(avg_speed),
                damage_delta=damage_delta,
                max_abs_lateral_error_m=float(max_lateral),
                max_abs_progress_delta_m=float(max_delta),
                max_stall_s=float(max_stall_run_s),
                reverse_distance_m=float(reverse_distance),
                clean=not rejection_reasons,
                rejection_reasons=tuple(rejection_reasons),
            )
        )

    return candidates


def select_fastest_clean_lap(candidates: Sequence[LapCandidate]) -> LapCandidate | None:
    """Pick the shortest-duration clean lap."""

    clean = [candidate for candidate in candidates if candidate.clean]
    if not clean:
        return None
    return min(clean, key=lambda candidate: candidate.duration_s)


def samples_for_lap(
    samples: Sequence[TelemetrySample],
    lap: LapCandidate,
    total_lap_length_m: float,
) -> list[TelemetrySample]:
    """Return the raw samples belonging to a lap candidate."""

    start = lap.lap_number * float(total_lap_length_m)
    end = (lap.lap_number + 1) * float(total_lap_length_m)
    return [sample for sample in samples if start <= sample.unwrapped_progress_m <= end]


def write_raceline_json(
    output_path: Path,
    *,
    track: str,
    source: str,
    points: Sequence[Mapping[str, float]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a TrackCentreline-compatible raceline JSON file."""

    payload: dict[str, Any] = {
        "track": track,
        "source": source,
        "point_count": len(points),
        "points": [dict(point) for point in points],
    }
    if metadata:
        payload.update(dict(metadata))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_telemetry_csv(output_path: Path, samples: Iterable[TelemetrySample]) -> None:
    """Write raw AI telemetry samples to CSV."""

    rows = []
    for sample in samples:
        row = asdict(sample)
        row.update(
            {
                "x": sample.pos[0],
                "y": sample.pos[1],
                "z": sample.pos[2],
                "vel_x": sample.velocity[0],
                "vel_y": sample.velocity[1],
                "vel_z": sample.velocity[2],
            }
        )
        row.pop("pos")
        row.pop("velocity")
        rows.append(row)

    fieldnames = [
        "time_s",
        "x",
        "y",
        "z",
        "vel_x",
        "vel_y",
        "vel_z",
        "speed_mps",
        "heading_rad",
        "raw_progress_m",
        "unwrapped_progress_m",
        "lateral_error_m",
        "damage",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_lap_summary_json(
    output_path: Path,
    *,
    candidates: Sequence[LapCandidate],
    selected_lap: LapCandidate | None,
    metadata: Mapping[str, Any],
) -> None:
    """Write per-run lap acceptance diagnostics."""

    payload = {
        **dict(metadata),
        "selected_lap": asdict(selected_lap) if selected_lap is not None else None,
        "laps": [asdict(candidate) for candidate in candidates],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _interpolate_lap_values(
    lap_samples: Sequence[TelemetrySample],
    progress_values_m: Sequence[float],
    field_name: str,
    total_lap_length_m: float,
) -> np.ndarray:
    if not lap_samples:
        return np.zeros(len(progress_values_m), dtype=float)

    lap_start = math.floor(lap_samples[0].unwrapped_progress_m / total_lap_length_m)
    lap_start_progress = lap_start * float(total_lap_length_m)
    xp = np.asarray(
        [sample.unwrapped_progress_m - lap_start_progress for sample in lap_samples],
        dtype=float,
    )
    fp = np.asarray([getattr(sample, field_name) for sample in lap_samples], dtype=float)
    xp, fp = _strictly_increasing_series(xp, fp)
    if xp.size == 0:
        return np.zeros(len(progress_values_m), dtype=float)
    return np.interp(np.asarray(progress_values_m, dtype=float), xp, fp)


def _interpolate_by_unwrapped_progress(
    samples: Sequence[TelemetrySample],
    progress_m: float,
    field_name: str,
) -> float:
    xp = np.asarray([sample.unwrapped_progress_m for sample in samples], dtype=float)
    fp = np.asarray([getattr(sample, field_name) for sample in samples], dtype=float)
    xp, fp = _strictly_increasing_series(xp, fp)
    if xp.size == 0:
        return 0.0
    return float(np.interp(float(progress_m), xp, fp))


def _strictly_increasing_series(xp: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(xp)
    xp_sorted = xp[order]
    fp_sorted = fp[order]
    keep_indices: list[int] = []
    previous: float | None = None
    for index, value in enumerate(xp_sorted):
        value = float(value)
        if previous is None or value > previous + 1.0e-9:
            keep_indices.append(index)
            previous = value
    if not keep_indices:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    keep = np.asarray(keep_indices, dtype=int)
    return xp_sorted[keep], fp_sorted[keep]


def _cumulative_distances(points: Sequence[Float3]) -> np.ndarray:
    if not points:
        return np.asarray([], dtype=float)

    distances = [0.0]
    for prev, curr in zip(points, points[1:]):
        distances.append(
            distances[-1]
            + float(
                math.dist(
                    (float(prev[0]), float(prev[1]), float(prev[2])),
                    (float(curr[0]), float(curr[1]), float(curr[2])),
                )
            )
        )
    return np.asarray(distances, dtype=float)


def _heading_from_resampled_point(points: Sequence[Float3], index: int) -> float:
    if len(points) < 2:
        return 0.0
    if index <= 0:
        prev_point, next_point = points[0], points[1]
    elif index >= len(points) - 1:
        prev_point, next_point = points[-2], points[-1]
    else:
        prev_point, next_point = points[index - 1], points[index + 1]

    dx = float(next_point[0]) - float(prev_point[0])
    dy = float(next_point[1]) - float(prev_point[1])
    if abs(dx) <= 1.0e-9 and abs(dy) <= 1.0e-9:
        return 0.0
    return float(math.atan2(dy, dx))
