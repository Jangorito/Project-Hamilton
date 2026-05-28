"""Collect solo BeamNG AI reference laps and export raceline speed profiles.

Examples:
    python scripts/eval/collect_ai_raceline.py --dry-run
    python scripts/eval/collect_ai_raceline.py --vehicles both --aggressions 0.9 1.0
    python scripts/eval/collect_ai_raceline.py --vehicles etkc --aggressions 1.0 --max-laps 3
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.bootstrap.beamng_setup import (
    HOST,
    PORT,
    apply_low_graphics_preset,
    apply_shadow_disabling,
    build_hirochi_etkc_scenario,
    build_hirochi_sbr_scenario,
    resolve_beamng_home_from_path_or_env,
)
from beamng_rl.track.ai_raceline import (
    TelemetrySample,
    build_driven_speed_raceline,
    build_static_speed_raceline,
    heading_from_direction,
    samples_for_lap,
    segment_laps,
    select_fastest_clean_lap,
    speed_from_velocity,
    static_track_points,
    unwrap_progress,
    write_lap_summary_json,
    write_raceline_json,
    write_telemetry_csv,
)
from beamng_rl.track.query_utils import TrackCentreline


CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
OUTPUT_RACELINE_DIR = REPO_ROOT / "data" / "hirochi_track"
LOG_ROOT = REPO_ROOT / "logs" / "ai_raceline"
VEHICLE_BUILDERS = {
    "etkc": build_hirochi_etkc_scenario,
    "sbr": build_hirochi_sbr_scenario,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect solo BeamNG AI raceline telemetry for Hirochi Raceway."
    )
    parser.add_argument(
        "--vehicles",
        nargs="+",
        default=["both"],
        choices=["both", "etkc", "sbr"],
        help="Vehicle(s) to collect. Default: both.",
    )
    parser.add_argument(
        "--aggressions",
        nargs="+",
        type=float,
        default=[0.9, 1.0],
        help="AI aggression values to sweep. Default: 0.9 1.0.",
    )
    parser.add_argument(
        "--racer-skill",
        type=float,
        default=1.0,
        help="BeamNG aiRace-style racerSkill value in [0, 1]. Default: 1.0.",
    )
    parser.add_argument(
        "--randomness-scale",
        type=float,
        default=0.0,
        help="BeamNG aiRace-style randomization scale. Default: 0.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=["curvature", "constant"],
        default="curvature",
        help="Speed targets sent to vehicle.ai.set_line. Default: curvature.",
    )
    parser.add_argument(
        "--line-speed",
        type=float,
        default=28.0,
        help="Constant speed in m/s when --speed-profile constant is used. Default: 28.",
    )
    parser.add_argument(
        "--max-line-speed",
        type=float,
        default=32.0,
        help="Maximum curvature-profile line speed in m/s. Default: 32.",
    )
    parser.add_argument(
        "--min-line-speed",
        type=float,
        default=8.0,
        help="Minimum curvature-profile line speed in m/s. Default: 8.",
    )
    parser.add_argument(
        "--curvature-speed-scale",
        type=float,
        default=700.0,
        help="Speed reduction per 1/m of upcoming curvature. Default: 700.",
    )
    parser.add_argument(
        "--start-ramp-m",
        type=float,
        default=150.0,
        help="Distance used to ramp from min speed after the start line. Default: 150.",
    )
    parser.add_argument(
        "--retry-speed-factors",
        nargs="+",
        type=float,
        default=[1.0, 0.85, 0.70],
        help="Per-attempt multipliers for max/constant line speed. Default: 1.0 0.85 0.70.",
    )
    parser.add_argument(
        "--retry-aggression-step",
        type=float,
        default=0.10,
        help="Aggression reduction after each failed attempt. Default: 0.10.",
    )
    parser.add_argument(
        "--warmup-laps",
        type=int,
        default=1,
        help="Completed projected laps to discard before selecting clean laps. Default: 1.",
    )
    parser.add_argument(
        "--clean-laps",
        type=int,
        default=1,
        help="Number of clean laps to collect before stopping when possible. Default: 1.",
    )
    parser.add_argument(
        "--max-laps",
        type=int,
        default=4,
        help="Maximum projected laps to run per vehicle/aggression combo. Default: 4.",
    )
    parser.add_argument(
        "--max-sim-time",
        type=float,
        default=600.0,
        help="Maximum simulated seconds per vehicle/aggression combo. Default: 600.",
    )
    parser.add_argument(
        "--abort-lateral-error",
        type=float,
        default=12.0,
        help="Abort and retry an attempt if lateral error exceeds this many metres. Default: 12.",
    )
    parser.add_argument(
        "--abort-damage",
        type=float,
        default=25.0,
        help="Abort and retry an attempt if total damage exceeds this value. Default: 25.",
    )
    parser.add_argument(
        "--abort-stall-s",
        type=float,
        default=5.0,
        help="Abort and retry after this many seconds with almost no progress. Default: 5.",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=20.0,
        help="Telemetry sample rate in deterministic sim time. Default: 20 Hz.",
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=2.0,
        help="Driven trajectory export spacing in metres. Default: 2.",
    )
    parser.add_argument(
        "--centreline",
        type=Path,
        default=CENTRELINE_PATH,
        help=f"Canonical Hirochi centreline JSON. Default: {CENTRELINE_PATH}.",
    )
    parser.add_argument(
        "--output-raceline-dir",
        type=Path,
        default=OUTPUT_RACELINE_DIR,
        help=f"Directory for final raceline JSON exports. Default: {OUTPUT_RACELINE_DIR}.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for raw telemetry and summaries. Default: logs/ai_raceline/<timestamp>.",
    )
    parser.add_argument("--beamng-home", type=Path, default=None)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Connect to an already-running BeamNG.tech instead of launching one.",
    )
    parser.add_argument(
        "--nogfx",
        action="store_true",
        help="Launch BeamNG with nogpu=True for headless physics-only collection.",
    )
    parser.add_argument("--low-graphics", action="store_true")
    parser.add_argument("--disable-shadows", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate static route/export assumptions without launching BeamNG.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    track = TrackCentreline.from_json(args.centreline, closed_loop=True)
    vehicles = _resolve_vehicles(args.vehicles)
    output_log_dir = args.log_dir or LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Loaded centreline: {args.centreline}")
    print(f"Track length: {track.total_lap_length:.2f} m")
    print(f"Vehicles: {', '.join(vehicles)}")
    print(f"Aggressions: {', '.join(_aggression_label(v) for v in args.aggressions)}")
    print(f"Log dir: {output_log_dir}")

    if args.dry_run:
        points = static_track_points(track)
        print(f"Dry run OK: static route has {len(points)} points.")
        print("No BeamNG process launched and no files written.")
        return

    beamng_home = resolve_beamng_home_from_path_or_env(args.beamng_home)
    print(f"Using BeamNG home: {beamng_home}")

    from beamngpy import BeamNGpy, set_up_simple_logging

    set_up_simple_logging()
    bng = BeamNGpy(
        args.host,
        args.port,
        home=str(beamng_home) if not args.connect_existing else None,
        quit_on_close=not args.connect_existing,
        nogpu=args.nogfx,
    )

    try:
        bng.open(launch=not args.connect_existing)
        _configure_beamng(bng, args)

        for vehicle_model in vehicles:
            for aggression in args.aggressions:
                _run_combo(
                    bng,
                    args=args,
                    track=track,
                    vehicle_model=vehicle_model,
                    aggression=float(aggression),
                    output_log_dir=output_log_dir,
                )
    finally:
        bng.close()


def _run_combo(
    bng: Any,
    *,
    args: argparse.Namespace,
    track: TrackCentreline,
    vehicle_model: str,
    aggression: float,
    output_log_dir: Path,
) -> None:
    combo_label = f"{vehicle_model}_aggr_{_aggression_label(aggression)}"
    print()
    print(f"=== Collecting {combo_label} ===")

    from beamngpy.sensors import Damage

    combo_log_dir = output_log_dir / combo_label
    selected_lap = None
    selected_samples: list[TelemetrySample] = []
    selected_attempt_metadata: dict[str, Any] = {}

    speed_factors = [float(value) for value in args.retry_speed_factors]
    for attempt_index, speed_factor in enumerate(speed_factors, start=1):
        effective_aggression = max(
            0.50,
            float(aggression) - (attempt_index - 1) * float(args.retry_aggression_step),
        )
        attempt_label = f"attempt_{attempt_index}_speed_{_aggression_label(speed_factor)}"
        attempt_log_dir = combo_log_dir / attempt_label
        print(
            f"-- {attempt_label}: speed_factor={speed_factor:.2f}, "
            f"effective_aggression={effective_aggression:.2f}"
        )

        scenario_name = f"ai_raceline_{combo_label}_{attempt_index}_{os.getpid()}"
        scenario, vehicle = VEHICLE_BUILDERS[vehicle_model](
            bng,
            vehicle_id="ai_reference",
            scenario_instance_name=scenario_name,
        )

        vehicle.attach_sensor("damage", Damage())
        bng.scenario.load(scenario)
        _try_hide_hud(bng)
        bng.scenario.start()
        try:
            bng.pause()
        except Exception:
            pass

        _step(bng, 10)
        _poll_vehicle_state(vehicle)

        ai_line = _build_repeated_ai_line(
            track,
            speed_profile=str(args.speed_profile),
            constant_speed_mps=float(args.line_speed) * speed_factor,
            max_speed_mps=float(args.max_line_speed) * speed_factor,
            min_speed_mps=float(args.min_line_speed),
            curvature_speed_scale=float(args.curvature_speed_scale),
            start_ramp_m=float(args.start_ramp_m),
            laps=max(int(args.max_laps) + 1, 2),
        )
        print(
            "Line speed range: "
            f"{min(point['speed'] for point in ai_line):.1f}-"
            f"{max(point['speed'] for point in ai_line):.1f} m/s"
        )
        vehicle.ai.set_line(ai_line, cling=False)
        _apply_ai_race_parameters(
            vehicle,
            aggression=effective_aggression,
            racer_skill=float(args.racer_skill),
            randomness_scale=float(args.randomness_scale),
            awareness=False,
            rubberband=False,
        )
        _step(bng, 10)

        samples = _collect_samples(
            bng,
            vehicle,
            track=track,
            sample_hz=float(args.sample_hz),
            warmup_laps=int(args.warmup_laps),
            clean_laps=int(args.clean_laps),
            max_laps=int(args.max_laps),
            max_sim_time_s=float(args.max_sim_time),
            abort_lateral_error_m=float(args.abort_lateral_error),
            abort_damage=float(args.abort_damage),
            abort_stall_s=float(args.abort_stall_s),
        )
        print(f"Collected {len(samples)} samples.")

        telemetry_path = attempt_log_dir / "telemetry.csv"
        write_telemetry_csv(telemetry_path, samples)
        print(f"Wrote telemetry: {telemetry_path}")

        candidates = segment_laps(
            samples,
            track.total_lap_length,
            warmup_laps=int(args.warmup_laps),
        )
        attempt_selected_lap = select_fastest_clean_lap(candidates)
        summary_metadata = {
            "vehicle": vehicle_model,
            "requestedAggression": aggression,
            "effectiveAggression": effective_aggression,
            "attempt": attempt_index,
            "speedFactor": speed_factor,
            "speedProfile": str(args.speed_profile),
            "lineSpeedMps": float(args.line_speed),
            "maxLineSpeedMps": float(args.max_line_speed) * speed_factor,
            "minLineSpeedMps": float(args.min_line_speed),
            "curvatureSpeedScale": float(args.curvature_speed_scale),
            "startRampM": float(args.start_ramp_m),
            "racerSkill": float(args.racer_skill),
            "racerRandomnessScale": float(args.randomness_scale),
            "racerRubberbandMode": False,
            "racerAwareness": False,
            "centreline": str(args.centreline),
            "trackLengthM": float(track.total_lap_length),
        }
        summary_path = attempt_log_dir / "summary.json"
        write_lap_summary_json(
            summary_path,
            candidates=candidates,
            selected_lap=attempt_selected_lap,
            metadata=summary_metadata,
        )
        print(f"Wrote summary: {summary_path}")

        for lap in candidates:
            status = "clean" if lap.clean else ",".join(lap.rejection_reasons)
            print(
                f"lap={lap.lap_number} duration={lap.duration_s:.3f}s "
                f"avg={lap.average_speed_mps:.3f}m/s status={status}"
            )

        if attempt_selected_lap is not None:
            selected_lap = attempt_selected_lap
            selected_samples = samples
            selected_attempt_metadata = summary_metadata
            print(f"Selected lap from {attempt_label}.")
            break

        print(f"No clean lap for {attempt_label}; retrying with safer settings if available.")

    if selected_lap is None:
        print(f"No clean lap found for {combo_label}; final raceline JSONs were not exported.")
        return

    lap_samples = samples_for_lap(selected_samples, selected_lap, track.total_lap_length)
    static_points = build_static_speed_raceline(track, lap_samples)
    driven_points = build_driven_speed_raceline(
        track,
        lap_samples,
        spacing_m=float(args.resample_spacing),
    )

    export_metadata = {
        "vehicle": vehicle_model,
        "requestedAggression": aggression,
        "effectiveAggression": selected_attempt_metadata.get("effectiveAggression", aggression),
        "attempt": selected_attempt_metadata.get("attempt"),
        "speedFactor": selected_attempt_metadata.get("speedFactor"),
        "speedProfile": selected_attempt_metadata.get("speedProfile"),
        "maxLineSpeedMps": selected_attempt_metadata.get("maxLineSpeedMps"),
        "minLineSpeedMps": selected_attempt_metadata.get("minLineSpeedMps"),
        "curvatureSpeedScale": selected_attempt_metadata.get("curvatureSpeedScale"),
        "startRampM": selected_attempt_metadata.get("startRampM"),
        "racerSkill": float(args.racer_skill),
        "racerRandomnessScale": float(args.randomness_scale),
        "racerRubberbandMode": False,
        "racerAwareness": False,
        "selectedLap": selected_lap.lap_number,
        "selectedLapTimeS": selected_lap.duration_s,
        "selectedAverageSpeedMps": selected_lap.average_speed_mps,
        "spacing": float(args.resample_spacing),
    }

    static_output = args.output_raceline_dir / f"ai_static_raceline_{combo_label}.json"
    driven_output = args.output_raceline_dir / f"ai_driven_raceline_{combo_label}.json"
    write_raceline_json(
        static_output,
        track="hirochi_raceway",
        source="static_centreline_with_solo_beamng_ai_speed_profile",
        points=static_points,
        metadata=export_metadata,
    )
    write_raceline_json(
        driven_output,
        track="hirochi_raceway",
        source="solo_beamng_ai_driven_trajectory",
        points=driven_points,
        metadata=export_metadata,
    )
    print(f"Exported static raceline: {static_output}")
    print(f"Exported driven raceline: {driven_output}")


def _configure_beamng(bng: Any, args: argparse.Namespace) -> None:
    bng.settings.set_nondeterministic()
    bng.settings.set_deterministic(60)
    try:
        bng.pause()
    except Exception:
        pass

    if args.disable_shadows:
        apply_shadow_disabling(bng)
    if args.low_graphics:
        apply_low_graphics_preset(bng)


def _collect_samples(
    bng: Any,
    vehicle: Any,
    *,
    track: TrackCentreline,
    sample_hz: float,
    warmup_laps: int,
    clean_laps: int,
    max_laps: int,
    max_sim_time_s: float,
    abort_lateral_error_m: float,
    abort_damage: float,
    abort_stall_s: float,
) -> list[TelemetrySample]:
    frames_per_sample = max(1, int(round(60.0 / max(sample_hz, 1.0e-6))))
    dt_s = frames_per_sample / 60.0
    samples: list[TelemetrySample] = []
    previous_raw: float | None = None
    previous_unwrapped: float | None = None
    initial_unwrapped: float | None = None
    target_unwrapped: float | None = None
    sim_time_s = 0.0
    completed_lap_floor = -1
    stall_run_s = 0.0

    while sim_time_s <= max_sim_time_s:
        _step(bng, frames_per_sample)
        sim_time_s += dt_s
        state = _poll_vehicle_state(vehicle)
        pos = _coerce_float3(state.get("pos"))
        velocity = _coerce_float3(
            _first_present(state, "vel", "velocity"),
            default=(0.0, 0.0, 0.0),
        )
        direction = _coerce_float3(
            _first_present(state, "dir", "direction"),
            default=(1.0, 0.0, 0.0),
        )
        heading_rad = heading_from_direction(direction)
        query = track.query(pos, vehicle_heading_rad=heading_rad)
        raw_progress = float(query.lap_progress)
        unwrapped = unwrap_progress(
            raw_progress_m=raw_progress,
            previous_raw_progress_m=previous_raw,
            previous_unwrapped_progress_m=previous_unwrapped,
            total_lap_length_m=track.total_lap_length,
        )
        progress_delta = 0.0 if previous_unwrapped is None else unwrapped - previous_unwrapped
        if initial_unwrapped is None:
            initial_unwrapped = unwrapped
            target_unwrapped = initial_unwrapped + max_laps * track.total_lap_length

        damage = _read_damage(vehicle)
        sample = TelemetrySample(
            time_s=float(sim_time_s),
            pos=pos,
            velocity=velocity,
            speed_mps=speed_from_velocity(velocity),
            heading_rad=float(heading_rad),
            raw_progress_m=raw_progress,
            unwrapped_progress_m=float(unwrapped),
            lateral_error_m=float(query.signed_lateral_error),
            damage=float(damage),
        )
        samples.append(sample)
        previous_raw = raw_progress
        previous_unwrapped = unwrapped

        if abs(sample.lateral_error_m) > abort_lateral_error_m:
            print(
                "Aborting attempt: lateral error exceeded "
                f"{abort_lateral_error_m:.1f} m ({sample.lateral_error_m:.1f} m)."
            )
            break
        if sample.damage > abort_damage:
            print(
                f"Aborting attempt: damage exceeded {abort_damage:.1f} "
                f"({sample.damage:.1f})."
            )
            break
        if sim_time_s > 5.0 and progress_delta < 0.05:
            stall_run_s += dt_s
        else:
            stall_run_s = 0.0
        if stall_run_s > abort_stall_s:
            print(f"Aborting attempt: stalled for {stall_run_s:.1f} s.")
            break

        if len(samples) % max(1, int(sample_hz * 5)) == 0:
            print(
                f"t={sim_time_s:.1f}s progress={unwrapped:.1f}m "
                f"speed={sample.speed_mps:.1f}m/s lateral={sample.lateral_error_m:.2f}m"
            )

        if target_unwrapped is not None and unwrapped >= target_unwrapped:
            break

        current_floor = int(math.floor(unwrapped / track.total_lap_length))
        if current_floor > completed_lap_floor:
            completed_lap_floor = current_floor
            candidates = segment_laps(
                samples,
                track.total_lap_length,
                warmup_laps=warmup_laps,
            )
            clean_count = sum(1 for candidate in candidates if candidate.clean)
            if clean_count >= clean_laps:
                print(f"Collected requested clean laps: {clean_count}/{clean_laps}")
                break

    return samples


def _apply_ai_race_parameters(
    vehicle: Any,
    *,
    aggression: float,
    racer_skill: float,
    randomness_scale: float,
    awareness: bool,
    rubberband: bool,
) -> None:
    # Match BeamNG's raceAiParameters.lua defaults with randomness disabled by default.
    skill = max(0.0, min(1.0, float(racer_skill)))
    scale = max(0.0, min(2.0, float(randomness_scale)))
    base_edge_dist = 1.5 - skill * 1.5
    base_turn_force = 4.0 * math.pow(skill, 1.5)
    base_awareness_force = 0.25 * math.pow(skill, 1.5)

    aggression_value = max(0.25, min(2.0, float(aggression) + _random_offset(scale, 0.05)))
    edge_dist = max(0.0, min(1.5, base_edge_dist + _random_offset(scale, 0.25)))
    turn_force = max(0.01, min(10.0, base_turn_force + _random_offset(scale, 2.0)))
    awareness_force = max(0.01, min(0.5, base_awareness_force + _random_offset(scale, 0.15)))

    params = {
        "edgeDist": edge_dist,
        "turnForceCoef": turn_force,
        "awarenessForceCoef": awareness_force,
    }
    vehicle.queue_lua_command(f"ai.setAggression({aggression_value:.6f})")
    vehicle.queue_lua_command(f"ai.setParameters({_lua_table(params)})")
    vehicle.queue_lua_command("ai.setRacing(true)")
    vehicle.queue_lua_command(f"ai.setAvoidCars(\"{'on' if awareness else 'off'}\")")
    vehicle.queue_lua_command(f"ai.setAggressionMode(\"{'rubberBand' if rubberband else 'off'}\")")
    print(
        "Applied AI params: "
        f"aggression={aggression_value:.3f}, skill={skill:.3f}, randomness={scale:.3f}, "
        f"edgeDist={edge_dist:.3f}, turnForceCoef={turn_force:.3f}, "
        f"awarenessForceCoef={awareness_force:.3f}, rubberband={rubberband}"
    )


def _build_repeated_ai_line(
    track: TrackCentreline,
    *,
    speed_profile: str,
    constant_speed_mps: float,
    max_speed_mps: float,
    min_speed_mps: float,
    curvature_speed_scale: float,
    start_ramp_m: float,
    laps: int,
) -> list[dict[str, Any]]:
    points = [tuple(map(float, xyz)) for xyz in track.points_xyz]
    speeds = [
        _line_speed_for_progress(
            track,
            progress_m=float(track.cumulative_arc_lengths[index]),
            speed_profile=speed_profile,
            constant_speed_mps=constant_speed_mps,
            max_speed_mps=max_speed_mps,
            min_speed_mps=min_speed_mps,
            curvature_speed_scale=curvature_speed_scale,
            start_ramp_m=start_ramp_m,
        )
        for index in range(len(points))
    ]
    line: list[dict[str, Any]] = []
    for _ in range(max(1, int(laps))):
        for point, speed in zip(points, speeds):
            line.append({"pos": point, "speed": float(speed)})
    line.append({"pos": points[0], "speed": float(speeds[0])})
    return line


def _line_speed_for_progress(
    track: TrackCentreline,
    *,
    progress_m: float,
    speed_profile: str,
    constant_speed_mps: float,
    max_speed_mps: float,
    min_speed_mps: float,
    curvature_speed_scale: float,
    start_ramp_m: float,
) -> float:
    min_speed = max(0.1, float(min_speed_mps))
    if speed_profile == "constant":
        return max(min_speed, float(constant_speed_mps))

    max_speed = max(min_speed, float(max_speed_mps))
    max_curvature_ahead = max(
        track.curvature_at(progress_m + lookahead_m)
        for lookahead_m in (20.0, 40.0, 80.0)
    )
    target_speed = max(min_speed, max_speed - float(curvature_speed_scale) * max_curvature_ahead)
    if start_ramp_m > 0.0:
        ramp = min(1.0, max(0.0, float(progress_m) / float(start_ramp_m)))
        target_speed = min_speed + (target_speed - min_speed) * ramp
    return float(target_speed)


def _poll_vehicle_state(vehicle: Any) -> dict[str, Any]:
    vehicle.sensors.poll("state", "damage")
    return dict(getattr(vehicle, "state", {}) or {})


def _read_damage(vehicle: Any) -> float:
    try:
        return float(vehicle.sensors["damage"].get("damage", 0.0))
    except Exception:
        return 0.0


def _step(bng: Any, frames: int) -> None:
    bng.step(int(frames), wait=True)


def _try_hide_hud(bng: Any) -> None:
    try:
        bng.ui.hide_hud()
    except Exception:
        pass


def _resolve_vehicles(values: Sequence[str]) -> list[str]:
    selected = list(values)
    if "both" in selected:
        return ["etkc", "sbr"]
    return selected


def _aggression_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "_")


def _coerce_float3(value: Any, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if value is None:
        if default is None:
            raise RuntimeError("Expected a BeamNG XYZ vector, got None.")
        return default

    if isinstance(value, dict):
        raw = (value.get("x"), value.get("y"), value.get("z"))
    else:
        raw = tuple(value)

    if len(raw) < 3:
        if default is None:
            raise RuntimeError(f"Expected at least 3 vector components, got {raw!r}.")
        return default

    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _lua_table(values: dict[str, float]) -> str:
    parts = [f"{key} = {float(value):.6f}" for key, value in values.items()]
    return "{ " + ", ".join(parts) + " }"


def _random_offset(scale: float, amplitude: float) -> float:
    return (random.random() * 2.0 - 1.0) * float(scale) * float(amplitude)


if __name__ == "__main__":
    main()
