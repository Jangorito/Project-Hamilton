from __future__ import annotations

import csv
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


STEPS_PER_ACTION = 30
MAX_STEPS = 200
OUTPUT_DIR = REPO_ROOT / "logs" / "live_rollouts"

CSV_FIELDS = [
    "step",
    "action_steering",
    "action_throttle_brake",
    "control_steering",
    "control_throttle",
    "control_brake",
    "raw_progress_m",
    "raw_progress_ratio",
    "episode_progress_m",
    "progress_delta_m",
    "forward_speed_mps",
    "speed_reward",
    "progress_reward",
    "heading_error_rad",
    "heading_penalty",
    "lateral_error_m",
    "lateral_penalty",
    "off_track_penalty",
    "reverse_progress_penalty",
    "stuck_penalty",
    "total_reward",
    "terminated",
    "truncated",
    "termination_reason",
    "vehicle_x",
    "vehicle_y",
    "vehicle_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
]


def action_for_step(step_index: int) -> np.ndarray:
    """Return a deterministic action for a zero-based rollout step index."""

    if step_index < 5:
        return np.asarray([0.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, 0.6], dtype=np.float32)


def _blankable_float(value: Any) -> float | str:
    if value is None:
        return ""

    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _xyz(value: Any) -> tuple[float | str, float | str, float | str]:
    if isinstance(value, Mapping):
        values = (value.get("x"), value.get("y"), value.get("z"))
    elif value is None or isinstance(value, (str, bytes)):
        values = (None, None, None)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (None, None, None)

    padded = tuple(values[:3]) + (None,) * max(0, 3 - len(values[:3]))
    return (
        _blankable_float(padded[0]),
        _blankable_float(padded[1]),
        _blankable_float(padded[2]),
    )


def _build_csv_row(
    *,
    step_number: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    reward_info = info["reward"]
    control = info["control"]

    vehicle_state = info.get("vehicle_state", {})
    if not isinstance(vehicle_state, Mapping):
        vehicle_state = {}

    vehicle_pos = vehicle_state.get("pos", info.get("vehicle_pos"))
    vehicle_velocity = vehicle_state.get("velocity", info.get("vehicle_velocity"))
    vehicle_x, vehicle_y, vehicle_z = _xyz(vehicle_pos)
    velocity_x, velocity_y, velocity_z = _xyz(vehicle_velocity)

    return {
        "step": step_number,
        "action_steering": float(action[0]),
        "action_throttle_brake": float(action[1]),
        "control_steering": float(control["steering"]),
        "control_throttle": float(control["throttle"]),
        "control_brake": float(control["brake"]),
        "raw_progress_m": float(info["raw_progress_m"]),
        "raw_progress_ratio": float(info["raw_progress_ratio"]),
        "episode_progress_m": float(info["episode_progress_m"]),
        "progress_delta_m": float(reward_info["progress_delta_m"]),
        "forward_speed_mps": float(reward_info["forward_speed_mps"]),
        "speed_reward": float(reward_info["speed_reward"]),
        "progress_reward": float(reward_info["progress_reward"]),
        "heading_error_rad": float(reward_info["heading_error_rad"]),
        "heading_penalty": float(reward_info["heading_penalty"]),
        "lateral_error_m": float(reward_info["lateral_error_m"]),
        "lateral_penalty": float(reward_info["lateral_penalty"]),
        "off_track_penalty": float(reward_info["off_track_penalty"]),
        "reverse_progress_penalty": float(reward_info["reverse_progress_penalty"]),
        "stuck_penalty": float(reward_info["stuck_penalty"]),
        "total_reward": float(reward_info.get("total_reward", reward)),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": str(info["termination_reason"]),
        "vehicle_x": vehicle_x,
        "vehicle_y": vehicle_y,
        "vehicle_z": vehicle_z,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "velocity_z": velocity_z,
    }


def _print_step(row: Mapping[str, Any]) -> None:
    print(
        f"step={row['step']} "
        f"episode_progress_m={row['episode_progress_m']:.3f} "
        f"progress_delta_m={row['progress_delta_m']:.4f} "
        f"forward_speed_mps={row['forward_speed_mps']:.3f} "
        f"lateral_error_m={row['lateral_error_m']:.3f} "
        f"reward={row['total_reward']:.4f} "
        f"termination_reason={row['termination_reason']}"
    )


def _print_live_failure_hints() -> None:
    print()
    print("Live BeamNG rollout setup failed. Helpful checks:")
    print("- check BeamNG.tech/BeamNGpy install")
    print("- check BEAMNG_HOME")
    print("- check hirochi_raceway")
    print("- check no existing BeamNG process conflict")


def main() -> None:
    centreline_path = (
        REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"live_rollout_{timestamp}.csv"

    env: BeamNGRacingEnv | None = None
    try:
        try:
            env = BeamNGRacingEnv(
                centreline_path,
                use_mock=False,
                launch_beamng=True,
                live_spawn_mode="bootstrap",
                vehicle_id="ego_vehicle",
                steps_per_action=STEPS_PER_ACTION,
            )
            _, reset_info = env.reset()
        except Exception:
            _print_live_failure_hints()
            raise

        start_episode_progress_m = float(reset_info["episode_progress_m"])
        final_episode_progress_m = start_episode_progress_m
        max_forward_speed_mps = 0.0
        final_termination_reason = "none"

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for step_index in range(MAX_STEPS):
                step_number = step_index + 1
                action = action_for_step(step_index)
                _, reward, terminated, truncated, info = env.step(action)
                row = _build_csv_row(
                    step_number=step_number,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )

                writer.writerow(row)
                _print_step(row)

                final_episode_progress_m = float(row["episode_progress_m"])
                max_forward_speed_mps = max(
                    max_forward_speed_mps,
                    float(row["forward_speed_mps"]),
                )
                final_termination_reason = str(row["termination_reason"])

                if terminated or truncated:
                    break

        total_episode_progress_m = (
            final_episode_progress_m - start_episode_progress_m
        )

        print()
        print("Summary:")
        print(f"CSV path: {csv_path}")
        print(f"total episode progress: {total_episode_progress_m:.3f} m")
        print(f"max forward speed: {max_forward_speed_mps:.3f} m/s")
        print(f"final termination reason: {final_termination_reason}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
