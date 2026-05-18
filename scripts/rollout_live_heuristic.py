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


STEPS_PER_ACTION = 15
MAX_STEPS = 300
MAX_EPISODE_STEPS = 500
CENTRELINE_PATH = (
    REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
)
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
    "lateral_error_m",
    "heading_error_rad",
    "total_reward",
    "terminated",
    "truncated",
    "termination_reason",
]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _info_float(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _float_or_default(info.get(key), default)


def _reward_float(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    reward_info = _as_mapping(info.get("reward"))
    return _float_or_default(reward_info.get(key), default)


def heuristic_action(
    info: dict[str, Any],
    previous_action: np.ndarray | None,
) -> np.ndarray:
    """Return a simple diagnostic control action from the latest env info."""

    lateral_error_m = _info_float(info, "lateral_error_m")
    heading_error_rad = _info_float(info, "heading_error_rad")
    forward_speed_mps = _reward_float(info, "forward_speed_mps")

    # This script is diagnostic: these signs assume the current env convention.
    # If the live run steers away from the track, flip/tune these gains first.
    steering = (-0.9 * heading_error_rad) + (-0.08 * lateral_error_m)
    steering = float(np.clip(steering, -1.0, 1.0))

    if abs(heading_error_rad) > 0.5 or abs(lateral_error_m) > 6.0:
        throttle_brake = 0.2
    else:
        throttle_brake = 0.45

    # Reserved for quick follow-up experiments such as speed caps or smoothing.
    _ = previous_action, forward_speed_mps

    return np.asarray([steering, throttle_brake], dtype=np.float32)


def _build_csv_row(
    *,
    step_number: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    reward_info = _as_mapping(info.get("reward"))
    control = _as_mapping(info.get("control"))

    return {
        "step": step_number,
        "action_steering": float(action[0]),
        "action_throttle_brake": float(action[1]),
        "control_steering": _float_or_default(control.get("steering")),
        "control_throttle": _float_or_default(control.get("throttle")),
        "control_brake": _float_or_default(control.get("brake")),
        "raw_progress_m": _info_float(info, "raw_progress_m"),
        "raw_progress_ratio": _info_float(info, "raw_progress_ratio"),
        "episode_progress_m": _info_float(info, "episode_progress_m"),
        "progress_delta_m": _float_or_default(reward_info.get("progress_delta_m")),
        "forward_speed_mps": _float_or_default(reward_info.get("forward_speed_mps")),
        "lateral_error_m": _info_float(info, "lateral_error_m"),
        "heading_error_rad": _info_float(info, "heading_error_rad"),
        "total_reward": _float_or_default(reward_info.get("total_reward"), reward),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": str(info.get("termination_reason", "none")),
    }


def _print_step(row: Mapping[str, Any]) -> None:
    print(
        f"step={row['step']} "
        f"action_steering={row['action_steering']:.3f} "
        f"action_throttle_brake={row['action_throttle_brake']:.3f} "
        f"episode_progress_m={row['episode_progress_m']:.3f} "
        f"progress_delta_m={row['progress_delta_m']:.4f} "
        f"forward_speed_mps={row['forward_speed_mps']:.3f} "
        f"lateral_error_m={row['lateral_error_m']:.3f} "
        f"heading_error_rad={row['heading_error_rad']:.4f} "
        f"reward={row['total_reward']:.4f} "
        f"termination_reason={row['termination_reason']}"
    )


def _print_live_failure_hints() -> None:
    print()
    print("Live BeamNG heuristic rollout failed. Helpful checks:")
    print("- check BeamNG.tech install path / BEAMNG_HOME")
    print("- check BeamNGpy is installed in the venv")
    print("- check hirochi_raceway exists")
    print("- check BeamNG is not blocked by another process")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"live_heuristic_{timestamp}.csv"

    env: BeamNGRacingEnv | None = None
    try:
        try:
            env = BeamNGRacingEnv(
                CENTRELINE_PATH,
                use_mock=False,
                launch_beamng=True,
                live_spawn_mode="bootstrap",
                vehicle_id="ego_vehicle",
                steps_per_action=STEPS_PER_ACTION,
                max_episode_steps=MAX_EPISODE_STEPS,
            )
            _, latest_info = env.reset()
        except Exception:
            _print_live_failure_hints()
            raise

        start_episode_progress_m = _info_float(latest_info, "episode_progress_m")
        final_episode_progress_m = start_episode_progress_m
        final_lateral_error_m = _info_float(latest_info, "lateral_error_m")
        final_heading_error_rad = _info_float(latest_info, "heading_error_rad")
        max_forward_speed_mps = 0.0
        final_termination_reason = "none"
        previous_action: np.ndarray | None = None

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for step_index in range(MAX_STEPS):
                step_number = step_index + 1
                action = heuristic_action(latest_info, previous_action)
                _, reward, terminated, truncated, latest_info = env.step(action)

                row = _build_csv_row(
                    step_number=step_number,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=latest_info,
                )

                writer.writerow(row)
                _print_step(row)

                final_episode_progress_m = float(row["episode_progress_m"])
                final_lateral_error_m = float(row["lateral_error_m"])
                final_heading_error_rad = float(row["heading_error_rad"])
                max_forward_speed_mps = max(
                    max_forward_speed_mps,
                    float(row["forward_speed_mps"]),
                )
                final_termination_reason = str(row["termination_reason"])
                previous_action = action

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
        print(f"final lateral error: {final_lateral_error_m:.3f} m")
        print(f"final heading error: {final_heading_error_rad:.4f} rad")
        print(f"final termination reason: {final_termination_reason}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
