from __future__ import annotations

import csv
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


STEPS_PER_ACTION = 10
MAX_EPISODE_STEPS = 200
MAX_STEPS_PER_CASE = 80
HEADING_GAIN = 0.7
LATERAL_GAIN = 0.05

CENTRELINE_PATH = (
    REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
)
OUTPUT_DIR = REPO_ROOT / "logs" / "live_rollouts"

CASES = [
    ("neg_heading_neg_lateral", -1.0, -1.0),
    ("pos_heading_pos_lateral", 1.0, 1.0),
    ("pos_heading_neg_lateral", 1.0, -1.0),
    ("neg_heading_pos_lateral", -1.0, 1.0),
]

CSV_FIELDS = [
    "case_name",
    "heading_sign",
    "lateral_sign",
    "step",
    "action_steering",
    "action_throttle_brake",
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


@dataclass(frozen=True)
class CaseResult:
    case_name: str
    heading_sign: float
    lateral_sign: float
    steps_run: int
    total_episode_progress_m: float
    max_forward_speed_mps: float
    max_abs_lateral_error_m: float
    final_lateral_error_m: float
    final_heading_error_rad: float
    final_termination_reason: str

    @property
    def did_not_terminate_off_track(self) -> bool:
        return self.final_termination_reason != "off_track"


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


def _latest_metric(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if key in info:
        return _info_float(info, key, default)
    return _reward_float(info, key, default)


def sign_sweep_action(
    info: Mapping[str, Any],
    *,
    heading_sign: float,
    lateral_sign: float,
) -> np.ndarray:
    """Return one low-speed diagnostic action for the selected sign case."""

    lateral_error_m = _latest_metric(info, "lateral_error_m")
    heading_error_rad = _latest_metric(info, "heading_error_rad")

    # This is not a final controller. It only probes whether lateral_error_m
    # and heading_error_rad signs match the steering action convention.
    # Copy the best sign case back into rollout_live_heuristic.py afterward.
    steering = (
        heading_sign * HEADING_GAIN * heading_error_rad
        + lateral_sign * LATERAL_GAIN * lateral_error_m
    )
    steering = float(np.clip(steering, -1.0, 1.0))

    throttle_brake = 0.25
    if abs(lateral_error_m) > 6.0 or abs(heading_error_rad) > 0.6:
        throttle_brake = 0.1

    return np.asarray([steering, throttle_brake], dtype=np.float32)


def _build_csv_row(
    *,
    case_name: str,
    heading_sign: float,
    lateral_sign: float,
    step_number: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    reward_info = _as_mapping(info.get("reward"))

    return {
        "case_name": case_name,
        "heading_sign": heading_sign,
        "lateral_sign": lateral_sign,
        "step": step_number,
        "action_steering": float(action[0]),
        "action_throttle_brake": float(action[1]),
        "episode_progress_m": _info_float(info, "episode_progress_m"),
        "progress_delta_m": _float_or_default(reward_info.get("progress_delta_m")),
        "forward_speed_mps": _float_or_default(reward_info.get("forward_speed_mps")),
        "lateral_error_m": _latest_metric(info, "lateral_error_m"),
        "heading_error_rad": _latest_metric(info, "heading_error_rad"),
        "total_reward": _float_or_default(reward_info.get("total_reward"), reward),
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": str(info.get("termination_reason", "none")),
    }


def _print_step(row: Mapping[str, Any]) -> None:
    print(
        f"case={row['case_name']} "
        f"step={row['step']} "
        f"steering={row['action_steering']:.3f} "
        f"episode_progress_m={row['episode_progress_m']:.3f} "
        f"lateral_error_m={row['lateral_error_m']:.3f} "
        f"heading_error_rad={row['heading_error_rad']:.4f} "
        f"reward={row['total_reward']:.4f} "
        f"termination_reason={row['termination_reason']}"
    )


def _print_case_summary(result: CaseResult) -> None:
    print()
    print(f"Case summary: {result.case_name}")
    print(f"total episode progress: {result.total_episode_progress_m:.3f} m")
    print(f"max forward speed: {result.max_forward_speed_mps:.3f} m/s")
    print(f"max abs lateral error: {result.max_abs_lateral_error_m:.3f} m")
    print(f"final lateral error: {result.final_lateral_error_m:.3f} m")
    print(f"final heading error: {result.final_heading_error_rad:.4f} rad")
    print(f"final termination reason: {result.final_termination_reason}")


def _rank_results(results: list[CaseResult]) -> list[CaseResult]:
    return sorted(
        results,
        key=lambda result: (
            result.final_termination_reason == "off_track",
            -result.total_episode_progress_m,
            result.max_abs_lateral_error_m,
        ),
    )


def _print_ranked_summary(results: list[CaseResult]) -> None:
    print()
    print("Ranked summary:")
    header = (
        f"{'rank':<4} "
        f"{'case':<24} "
        f"{'no_offtrack':<11} "
        f"{'progress_m':>10} "
        f"{'max_abs_lat':>11} "
        f"{'final_lat':>10} "
        f"{'final_head':>10} "
        f"{'max_speed':>9} "
        f"{'reason':<18}"
    )
    print(header)
    print("-" * len(header))

    for rank, result in enumerate(_rank_results(results), start=1):
        no_offtrack = "yes" if result.did_not_terminate_off_track else "no"
        print(
            f"{rank:<4} "
            f"{result.case_name:<24} "
            f"{no_offtrack:<11} "
            f"{result.total_episode_progress_m:>10.3f} "
            f"{result.max_abs_lateral_error_m:>11.3f} "
            f"{result.final_lateral_error_m:>10.3f} "
            f"{result.final_heading_error_rad:>10.4f} "
            f"{result.max_forward_speed_mps:>9.3f} "
            f"{result.final_termination_reason:<18}"
        )


def _print_live_failure_hints() -> None:
    print()
    print("Live BeamNG heuristic sign sweep failed. Helpful checks:")
    print("- check BeamNG.tech install path / BEAMNG_HOME")
    print("- check BeamNGpy is installed in the venv")
    print("- check hirochi_raceway exists")
    print("- check BeamNG is not blocked by another process")


def _run_case(
    *,
    env: BeamNGRacingEnv,
    writer: csv.DictWriter,
    csv_file: Any,
    case_name: str,
    heading_sign: float,
    lateral_sign: float,
) -> CaseResult:
    _, latest_info = env.reset()

    start_episode_progress_m = _info_float(latest_info, "episode_progress_m")
    final_episode_progress_m = start_episode_progress_m
    final_lateral_error_m = _latest_metric(latest_info, "lateral_error_m")
    final_heading_error_rad = _latest_metric(latest_info, "heading_error_rad")
    max_abs_lateral_error_m = abs(final_lateral_error_m)
    max_forward_speed_mps = 0.0
    final_termination_reason = "none"
    steps_run = 0

    for step_index in range(MAX_STEPS_PER_CASE):
        step_number = step_index + 1
        action = sign_sweep_action(
            latest_info,
            heading_sign=heading_sign,
            lateral_sign=lateral_sign,
        )
        _, reward, terminated, truncated, latest_info = env.step(action)

        row = _build_csv_row(
            case_name=case_name,
            heading_sign=heading_sign,
            lateral_sign=lateral_sign,
            step_number=step_number,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=latest_info,
        )

        writer.writerow(row)
        csv_file.flush()
        _print_step(row)

        steps_run = step_number
        final_episode_progress_m = float(row["episode_progress_m"])
        final_lateral_error_m = float(row["lateral_error_m"])
        final_heading_error_rad = float(row["heading_error_rad"])
        max_abs_lateral_error_m = max(
            max_abs_lateral_error_m,
            abs(final_lateral_error_m),
        )
        max_forward_speed_mps = max(
            max_forward_speed_mps,
            float(row["forward_speed_mps"]),
        )
        final_termination_reason = str(row["termination_reason"])

        if terminated or truncated:
            break

    return CaseResult(
        case_name=case_name,
        heading_sign=heading_sign,
        lateral_sign=lateral_sign,
        steps_run=steps_run,
        total_episode_progress_m=final_episode_progress_m - start_episode_progress_m,
        max_forward_speed_mps=max_forward_speed_mps,
        max_abs_lateral_error_m=max_abs_lateral_error_m,
        final_lateral_error_m=final_lateral_error_m,
        final_heading_error_rad=final_heading_error_rad,
        final_termination_reason=final_termination_reason,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"live_heuristic_sign_sweep_{timestamp}.csv"

    env: BeamNGRacingEnv | None = None
    results: list[CaseResult] = []

    try:
        try:
            # One live env is reused across cases to avoid relaunching BeamNG.
            # Each case calls reset(), which should restore the bootstrap spawn.
            env = BeamNGRacingEnv(
                CENTRELINE_PATH,
                use_mock=False,
                launch_beamng=True,
                live_spawn_mode="bootstrap",
                vehicle_id="ego_vehicle",
                steps_per_action=STEPS_PER_ACTION,
                max_episode_steps=MAX_EPISODE_STEPS,
            )
        except Exception:
            _print_live_failure_hints()
            raise

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for case_name, heading_sign, lateral_sign in CASES:
                print()
                print(
                    f"Starting case={case_name} "
                    f"heading_sign={heading_sign:+.1f} "
                    f"lateral_sign={lateral_sign:+.1f}"
                )
                result = _run_case(
                    env=env,
                    writer=writer,
                    csv_file=csv_file,
                    case_name=case_name,
                    heading_sign=heading_sign,
                    lateral_sign=lateral_sign,
                )
                results.append(result)
                _print_case_summary(result)

        print()
        print(f"CSV path: {csv_path}")
        _print_ranked_summary(results)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
