from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


STEPS_PER_ACTION = 30

ACTION_PHASES = (
    ("neutral_settling", 5, np.asarray([0.0, 0.0], dtype=np.float32)),
    ("full_throttle_straight", 30, np.asarray([0.0, 1.0], dtype=np.float32)),
    ("brake_test", 10, np.asarray([0.0, -1.0], dtype=np.float32)),
)


def _progress_gain_m(
    start_progress_m: float,
    end_progress_m: float,
    total_lap_length_m: float,
) -> float:
    """Return forward progress, accounting for one centreline wrap if needed."""

    progress_delta_m = float(end_progress_m) - float(start_progress_m)
    if progress_delta_m < -0.5 * total_lap_length_m:
        progress_delta_m += total_lap_length_m
    return float(progress_delta_m)


def _format_action(action: np.ndarray) -> str:
    return np.array2string(action, precision=2, separator=", ")


def _print_live_failure_hints() -> None:
    print()
    print("Live BeamNG smoke test failed. Helpful checks:")
    print("- check BeamNG.tech install path / BEAMNG_HOME")
    print("- check BeamNGpy is installed in the venv")
    print("- check hirochi_raceway exists")
    print("- check BeamNG is not blocked by another process")


def main() -> None:
    """Launch BeamNG.tech and diagnose live control/stepping behaviour."""

    centreline_path = (
        REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )

    env: BeamNGRacingEnv | None = None
    try:
        try:
            env = BeamNGRacingEnv(
                centreline_path,
                use_mock=False,
                launch_beamng=True,
                vehicle_id="ego_vehicle",
                steps_per_action=STEPS_PER_ACTION,
            )
            observation, info = env.reset()

            start_progress_m = float(info["progress_m"])
            end_progress_m = start_progress_m
            initial_lateral_error_m = float(info["lateral_error_m"])
            initial_progress_ratio = float(info["progress_ratio"])
            total_lap_length_m = float(env.track.total_lap_length)
            max_forward_speed_mps = 0.0
            final_termination_reason = "none"
            full_throttle_start_progress_m: float | None = None
            full_throttle_end_progress_m: float | None = None

            print(f"Initial observation shape: {observation.shape}")
            print(f"mock_simulation: {info['mock_simulation']}")
            print(f"steps_per_action: {STEPS_PER_ACTION}")
            print(
                "Initial progress: "
                f"{start_progress_m:.3f} m ({initial_progress_ratio:.4f})"
            )
            print(f"Initial lateral_error_m: {initial_lateral_error_m:.3f}")
            print(f"Initial heading_error_rad: {info['heading_error_rad']:.4f}")

            global_step = 0
            episode_stopped = False

            for phase_name, phase_steps, action in ACTION_PHASES:
                if phase_name == "full_throttle_straight":
                    full_throttle_start_progress_m = end_progress_m

                for local_step in range(1, phase_steps + 1):
                    global_step += 1
                    observation, reward, terminated, truncated, info = env.step(action)
                    reward_info = info["reward"]
                    control = info["control"]

                    end_progress_m = float(info["progress_m"])
                    forward_speed_mps = float(reward_info["forward_speed_mps"])
                    max_forward_speed_mps = max(
                        max_forward_speed_mps,
                        forward_speed_mps,
                    )
                    final_termination_reason = str(info["termination_reason"])

                    if phase_name == "full_throttle_straight":
                        full_throttle_end_progress_m = end_progress_m

                    print(
                        f"phase={phase_name} "
                        f"local_step={local_step}/{phase_steps} "
                        f"global_step={global_step} "
                        f"action={_format_action(action)} "
                        "control="
                        f"(steering={control['steering']:.3f}, "
                        f"throttle={control['throttle']:.3f}, "
                        f"brake={control['brake']:.3f}) "
                        f"reward={reward:.4f} "
                        f"progress_m={end_progress_m:.3f} "
                        f"progress_delta_m={reward_info['progress_delta_m']:.4f} "
                        f"forward_speed_mps={forward_speed_mps:.3f} "
                        f"speed_reward={reward_info['speed_reward']:.4f} "
                        f"lateral_error_m={info['lateral_error_m']:.3f} "
                        f"heading_error_rad={info['heading_error_rad']:.4f} "
                        f"terminated={terminated} "
                        f"truncated={truncated} "
                        f"termination_reason={final_termination_reason}"
                    )

                    if terminated or truncated:
                        episode_stopped = True
                        break

                if episode_stopped:
                    break

            total_progress_gained_m = _progress_gain_m(
                start_progress_m,
                end_progress_m,
                total_lap_length_m,
            )
            full_throttle_progress_gained_m = 0.0
            if (
                full_throttle_start_progress_m is not None
                and full_throttle_end_progress_m is not None
            ):
                full_throttle_progress_gained_m = _progress_gain_m(
                    full_throttle_start_progress_m,
                    full_throttle_end_progress_m,
                    total_lap_length_m,
                )

            print()
            print("Summary:")
            print(f"start progress: {start_progress_m:.3f} m")
            print(f"end progress: {end_progress_m:.3f} m")
            print(f"total progress gained: {total_progress_gained_m:.3f} m")
            print(
                "full throttle progress gained: "
                f"{full_throttle_progress_gained_m:.3f} m"
            )
            print(f"max forward_speed_mps: {max_forward_speed_mps:.3f}")
            print(f"final termination reason: {final_termination_reason}")

            if full_throttle_progress_gained_m < 1.0:
                print(
                    "WARNING: Full throttle did not move the vehicle by at least 1m. "
                    "Check BeamNG stepping, vehicle control, gear/parking brake, or "
                    "velocity extraction."
                )

            if max_forward_speed_mps < 1.0:
                print(
                    "WARNING: Forward speed stayed below 1m/s. Control may not be "
                    "reaching the vehicle, or the vehicle may not be properly active."
                )

            if abs(initial_lateral_error_m) > 2.0:
                print(
                    "NOTE: Spawn starts more than 2m from the centreline, so the "
                    "current lateral penalty will make early reward negative."
                )

            if initial_progress_ratio > 0.95 or initial_progress_ratio < 0.05:
                print(
                    "NOTE: Spawn is close to the centreline wrap boundary. Keep "
                    "lap-completion termination disabled for now."
                )

        except Exception:
            _print_live_failure_hints()
            raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
