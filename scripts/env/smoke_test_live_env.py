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
LIVE_SPAWN_MODE = "bootstrap"
LIVE_SPAWN_PROGRESS_M = 50.0
LIVE_SPAWN_LATERAL_OFFSET_M = 0.0
LIVE_SPAWN_Z_OFFSET_M = 0.5

# To test centreline-aligned reset, change:
#   LIVE_SPAWN_MODE = "centreline"
#   LIVE_SPAWN_PROGRESS_M = 50.0
# Then run:
#   python scripts/smoke_test_live_env.py
#
# If centreline placement has BeamNG terrain-height issues, switch
# LIVE_SPAWN_MODE back to "bootstrap".

ACTION_PHASES = (
    ("neutral_settling", 5, np.asarray([0.0, 0.0], dtype=np.float32)),
    ("full_throttle_straight", 30, np.asarray([0.0, 1.0], dtype=np.float32)),
    ("brake_test", 10, np.asarray([0.0, -1.0], dtype=np.float32)),
)


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
                live_spawn_mode=LIVE_SPAWN_MODE,
                live_spawn_progress_m=LIVE_SPAWN_PROGRESS_M,
                live_spawn_lateral_offset_m=LIVE_SPAWN_LATERAL_OFFSET_M,
                live_spawn_z_offset_m=LIVE_SPAWN_Z_OFFSET_M,
            )
            observation, info = env.reset()

            raw_start_progress_m = float(info["raw_progress_m"])
            raw_end_progress_m = raw_start_progress_m
            raw_end_progress_ratio = float(info["raw_progress_ratio"])
            episode_start_progress_m = float(info["episode_start_progress_m"])
            start_episode_progress_m = float(info["episode_progress_m"])
            end_episode_progress_m = start_episode_progress_m
            initial_lateral_error_m = float(info["lateral_error_m"])
            initial_raw_progress_ratio = float(info["raw_progress_ratio"])
            initial_near_progress_wrap = bool(info["near_progress_wrap"])
            max_forward_speed_mps = 0.0
            final_termination_reason = "none"
            full_throttle_start_episode_progress_m: float | None = None
            full_throttle_end_episode_progress_m: float | None = None

            print(f"Initial observation shape: {observation.shape}")
            print(f"mock_simulation: {info['mock_simulation']}")
            print(f"steps_per_action: {STEPS_PER_ACTION}")
            print(f"spawn_mode: {info['spawn_mode']}")
            print(f"Initial raw_progress_m: {raw_start_progress_m:.3f} m")
            print(f"Initial raw_progress_ratio: {initial_raw_progress_ratio:.4f}")
            print(
                "Initial episode_start_progress_m: "
                f"{episode_start_progress_m:.3f} m"
            )
            print(f"Initial episode_progress_m: {end_episode_progress_m:.3f} m")
            print(f"Initial lateral_error_m: {initial_lateral_error_m:.3f}")
            print(f"near_progress_wrap: {initial_near_progress_wrap}")
            print(f"Initial heading_error_rad: {info['heading_error_rad']:.4f}")

            global_step = 0
            episode_stopped = False

            for phase_name, phase_steps, action in ACTION_PHASES:
                if phase_name == "full_throttle_straight":
                    full_throttle_start_episode_progress_m = end_episode_progress_m

                for local_step in range(1, phase_steps + 1):
                    global_step += 1
                    observation, reward, terminated, truncated, info = env.step(action)
                    reward_info = info["reward"]
                    control = info["control"]

                    raw_end_progress_m = float(info["raw_progress_m"])
                    raw_end_progress_ratio = float(info["raw_progress_ratio"])
                    end_episode_progress_m = float(info["episode_progress_m"])
                    forward_speed_mps = float(reward_info["forward_speed_mps"])
                    max_forward_speed_mps = max(
                        max_forward_speed_mps,
                        forward_speed_mps,
                    )
                    final_termination_reason = str(info["termination_reason"])

                    if phase_name == "full_throttle_straight":
                        full_throttle_end_episode_progress_m = end_episode_progress_m

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
                        f"raw_progress_m={raw_end_progress_m:.3f} "
                        f"raw_progress_ratio={raw_end_progress_ratio:.4f} "
                        f"episode_progress_m={end_episode_progress_m:.3f} "
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

            episode_progress_gained_m = (
                end_episode_progress_m - start_episode_progress_m
            )
            full_throttle_episode_progress_gained_m = 0.0
            if (
                full_throttle_start_episode_progress_m is not None
                and full_throttle_end_episode_progress_m is not None
            ):
                full_throttle_episode_progress_gained_m = (
                    full_throttle_end_episode_progress_m
                    - full_throttle_start_episode_progress_m
                )

            print()
            print("Summary:")
            print(f"episode progress gained: {episode_progress_gained_m:.3f} m")
            print(f"raw start progress: {raw_start_progress_m:.3f} m")
            print(f"raw end progress: {raw_end_progress_m:.3f} m")
            print(
                "full throttle episode progress gained: "
                f"{full_throttle_episode_progress_gained_m:.3f} m"
            )
            print(f"max forward_speed_mps: {max_forward_speed_mps:.3f}")
            print(f"final termination reason: {final_termination_reason}")

            if full_throttle_episode_progress_gained_m < 1.0:
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

            if initial_near_progress_wrap:
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
