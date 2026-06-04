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

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required to run this live PPO smoke training. "
        "Install project requirements with:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        "or install it directly with:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install \"stable-baselines3>=2.3.0\""
    ) from exc

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


TOTAL_TIMESTEPS = 256
MAX_ROLLOUT_STEPS = 100

CENTRELINE_PATH = (
    REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
)
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "live_ppo_smoke.zip"
LOG_DIR = REPO_ROOT / "logs" / "live_training"

CSV_FIELDS = [
    "step",
    "action_steering",
    "action_throttle_brake",
    "reward",
    "episode_progress_m",
    "progress_delta_m",
    "forward_speed_mps",
    "lateral_error_m",
    "heading_error_rad",
    "progress_jump_detected",
    "original_progress_delta_m",
    "termination_reason",
    "terminated",
    "truncated",
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


def _reward_float(
    reward_info: Mapping[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    return _float_or_default(reward_info.get(key), default)


def _normalise_action(action: Any) -> np.ndarray:
    return np.asarray(action, dtype=np.float32).reshape(-1)


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
    progress_delta_m = _reward_float(reward_info, "progress_delta_m")

    return {
        "step": step_number,
        "action_steering": float(action[0]),
        "action_throttle_brake": float(action[1]),
        "reward": float(reward),
        "episode_progress_m": _info_float(info, "episode_progress_m"),
        "progress_delta_m": progress_delta_m,
        "forward_speed_mps": _reward_float(reward_info, "forward_speed_mps"),
        "lateral_error_m": _info_float(
            info,
            "lateral_error_m",
            _reward_float(reward_info, "lateral_error_m"),
        ),
        "heading_error_rad": _info_float(
            info,
            "heading_error_rad",
            _reward_float(reward_info, "heading_error_rad"),
        ),
        "progress_jump_detected": bool(
            reward_info.get(
                "progress_jump_detected",
                info.get("progress_jump_detected", False),
            )
        ),
        "original_progress_delta_m": _reward_float(
            reward_info,
            "original_progress_delta_m",
            progress_delta_m,
        ),
        "termination_reason": str(info.get("termination_reason", "none")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _print_step(row: Mapping[str, Any]) -> None:
    print(
        f"step={row['step']} "
        "action="
        f"[{row['action_steering']:.3f}, {row['action_throttle_brake']:.3f}] "
        f"reward={row['reward']:.4f} "
        f"episode_progress_m={row['episode_progress_m']:.3f} "
        f"progress_delta_m={row['progress_delta_m']:.4f} "
        f"forward_speed_mps={row['forward_speed_mps']:.3f} "
        f"lateral_error_m={row['lateral_error_m']:.3f} "
        f"heading_error_rad={row['heading_error_rad']:.4f} "
        f"progress_jump_detected={row['progress_jump_detected']} "
        f"termination_reason={row['termination_reason']}"
    )


def _print_live_failure_hints() -> None:
    print()
    print("Live BeamNG PPO smoke training failed. Helpful checks:")
    print("- check BeamNG.tech install path / BEAMNG_HOME")
    print("- check BeamNGpy is installed")
    print("- check hirochi_raceway exists")
    print("- check no existing BeamNG process conflict")


def _run_deterministic_rollout(
    *,
    env: BeamNGRacingEnv,
    model: PPO,
    csv_path: Path,
) -> tuple[float, str, int]:
    obs, info = env.reset()
    start_episode_progress_m = _info_float(info, "episode_progress_m")
    final_episode_progress_m = start_episode_progress_m
    final_termination_reason = "none"
    progress_jump_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for step_number in range(1, MAX_ROLLOUT_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            action_array = _normalise_action(action)
            obs, reward, terminated, truncated, info = env.step(action_array)
            row = _build_csv_row(
                step_number=step_number,
                action=action_array,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

            writer.writerow(row)
            _print_step(row)

            final_episode_progress_m = float(row["episode_progress_m"])
            final_termination_reason = str(row["termination_reason"])
            if row["progress_jump_detected"]:
                progress_jump_count += 1

            if terminated or truncated:
                break

    total_episode_progress_m = final_episode_progress_m - start_episode_progress_m
    return total_episode_progress_m, final_termination_reason, progress_jump_count


def main() -> None:
    if not CENTRELINE_PATH.is_file():
        raise FileNotFoundError(f"Centreline JSON not found: {CENTRELINE_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = LOG_DIR / f"live_ppo_smoke_rollout_{timestamp}.csv"

    env: BeamNGRacingEnv | None = None
    try:
        try:
            env = BeamNGRacingEnv(
                CENTRELINE_PATH,
                use_mock=False,
                launch_beamng=True,
                live_spawn_mode="bootstrap",
                vehicle_id="ego_vehicle",
                steps_per_action=15,
                max_episode_steps=500,
                max_progress_delta_m=50.0,
                progress_jump_penalty=5.0,
            )

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=64,
                batch_size=32,
                gamma=0.99,
                learning_rate=3e-4,
            )

            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            model.save(MODEL_PATH)
            print(f"Saved live PPO smoke model to: {MODEL_PATH}")

            (
                total_rollout_episode_progress_m,
                final_termination_reason,
                progress_jump_count,
            ) = _run_deterministic_rollout(
                env=env,
                model=model,
                csv_path=csv_path,
            )

            print()
            print("Summary:")
            print(f"model path: {MODEL_PATH}")
            print(f"rollout CSV path: {csv_path}")
            print(
                "total rollout episode progress: "
                f"{total_rollout_episode_progress_m:.3f} m"
            )
            print(f"final termination reason: {final_termination_reason}")
            print(f"progress jumps detected: {progress_jump_count}")

        except Exception:
            _print_live_failure_hints()
            raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
