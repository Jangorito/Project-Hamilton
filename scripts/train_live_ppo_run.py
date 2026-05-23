"""Live PPO training run with TensorBoard logging and checkpoints.

Designed for real training runs after the smoke test has validated the
pipeline. Adjust TOTAL_TIMESTEPS to taste:

  ~3_300  =>  ~15 min   (quick sanity check — is the reward curve moving?)
  ~27_000 =>  ~2 hours  (enough to see early learning)
  ~55_000 =>  ~4 hours
  100_000 =>  ~7.5 hours

Usage:
    python scripts/train_live_ppo_run.py

TensorBoard:
    tensorboard --logdir logs/live_training
"""

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
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required. Install with:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install \"stable-baselines3>=2.3.0\""
    ) from exc

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv

# ---------------------------------------------------------------------------
# Change these to control run length. See the table in the module docstring.
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 3_300   # ~15 min at 3.7 policy steps/sec

# How often to save a checkpoint (in timesteps). Lets you inspect the model
# mid-run or recover if BeamNG crashes late in training.
CHECKPOINT_EVERY = 1_000

# How many deterministic steps to roll out after training for comparison
# against the heuristic baseline (~965 m).
ROLLOUT_STEPS = 300

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
MODEL_DIR       = REPO_ROOT / "models"
LOG_DIR         = REPO_ROOT / "logs" / "live_training"
CHECKPOINT_DIR  = REPO_ROOT / "models" / "checkpoints"

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


# ---------------------------------------------------------------------------
# Helpers (shared with smoke script, kept local to avoid coupling)
# ---------------------------------------------------------------------------

def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _info_float(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _float_or_default(info.get(key), default)


def _reward_float(reward_info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _float_or_default(reward_info.get(key), default)


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
        "lateral_error_m": _info_float(info, "lateral_error_m", _reward_float(reward_info, "lateral_error_m")),
        "heading_error_rad": _info_float(info, "heading_error_rad", _reward_float(reward_info, "heading_error_rad")),
        "progress_jump_detected": bool(reward_info.get("progress_jump_detected", info.get("progress_jump_detected", False))),
        "original_progress_delta_m": _reward_float(reward_info, "original_progress_delta_m", progress_delta_m),
        "termination_reason": str(info.get("termination_reason", "none")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _run_deterministic_rollout(
    *,
    env: BeamNGRacingEnv,
    model: PPO,
    csv_path: Path,
) -> tuple[float, str, int]:
    obs, info = env.reset()
    start_progress = _info_float(info, "episode_progress_m")
    final_progress = start_progress
    final_reason = "none"
    jump_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for step in range(1, ROLLOUT_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            obs, reward, terminated, truncated, info = env.step(action)
            row = _build_csv_row(
                step_number=step, action=action, reward=reward,
                terminated=terminated, truncated=truncated, info=info,
            )
            writer.writerow(row)
            final_progress = float(row["episode_progress_m"])
            final_reason = str(row["termination_reason"])
            if row["progress_jump_detected"]:
                jump_count += 1
            if terminated or truncated:
                break

    return final_progress - start_progress, final_reason, jump_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CENTRELINE_PATH.is_file():
        raise FileNotFoundError(f"Centreline JSON not found: {CENTRELINE_PATH}")

    for d in (MODEL_DIR, LOG_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path   = MODEL_DIR / f"live_ppo_{timestamp}.zip"
    rollout_path = LOG_DIR   / f"live_ppo_rollout_{timestamp}.csv"

    env: BeamNGRacingEnv | None = None
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

        # Monitor wrapper gives SB3 episode reward/length stats for TensorBoard.
        monitored_env = Monitor(env)

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_EVERY,
            save_path=str(CHECKPOINT_DIR),
            name_prefix=f"ppo_{timestamp}",
            verbose=1,
        )

        model = PPO(
            "MlpPolicy",
            monitored_env,
            verbose=1,
            n_steps=64,
            batch_size=32,
            gamma=0.99,
            learning_rate=3e-4,
            tensorboard_log=str(LOG_DIR),
        )

        print(f"Training for {TOTAL_TIMESTEPS} timesteps (~15 min)...")
        print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")
        print(f"Checkpoints: {CHECKPOINT_DIR}")

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)
        model.save(str(model_path))
        print(f"\nSaved model: {model_path}")

        print(f"\nRunning {ROLLOUT_STEPS}-step deterministic rollout...")
        progress_m, reason, jumps = _run_deterministic_rollout(
            env=env, model=model, csv_path=rollout_path,
        )

        print("\n=== Summary ===")
        print(f"  Rollout progress : {progress_m:.1f} m  (heuristic baseline: ~965 m)")
        print(f"  Termination      : {reason}")
        print(f"  Progress jumps   : {jumps}")
        print(f"  Model            : {model_path}")
        print(f"  Rollout CSV      : {rollout_path}")

    except Exception:
        print("\nLive PPO run failed. Checks:")
        print("  - BeamNG install path / BEAMNG_HOME")
        print("  - No other BeamNG process running (port 25252)")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
