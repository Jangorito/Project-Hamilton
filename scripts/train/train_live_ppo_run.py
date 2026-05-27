"""Live PPO training run with TensorBoard logging and checkpoints.

Adjust TOTAL_TIMESTEPS to taste:

  ~3_300  =>  ~15 min
  ~27_000 =>  ~2 hours
  ~55_000 =>  ~4 hours
  100_000 =>  ~7.5 hours

Each run lives in models/runs/<run_name>/ with a config snapshot and
results summary written automatically. Runs are grouped by name so you can
compare experiments across TensorBoard and eval_checkpoints.py.

Usage:
    # Start a new experiment (prompts for name if omitted)
    python scripts/train_live_ppo_run.py --fresh
    python scripts/train_live_ppo_run.py --fresh --run-name v2_smoother

    # Resume the latest run (default)
    python scripts/train_live_ppo_run.py

    # Resume a specific run
    python scripts/train_live_ppo_run.py --run-name v2_smoother

TensorBoard:
    tensorboard --logdir logs/runs
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
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

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv, RewardConfig
from beamng_rl.training.callbacks import RewardComponentLogger, StepProgressWriter, StopSignalCallback
from beamng_rl.training.run_manager import RunManager, prompt_run_name

# ---------------------------------------------------------------------------
# Launcher session config — written by scripts/launcher/app.py before launch
# ---------------------------------------------------------------------------
_SESSION_CONFIG_PATH = REPO_ROOT / "config" / "launcher_session.json"


def _load_session_config() -> dict:
    if not _SESSION_CONFIG_PATH.exists():
        return {}
    try:
        import json
        return json.loads(_SESSION_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Tunable training constants (overridden by launcher_session.json if present)
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS  = 100_000
CHECKPOINT_EVERY = 8_000
ROLLOUT_STEPS    = 500

# Written every rollout so the launcher UI can show real-time step progress.
_PROGRESS_FILE = REPO_ROOT / "logs" / "remote" / "current_steps.txt"
# Written by the launcher stop button to trigger a graceful shutdown.
_STOP_SIGNAL_FILE = REPO_ROOT / "logs" / "remote" / "stop_signal.txt"

# ---------------------------------------------------------------------------
# Env / PPO config — all values here are captured in run_config.json
# ---------------------------------------------------------------------------
ENV_CONFIG = dict(
    steps_per_action      = 15,
    max_episode_steps     = 500,
    max_lateral_error_m   = 10.0,
    max_progress_delta_m  = 50.0,
    progress_jump_penalty = 5.0,
    max_damage            = 500.0,
    wall_bash_steps_limit = 30,
    stuck_steps_limit     = 100,
)

PPO_CONFIG = dict(
    n_steps       = 256,
    batch_size    = 64,
    gamma         = 0.99,
    learning_rate = 3e-4,
)

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"

CSV_FIELDS = [
    "step", "action_steering", "action_throttle_brake", "reward",
    "episode_progress_m", "pos_x", "pos_y", "pos_z",
    "progress_delta_m", "forward_speed_mps", "lateral_error_m",
    "heading_error_rad", "progress_jump_detected", "original_progress_delta_m",
    "termination_reason", "terminated", "truncated",
]


# ---------------------------------------------------------------------------
# Helpers
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
    pos = info.get("vehicle_pos") or (0.0, 0.0, 0.0)
    return {
        "step": step_number,
        "action_steering": float(action[0]),
        "action_throttle_brake": float(action[1]),
        "reward": float(reward),
        "episode_progress_m": _info_float(info, "episode_progress_m"),
        "pos_x": float(pos[0]),
        "pos_y": float(pos[1]),
        "pos_z": float(pos[2]),
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


def _build_run_config() -> dict[str, Any]:
    reward_defaults = vars(RewardConfig())
    return {
        "training": dict(
            total_timesteps  = TOTAL_TIMESTEPS,
            checkpoint_every = CHECKPOINT_EVERY,
            rollout_steps    = ROLLOUT_STEPS,
        ),
        "ppo":     PPO_CONFIG,
        "env":     ENV_CONFIG,
        "reward":  reward_defaults,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on BeamNG Hirochi Raceway.")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new experiment from scratch.")
    parser.add_argument("--run-name", metavar="NAME",
                        help="Name for this training run.")
    args = parser.parse_args()

    # CLI args take precedence over launcher session config.
    session = _load_session_config()
    total_timesteps = int(session.get("total_timesteps", TOTAL_TIMESTEPS))
    vehicle_model   = str(session.get("car", "etkc"))
    max_damage      = float(session.get("max_damage", ENV_CONFIG["max_damage"]))
    fresh           = args.fresh or bool(session.get("fresh", False))
    run_name_hint   = args.run_name or session.get("run_name", "") or None

    if session:
        print(f"Launcher config: car={vehicle_model}, steps={total_timesteps}, "
              f"max_damage={max_damage}, fresh={fresh}")

    if not CENTRELINE_PATH.is_file():
        raise FileNotFoundError(f"Centreline JSON not found: {CENTRELINE_PATH}")

    rm = RunManager()

    env_config = {**ENV_CONFIG, "max_damage": max_damage}

    # ------------------------------------------------------------------
    # Resolve run name and resume checkpoint
    # ------------------------------------------------------------------
    if fresh:
        run_name = run_name_hint or prompt_run_name()
        if rm.exists(run_name):
            sys.exit(
                f"Error: run '{run_name}' already exists.\n"
                "Choose a different name or omit --fresh to resume it."
            )
        resume_path = None
        rm.create_run(run_name, _build_run_config())
        print(f"Created run: {run_name}")
    else:
        run_name = run_name_hint or rm.find_latest_run()
        if run_name is None:
            sys.exit(
                "No existing runs found. Start a new one with --fresh."
            )
        if not rm.exists(run_name):
            sys.exit(f"Run '{run_name}' not found. Use --fresh to create it.")
        resume_path = rm.find_latest_checkpoint(run_name)
        rm.open_run(run_name)
        print(f"Resuming run: {run_name}")
        if resume_path:
            print(f"  checkpoint : {resume_path.name}")
        else:
            print("  no checkpoint found — starting weights from scratch")

    checkpoint_dir = rm.checkpoint_dir(run_name)
    log_dir        = rm.log_dir(run_name)
    rollout_path   = log_dir / "rollout_final.csv"

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            vehicle_model=vehicle_model,
            **env_config,
        )

        monitored_env = Monitor(env)

        checkpoint_cb = CheckpointCallback(
            save_freq=CHECKPOINT_EVERY,
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
            verbose=1,
        )

        if resume_path is not None:
            print(f"\nLoading checkpoint weights...")
            model = PPO.load(str(resume_path), env=monitored_env, tensorboard_log=str(log_dir))
        else:
            model = PPO(
                "MlpPolicy",
                monitored_env,
                verbose=1,
                tensorboard_log=str(log_dir),
                **PPO_CONFIG,
            )

        print(f"\nTraining for {total_timesteps} timesteps...")
        print(f"  TensorBoard : tensorboard --logdir {log_dir.parent}")
        print(f"  Checkpoints : {checkpoint_dir}")

        _PROGRESS_FILE.unlink(missing_ok=True)
        _STOP_SIGNAL_FILE.unlink(missing_ok=True)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[
                checkpoint_cb,
                RewardComponentLogger(),
                StepProgressWriter(_PROGRESS_FILE),
                StopSignalCallback(_STOP_SIGNAL_FILE),
            ],
        )
        _PROGRESS_FILE.unlink(missing_ok=True)
        _STOP_SIGNAL_FILE.unlink(missing_ok=True)

        model_path = rm.run_dir(run_name) / "model_final.zip"
        model.save(str(model_path))
        print(f"\nSaved final model: {model_path}")

        print(f"\nRunning {ROLLOUT_STEPS}-step deterministic rollout...")
        progress_m, reason, jumps = _run_deterministic_rollout(
            env=env, model=model, csv_path=rollout_path,
        )

        results = {
            "rollout_progress_m": f"{progress_m:.1f}",
            "termination_reason": reason,
            "progress_jumps": jumps,
            "full_lap_m": 2158,
            "heuristic_baseline_m": 965,
        }
        rm.finalize_run(run_name, results)

        print("\n=== Summary ===")
        print(f"  Progress    : {progress_m:.1f} m  (heuristic ~965 m / lap ~2158 m)")
        print(f"  Termination : {reason}")
        print(f"  Jumps       : {jumps}")
        print(f"  Run dir     : {rm.run_dir(run_name)}")

    except Exception:
        print("\nTraining failed. Checks:")
        print("  - BeamNG install path / BEAMNG_HOME")
        print("  - No other BeamNG process running (port 25252)")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
