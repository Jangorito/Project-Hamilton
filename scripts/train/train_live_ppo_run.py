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
import re
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

from beamng_rl.envs.beamng_racing_env import REWARD_CONFIGS, BeamNGRacingEnv, RewardConfig
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


def _save_session_config(config: dict) -> None:
    try:
        import json
        _SESSION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except Exception:
        pass


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

CAR_RUN_SUFFIXES = {
    "sbr": "subaru",
    "etkc": "etk",
}
_STEP_SUFFIX_RE = re.compile(r"_\d+(?:k|m)?steps$", re.IGNORECASE)


def _parse_timesteps(value, default: int = TOTAL_TIMESTEPS) -> int:
    try:
        timesteps = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValueError("total_timesteps must be a positive integer") from exc
    if timesteps < 1:
        raise ValueError("total_timesteps must be a positive integer")
    return timesteps


def _format_step_suffix(total_timesteps: int) -> str:
    if total_timesteps % 1_000_000 == 0:
        return f"{total_timesteps // 1_000_000}msteps"
    if total_timesteps % 1_000 == 0:
        return f"{total_timesteps // 1_000}ksteps"
    return f"{total_timesteps}steps"


def _ensure_step_suffix(run_name: str, total_timesteps: int) -> str:
    base = _STEP_SUFFIX_RE.sub("", run_name.strip())
    suffix = _format_step_suffix(total_timesteps)
    return f"{base}_{suffix}" if base else suffix


def _retarget_run_step_suffix(
    rm: RunManager,
    run_name: str,
    requested_timesteps: int,
    actual_timesteps: int,
) -> str:
    if actual_timesteps >= requested_timesteps or actual_timesteps < 1:
        return run_name

    new_run_name = _ensure_step_suffix(run_name, actual_timesteps)
    if new_run_name == run_name:
        return run_name

    old_run_dir = rm.run_dir(run_name)
    new_run_dir = rm.run_dir(new_run_name)
    old_log_dir = rm.log_dir(run_name)
    new_log_dir = rm.log_dir(new_run_name)

    if new_run_dir.exists() or new_log_dir.exists():
        print(
            f"Warning: early-stop run name '{new_run_name}' already exists; "
            f"keeping '{run_name}'."
        )
        return run_name

    run_moved = False
    log_moved = False
    try:
        if old_run_dir.exists():
            new_run_dir.parent.mkdir(parents=True, exist_ok=True)
            old_run_dir.rename(new_run_dir)
            run_moved = True
        if old_log_dir.exists():
            new_log_dir.parent.mkdir(parents=True, exist_ok=True)
            old_log_dir.rename(new_log_dir)
            log_moved = True
    except OSError as exc:
        print(f"Warning: could not rename early-stop run to '{new_run_name}': {exc}")
        if run_moved and new_run_dir.exists() and not old_run_dir.exists():
            try:
                new_run_dir.rename(old_run_dir)
            except OSError:
                pass
        if log_moved and new_log_dir.exists() and not old_log_dir.exists():
            try:
                new_log_dir.rename(old_log_dir)
            except OSError:
                pass
        return run_name

    print(
        f"Renamed early-stop run: {run_name} -> {new_run_name} "
        f"({actual_timesteps:,}/{requested_timesteps:,} timesteps)"
    )
    return new_run_name


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


def _build_run_config(
    reward_config: RewardConfig,
    reward_key: str,
    *,
    total_timesteps: int,
    env_config: dict[str, Any],
    vehicle_model: str,
    speed_factor: int | None,
    timing_profile: str,
) -> dict[str, Any]:
    # Include config_key so run_config.json and run_info.md are self-documenting:
    # any reader can immediately see which experiment arm produced these results.
    return {
        "training": dict(
            total_timesteps  = total_timesteps,
            checkpoint_every = CHECKPOINT_EVERY,
            rollout_steps    = ROLLOUT_STEPS,
        ),
        "ppo":     PPO_CONFIG,
        "env":     {
            **env_config,
            "vehicle_model": vehicle_model,
            "speed_factor": speed_factor,
            "timing_profile": timing_profile,
        },
        "reward":  {"config_key": reward_key, **vars(reward_config)},
    }


def _ensure_car_suffix(
    run_name: str,
    vehicle_model: str,
    reward_key: str,
    speed_factor: int | None = None,
    total_timesteps: int | None = None,
) -> str:
    suffix = CAR_RUN_SUFFIXES[vehicle_model]
    base = _STEP_SUFFIX_RE.sub("", run_name.strip() or f"{reward_key}_{suffix}")
    tokens = base.lower().replace("-", "_").split("_")
    if suffix not in tokens:
        base = f"{base}_{suffix}"
        tokens = base.lower().replace("-", "_").split("_")
    speed_value = int(speed_factor) if speed_factor is not None else 1
    speed_suffix = f"{speed_value}x"
    if speed_value > 1 and speed_suffix not in tokens:
        base = f"{base}_{speed_suffix}"
    if total_timesteps is not None:
        base = _ensure_step_suffix(base, total_timesteps)
    return base


def _run_vehicle_model(config: dict[str, Any]) -> str | None:
    env = _as_mapping(config.get("env"))
    car = str(env.get("vehicle_model", "")).strip().lower()
    return car if car in CAR_RUN_SUFFIXES else None


def _run_speed_factor(config: dict[str, Any]) -> int:
    env = _as_mapping(config.get("env"))
    try:
        speed_factor = int(env.get("speed_factor", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return speed_factor if speed_factor in (1, 2, 4, 8, 16, 32) else 1


def _run_timing_profile(config: dict[str, Any]) -> str:
    env = _as_mapping(config.get("env"))
    timing_profile = str(env.get("timing_profile", "")).strip().lower()
    if timing_profile in ("legacy_60hz", "det50ms"):
        return timing_profile
    return "det50ms" if _run_speed_factor(config) > 1 else "legacy_60hz"


def _can_resume_with_vehicle(
    config: dict[str, Any],
    vehicle_model: str,
    timing_profile: str,
) -> tuple[bool, str]:
    run_vehicle_model = _run_vehicle_model(config)
    vehicle_matches = run_vehicle_model == vehicle_model or (
        run_vehicle_model is None and vehicle_model == "sbr"
    )
    if not vehicle_matches and run_vehicle_model is None:
        return False, "has no saved vehicle metadata"
    if not vehicle_matches:
        return False, f"was created for {run_vehicle_model}"
    run_timing_profile = _run_timing_profile(config)
    if run_timing_profile != timing_profile:
        return False, f"was created for {run_timing_profile} timing"
    return True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on BeamNG Hirochi Raceway.")
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new experiment from scratch.")
    parser.add_argument("--run-name", metavar="NAME",
                        help="Name for this training run.")
    # Selects the named reward preset from REWARD_CONFIGS. Keeping this as an
    # explicit flag (rather than a constant) means both experiment arms can be
    # launched from the same codebase with a single flag change and no risk of
    # editing the wrong values between runs.
    parser.add_argument("--reward-config", metavar="KEY", default=None,
                        help="Reward config key from REWARD_CONFIGS (v1 or v2). "
                             "Default: v1. Can also be set via launcher_session.json.")
    args = parser.parse_args()

    # CLI args take precedence over launcher session config.
    session = _load_session_config()
    try:
        total_timesteps = _parse_timesteps(session.get("total_timesteps", TOTAL_TIMESTEPS))
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    vehicle_model   = str(session.get("car", "etkc")).strip().lower()
    if vehicle_model not in ("sbr", "etkc"):
        sys.exit("Error: unknown car in launcher_session.json. Valid values: sbr, etkc")
    max_damage      = float(session.get("max_damage", ENV_CONFIG["max_damage"]))
    fresh           = args.fresh or bool(session.get("fresh", False))
    _sf             = session.get("speed_factor")
    speed_factor    = int(_sf) if _sf is not None else None
    if speed_factor is not None and speed_factor not in (1, 2, 4, 8, 16, 32):
        sys.exit("Error: speed_factor must be one of 1, 2, 4, 8, 16, 32.")
    timing_profile = str(session.get("timing_profile", "")).strip().lower() or None
    if timing_profile is not None and timing_profile not in ("legacy_60hz", "det50ms"):
        sys.exit("Error: timing_profile must be 'legacy_60hz' or 'det50ms'.")
    debug_mode = bool(session.get("debug_mode", False))

    # Resolve reward config key: CLI > launcher session > default "v1".
    reward_key = args.reward_config or str(session.get("reward_config", "v1"))
    if reward_key not in REWARD_CONFIGS:
        sys.exit(
            f"Error: unknown reward config key {reward_key!r}. "
            f"Valid keys: {sorted(REWARD_CONFIGS)}"
        )
    reward_config = REWARD_CONFIGS[reward_key]

    # Auto-generate a run name that embeds the car model and reward config key
    # so TensorBoard log directories and checkpoint folders are self-labelling.
    # e.g. --reward-config v1 on etkc -> "etkc_v1", v2 -> "etkc_v2".
    # A manual --run-name or launcher session name always takes precedence.
    run_name_hint = args.run_name or session.get("run_name", "") or f"{reward_key}_{CAR_RUN_SUFFIXES[vehicle_model]}"

    if session:
        sf_label = f"{speed_factor}x" if speed_factor is not None else "default"
        print(f"Launcher config: car={vehicle_model}, steps={total_timesteps}, "
              f"max_damage={max_damage}, fresh={fresh}, speed_factor={sf_label}, "
              f"reward_config={reward_key}")

    if not CENTRELINE_PATH.is_file():
        raise FileNotFoundError(f"Centreline JSON not found: {CENTRELINE_PATH}")

    rm = RunManager()

    env_config = {**ENV_CONFIG, "max_damage": max_damage}

    # ------------------------------------------------------------------
    # Resolve run name and resume checkpoint
    # ------------------------------------------------------------------
    if fresh:
        if timing_profile is None:
            timing_profile = "det50ms" if (speed_factor or 1) > 1 else "legacy_60hz"
        run_name = _ensure_car_suffix(
            run_name_hint or prompt_run_name(),
            vehicle_model,
            reward_key,
            speed_factor,
            total_timesteps,
        )
        if rm.exists(run_name):
            sys.exit(
                f"Error: run '{run_name}' already exists.\n"
                "Choose a different name or omit --fresh to resume it."
            )
        resume_path = None
        rm.create_run(
            run_name,
            _build_run_config(
                reward_config,
                reward_key,
                total_timesteps=total_timesteps,
                env_config=env_config,
                vehicle_model=vehicle_model,
                speed_factor=speed_factor,
                timing_profile=timing_profile,
            ),
        )
        print(f"Created run: {run_name}")
    else:
        run_name = run_name_hint or rm.find_latest_run()
        if run_name is None:
            sys.exit(
                "No existing runs found. Start a new one with --fresh."
            )
        if not rm.exists(run_name):
            sys.exit(f"Run '{run_name}' not found. Use --fresh to create it.")
        existing_config = rm.load_config(run_name)
        if timing_profile is None:
            timing_profile = _run_timing_profile(existing_config)
        can_resume, reason = _can_resume_with_vehicle(
            existing_config,
            vehicle_model,
            timing_profile,
        )
        if not can_resume and not debug_mode:
            sys.exit(
                f"Run '{run_name}' {reason}, "
                f"but launcher selected {vehicle_model}. Start a fresh run "
                "or select the matching vehicle."
            )
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
        # Log the reward config key prominently so the terminal output is
        # self-documenting and easy to grep when comparing run logs.
        print(f"\nReward config  : {reward_key}")
        print(f"Vehicle        : {vehicle_model}")
        print(f"Speed factor   : {speed_factor if speed_factor is not None else 'default'}x")
        print(f"Timing profile : {timing_profile}")
        for field_name, value in vars(reward_config).items():
            print(f"  {field_name}: {value}")

        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            vehicle_model=vehicle_model,
            reward_config=reward_config,
            speed_factor=speed_factor,
            timing_profile=timing_profile,
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
        actual_timesteps = int(getattr(model, "num_timesteps", total_timesteps))
        if actual_timesteps < total_timesteps:
            try:
                model.logger.close()
            except Exception:
                pass
            renamed_run = _retarget_run_step_suffix(
                rm,
                run_name,
                requested_timesteps=total_timesteps,
                actual_timesteps=actual_timesteps,
            )
            if renamed_run != run_name:
                run_name = renamed_run
                checkpoint_dir = rm.checkpoint_dir(run_name)
                log_dir = rm.log_dir(run_name)
                rollout_path = log_dir / "rollout_final.csv"
                if session:
                    session["run_name"] = run_name
                    _save_session_config(session)
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
            "actual_timesteps": actual_timesteps,
            "requested_timesteps": total_timesteps,
            "rollout_progress_m": f"{progress_m:.1f}",
            "termination_reason": reason,
            "progress_jumps": jumps,
            "full_lap_m": 2158,
            "heuristic_baseline_m": 965,
        }
        rm.finalize_run(run_name, results)

        print("\n=== Summary ===")
        print(f"  Timesteps   : {actual_timesteps:,}/{total_timesteps:,}")
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
