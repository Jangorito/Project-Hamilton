"""Evaluate saved checkpoints and final model for a run and find the best one.

Run this after a training session to compare checkpoint performance and
identify which model to visualise with watch_model.py.

Usage:
    python scripts/eval/eval_checkpoints.py                      # latest run
    python scripts/eval/eval_checkpoints.py v2_smoother          # specific run
    python scripts/eval/eval_checkpoints.py --run v2_smoother    # specific run
    python scripts/eval/eval_checkpoints.py --steps 300          # shorter rollouts
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv
from beamng_rl.training.run_manager import RunManager, _parse_timestep

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
DEFAULT_EVAL_STEPS = 500

CSV_FIELDS = ["checkpoint", "timestep", "progress_m", "termination_reason", "progress_jumps"]


def _run_rollout(env: BeamNGRacingEnv, model: PPO, n_steps: int) -> tuple[float, str, int]:
    obs, info = env.reset()
    progress_m = float(info.get("episode_progress_m", 0.0))
    reason = "none"
    jumps = 0

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        progress_m = float(info.get("episode_progress_m", progress_m))

        reward_info = info.get("reward", {})
        if reward_info.get("progress_jump_detected", False):
            jumps += 1

        if terminated or truncated:
            reason = str(info.get("termination_reason", "unknown"))
            break

    return progress_m, reason, jumps


def _env_int(env_cfg: dict, key: str, default: int) -> int:
    try:
        return int(env_cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(env_cfg: dict, key: str, default: float) -> float:
    try:
        return float(env_cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _run_vehicle_model(run_name: str, env_cfg: dict) -> str:
    vehicle_model = str(env_cfg.get("vehicle_model", "")).strip().lower()
    if vehicle_model not in ("sbr", "etkc"):
        print(
            f"Warning: vehicle_model not found in run_config.json for run '{run_name}'"
            " - defaulting to 'etkc'."
        )
        return "etkc"
    return vehicle_model


def _run_speed_factor(env_cfg: dict) -> int | None:
    value = env_cfg.get("speed_factor")
    if value is None:
        return None
    try:
        speed_factor = int(value)
    except (TypeError, ValueError):
        return None
    return speed_factor if speed_factor in (1, 2, 4, 8, 16, 32) else None


def _run_timing_profile(env_cfg: dict, speed_factor: int | None) -> str:
    timing_profile = str(env_cfg.get("timing_profile", "")).strip().lower()
    if timing_profile in ("legacy_60hz", "det50ms"):
        return timing_profile
    return "det50ms" if (speed_factor or 1) > 1 else "legacy_60hz"


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _actual_timestep_from_run_info(run_name: str, rm: RunManager) -> int | None:
    info_path = rm.run_dir(run_name) / "run_info.md"
    if not info_path.exists():
        return None
    try:
        lines = info_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if "**actual_timesteps**" not in line:
            continue
        _, _, value = line.partition(":")
        return _parse_int(value)
    return None


def _configured_timestep(run_cfg: dict) -> int | None:
    training = run_cfg.get("training", {})
    if not isinstance(training, dict):
        return None
    return _parse_int(training.get("total_timesteps"))


def _models_to_evaluate(
    run_name: str,
    rm: RunManager,
    run_cfg: dict,
) -> list[tuple[Path, int]]:
    models = [(ckpt, _parse_timestep(ckpt)) for ckpt in rm.find_all_checkpoints(run_name)]

    final_model = rm.run_dir(run_name) / "model_final.zip"
    if final_model.is_file():
        final_timestep = (
            _actual_timestep_from_run_info(run_name, rm)
            or _configured_timestep(run_cfg)
            or _parse_timestep(final_model)
        )
        models.append((final_model, final_timestep))

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BeamNG PPO model files.")
    parser.add_argument("run_name", nargs="?", metavar="RUN",
                        help="Run name to evaluate.")
    parser.add_argument("--run", metavar="NAME", help="Run name to evaluate.")
    parser.add_argument("--steps", type=int, default=DEFAULT_EVAL_STEPS,
                        help="Max rollout steps per model.")
    parser.add_argument("--port", metavar="PORT", type=int, default=None,
                        help="BeamNG.tech RPC port (default: 25252). Must differ per parallel eval.")
    parser.add_argument("--beamng-user", metavar="PATH", default=None,
                        help="BeamNG user data folder. Required for a second parallel instance.")
    args = parser.parse_args()

    rm = RunManager()

    if args.run and args.run_name and args.run != args.run_name:
        sys.exit("Use either positional RUN or --run NAME, not both.")

    from beamng_rl.bootstrap.beamng_setup import PORT as _DEFAULT_PORT
    port = args.port if args.port is not None else _DEFAULT_PORT
    beamng_user = args.beamng_user or None

    run_name = args.run or args.run_name or rm.find_latest_run()
    if run_name is None:
        sys.exit("No runs found. Train a model first.")
    if not rm.exists(run_name):
        sys.exit(f"Run '{run_name}' not found.")

    run_cfg = rm.load_config(run_name)
    models = _models_to_evaluate(run_name, rm, run_cfg)
    if not models:
        sys.exit(f"No checkpoints or final model found in run '{run_name}'.")

    out_csv = rm.log_dir(run_name) / "checkpoints_eval.csv"
    rm.log_dir(run_name).mkdir(parents=True, exist_ok=True)

    env_cfg_raw = run_cfg.get("env", {})
    env_cfg = env_cfg_raw if isinstance(env_cfg_raw, dict) else {}

    vehicle_model = _run_vehicle_model(run_name, env_cfg)
    speed_factor = _run_speed_factor(env_cfg)
    timing_profile = _run_timing_profile(env_cfg, speed_factor)
    obs_config = str(run_cfg.get("obs", {}).get("config_key", "v1"))
    steps_per_action = _env_int(env_cfg, "steps_per_action", 15)
    max_episode_steps = _env_int(env_cfg, "max_episode_steps", 500)
    max_lateral_error_m = _env_float(env_cfg, "max_lateral_error_m", 10.0)
    stuck_steps_limit = _env_int(env_cfg, "stuck_steps_limit", 100)
    min_progress_delta_m = _env_float(env_cfg, "min_progress_delta_m", 0.05)
    max_progress_delta_m = _env_float(env_cfg, "max_progress_delta_m", 50.0)
    progress_jump_penalty = _env_float(env_cfg, "progress_jump_penalty", 5.0)
    max_damage = _env_float(env_cfg, "max_damage", 500.0)
    wall_bash_steps_limit = _env_int(env_cfg, "wall_bash_steps_limit", 30)

    print(f"Run      : {run_name}")
    print(f"Vehicle  : {vehicle_model}")
    print(f"Obs      : {obs_config}")
    print(f"Timing   : {timing_profile}")
    print(f"Speed    : {speed_factor if speed_factor is not None else 'default'}x")
    print(f"Found    : {len(models)} model files")
    print(f"Steps    : {args.steps} per rollout")
    print("Launching BeamNG...")

    env: BeamNGRacingEnv | None = None
    results = []

    try:
        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            vehicle_model=vehicle_model,
            speed_factor=speed_factor,
            timing_profile=timing_profile,
            steps_per_action=steps_per_action,
            max_episode_steps=max_episode_steps,
            max_lateral_error_m=max_lateral_error_m,
            stuck_steps_limit=stuck_steps_limit,
            min_progress_delta_m=min_progress_delta_m,
            max_progress_delta_m=max_progress_delta_m,
            progress_jump_penalty=progress_jump_penalty,
            max_damage=max_damage,
            wall_bash_steps_limit=wall_bash_steps_limit,
            obs_config=obs_config,
            beamng_port=port,
            **({"beamng_user": beamng_user} if beamng_user is not None else {}),
        )

        for model_path, timestep in models:
            print(f"\nEvaluating {model_path.name} (t={timestep})...")
            model = PPO.load(str(model_path))
            progress_m, reason, jumps = _run_rollout(env, model, args.steps)
            print(f"  progress={progress_m:.1f}m  reason={reason}  jumps={jumps}")
            results.append({
                "checkpoint": str(model_path),
                "timestep": timestep,
                "progress_m": progress_m,
                "termination_reason": reason,
                "progress_jumps": jumps,
            })

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(results)

        best = max(results, key=lambda r: r["progress_m"])
        print(f"\n=== Results saved to {out_csv} ===")
        print("\nBest model:")
        print(f"  {Path(best['checkpoint']).name}")
        print(f"  progress={best['progress_m']:.1f} m  at timestep={best['timestep']}")
        print("\nTo visualise:")
        print(f"  python scripts/eval/watch_model.py --run {run_name}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
