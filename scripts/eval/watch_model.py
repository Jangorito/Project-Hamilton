"""Watch a trained PPO model drive in BeamNG and draw its path on the track.

Loads a checkpoint, runs one deterministic episode with BeamNG visible,
then draws the driven trajectory on the circuit in 3D.

Usage:
    # Best checkpoint from a named run (uses checkpoints_eval.csv if present)
    python scripts/eval/watch_model.py --run v2_smoother

    # Latest checkpoint from the most recent run
    python scripts/eval/watch_model.py

    # Explicit model file
    python scripts/eval/watch_model.py path/to/model.zip

    python scripts/eval/watch_model.py --run v2_smoother --steps 800
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
from beamng_rl.visualisation.debug_draw_path import DebugPathDrawer

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"


def _best_checkpoint_from_eval(run_name: str, rm: RunManager) -> Path | None:
    """Return the checkpoint with the highest progress_m from the eval CSV."""
    csv_path = rm.log_dir(run_name) / "checkpoints_eval.csv"
    if not csv_path.exists():
        return None
    best_ckpt: str | None = None
    best_progress = -1.0
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                p = float(row["progress_m"])
            except (KeyError, ValueError):
                continue
            if p > best_progress:
                best_progress = p
                best_ckpt = row.get("checkpoint")
    return Path(best_ckpt) if best_ckpt and Path(best_ckpt).is_file() else None


def _vehicle_model_for_run(run_name: str | None, rm: RunManager) -> str:
    if run_name is None:
        print("Warning: no run name known — defaulting vehicle_model to 'etkc'.")
        return "etkc"
    config = rm.load_config(run_name)
    vehicle_model = config.get("env", {}).get("vehicle_model", "").strip().lower()
    if vehicle_model not in ("sbr", "etkc"):
        print(
            f"Warning: vehicle_model not found in run_config.json for run '{run_name}'"
            " — defaulting to 'etkc'."
        )
        return "etkc"
    return vehicle_model


def _obs_config_for_run(run_name: str | None, rm: RunManager) -> str:
    if run_name is None:
        return "v1"
    config = rm.load_config(run_name)
    return str(config.get("obs", {}).get("config_key", "v1"))


def _infer_run_from_path(model_path: Path, rm: RunManager) -> str | None:
    """Try to extract a run name from the model path by matching against known runs."""
    parts = model_path.parts
    runs_root = rm.runs_root.resolve()
    try:
        resolved = model_path.resolve()
        rel = resolved.relative_to(runs_root)
        return rel.parts[0] if rel.parts else None
    except (ValueError, IndexError):
        return None


def _resolve_model(args: argparse.Namespace, rm: RunManager) -> tuple[Path, str | None]:
    """Return (model_path, run_name). Exits if nothing can be found."""
    if args.model_path:
        p = Path(args.model_path)
        if not p.is_file():
            sys.exit(f"Model not found: {p}")
        # Prefer explicit --run; fall back to inferring from the path
        run_name = args.run or _infer_run_from_path(p, rm)
        return p, run_name

    run_name = args.run or rm.find_latest_run()
    if run_name is None:
        sys.exit("No runs found. Train a model first or pass a model path directly.")
    if not rm.exists(run_name):
        sys.exit(f"Run '{run_name}' not found.")

    # Prefer eval-ranked best; fall back to highest timestep checkpoint.
    model_path = _best_checkpoint_from_eval(run_name, rm) or rm.find_latest_checkpoint(run_name)
    if model_path is None:
        sys.exit(f"No checkpoints found in run '{run_name}'.")
    return model_path, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a PPO model drive in BeamNG.")
    parser.add_argument("model_path", nargs="?", type=Path,
                        help="Path to .zip model or checkpoint (optional if --run is given).")
    parser.add_argument("--run", metavar="NAME",
                        help="Run name to load the best checkpoint from.")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max episode steps to run.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip the interactive pause at end (for GUI launcher).")
    parser.add_argument("--speed-factor", type=int, default=None,
                        help="Override physics speed factor (1 = real time for recording). "
                             "Default: use value from run_config.json.")
    args = parser.parse_args()

    rm = RunManager()
    model_path, run_name = _resolve_model(args, rm)

    vehicle_model = _vehicle_model_for_run(run_name, rm)
    obs_config = _obs_config_for_run(run_name, rm)

    # Read timing/speed from run_config so eval physics matches training physics
    run_cfg = rm.load_config(run_name) if run_name else {}
    env_cfg = run_cfg.get("env", {}) if isinstance(run_cfg.get("env"), dict) else {}
    speed_factor: int | None = None
    try:
        sf = env_cfg.get("speed_factor")
        if sf is not None:
            speed_factor = int(sf)
    except (TypeError, ValueError):
        pass
    timing_profile: str | None = str(env_cfg.get("timing_profile", "")).strip().lower() or None
    if timing_profile not in ("legacy_60hz", "det50ms"):
        timing_profile = None
    steps_per_action = int(env_cfg.get("steps_per_action", 15))
    # CLI --speed-factor overrides run_config (e.g. force 1x for recording)
    if args.speed_factor is not None:
        speed_factor = args.speed_factor

    print(f"Run     : {run_name or '(direct path)'}")
    print(f"Model   : {model_path.name}")
    print(f"Vehicle : {vehicle_model}")
    print(f"Obs     : {obs_config}")
    print(f"Timing  : {timing_profile or 'default'}  Speed: {speed_factor or 1}x")
    print("Launching BeamNG — watch the car drive in the BeamNG window.")

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            vehicle_model=vehicle_model,
            obs_config=obs_config,
            steps_per_action=steps_per_action,
            timing_profile=timing_profile,
            speed_factor=speed_factor,
            max_episode_steps=args.steps,
            max_lateral_error_m=10.0,
        )

        model = PPO.load(str(model_path))
        obs, info = env.reset()
        positions: list[tuple[float, float, float]] = []
        total_reward = 0.0

        print(f"Running episode (max {args.steps} steps)...")

        for step in range(1, args.steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            pos = info.get("vehicle_pos")
            if pos is not None:
                positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

            if terminated or truncated:
                print(f"Episode ended at step {step}: {info.get('termination_reason', 'unknown')}")
                break

        progress_m = float(info.get("episode_progress_m", 0.0))
        print(f"\n=== Episode Summary ===")
        print(f"  Progress     : {progress_m:.1f} m  (heuristic ~965 m / full lap ~2158 m)")
        print(f"  Total reward : {total_reward:.2f}")
        print(f"  Positions    : {len(positions)} logged")

        if positions and env.beamng is not None:
            print("\nDrawing driven path on track...")
            drawer = DebugPathDrawer(env.beamng)
            drawer.draw_path(
                positions,
                line_color=(1.0, 0.3, 0.1, 1.0),
                sphere_color=(1.0, 0.6, 0.0, 1.0),
                sphere_every=10,
                offset=0.5,
            )
            print("Path drawn. Look around in BeamNG to review the trajectory.")
        elif not positions:
            print("No positions logged — path drawing skipped.")

        if args.no_wait:
            import time
            time.sleep(86400)
        else:
            input("\nPress Enter to close BeamNG...")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
