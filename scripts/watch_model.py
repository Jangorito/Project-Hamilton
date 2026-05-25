"""Watch a trained PPO model drive in BeamNG and draw its path on the track.

Loads a checkpoint, runs one deterministic episode with BeamNG visible,
then draws the driven trajectory on the circuit in 3D.

Usage:
    # Best checkpoint from a named run (uses checkpoints_eval.csv if present)
    python scripts/watch_model.py --run v2_smoother

    # Latest checkpoint from the most recent run
    python scripts/watch_model.py

    # Explicit model file
    python scripts/watch_model.py path/to/model.zip

    python scripts/watch_model.py --run v2_smoother --steps 800
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _resolve_model(args: argparse.Namespace, rm: RunManager) -> tuple[Path, str | None]:
    """Return (model_path, run_name). Exits if nothing can be found."""
    if args.model_path:
        p = Path(args.model_path)
        if not p.is_file():
            sys.exit(f"Model not found: {p}")
        return p, args.run

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
    args = parser.parse_args()

    rm = RunManager()
    model_path, run_name = _resolve_model(args, rm)

    print(f"Run     : {run_name or '(direct path)'}")
    print(f"Model   : {model_path.name}")
    print("Launching BeamNG — watch the car drive in the BeamNG window.")

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            steps_per_action=15,
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

        input("\nPress Enter to close BeamNG...")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
