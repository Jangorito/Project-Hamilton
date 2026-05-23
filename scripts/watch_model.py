"""Watch a trained PPO model drive in BeamNG and draw its path on the track.

Loads a checkpoint, runs one deterministic episode with BeamNG visible,
then draws the driven trajectory on the circuit in 3D.

Usage:
    python scripts/watch_model.py models/checkpoints/ppo_XXX_steps.zip
    python scripts/watch_model.py models/live_ppo_20260523_015728.zip --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv
from beamng_rl.visualisation.debug_draw_path import DebugPathDrawer

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a PPO model drive in BeamNG.")
    parser.add_argument("model_path", type=Path, help="Path to .zip model or checkpoint")
    parser.add_argument("--steps", type=int, default=500, help="Max episode steps to run")
    args = parser.parse_args()

    if not args.model_path.is_file():
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    print(f"Loading model: {args.model_path}")
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

        model = PPO.load(str(args.model_path))

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
        print(f"  Progress     : {progress_m:.1f} m  (heuristic baseline: ~965 m)")
        print(f"  Total reward : {total_reward:.2f}")
        print(f"  Positions    : {len(positions)} logged")

        # Draw the driven path on the track in BeamNG.
        if positions and env.beamng is not None:
            print("\nDrawing driven path on track...")
            drawer = DebugPathDrawer(env.beamng)
            drawer.draw_path(
                positions,
                line_color=(1.0, 0.3, 0.1, 1.0),   # orange — distinct from blue centreline
                sphere_color=(1.0, 0.6, 0.0, 1.0),
                sphere_every=10,
                offset=0.5,  # lift slightly above road surface
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
