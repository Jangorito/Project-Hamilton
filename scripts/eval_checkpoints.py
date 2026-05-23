"""Evaluate all saved checkpoints and find the best one.

Run this after a training session to compare checkpoint performance and
identify which model to visualise with watch_model.py.

Usage:
    python scripts/eval_checkpoints.py
    python scripts/eval_checkpoints.py --checkpoint-dir models/checkpoints
    python scripts/eval_checkpoints.py --steps 200
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
LOG_DIR = REPO_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
DEFAULT_EVAL_STEPS = 500

CSV_FIELDS = ["checkpoint", "timestep", "progress_m", "termination_reason", "progress_jumps"]


def _parse_timestep(path: Path) -> int:
    """Extract step count from SB3 checkpoint filename e.g. ppo_20260523_8000_steps.zip."""
    match = re.search(r"_(\d+)_steps", path.stem)
    return int(match.group(1)) if match else 0


def _run_rollout(env: BeamNGRacingEnv, model: PPO, n_steps: int) -> tuple[float, str, int, list]:
    obs, info = env.reset()
    progress_m = 0.0
    reason = "none"
    jumps = 0
    positions: list[tuple[float, float, float]] = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        pos = info.get("vehicle_pos")
        if pos is not None:
            positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

        reward_info = info.get("reward", {})
        if reward_info.get("progress_jump_detected", False):
            jumps += 1

        if terminated or truncated:
            progress_m = float(info.get("episode_progress_m", 0.0))
            reason = str(info.get("termination_reason", "unknown"))
            break

    return progress_m, reason, jumps, positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BeamNG PPO checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--steps", type=int, default=DEFAULT_EVAL_STEPS,
                        help="Max rollout steps per checkpoint")
    args = parser.parse_args()

    checkpoints = sorted(args.checkpoint_dir.glob("*.zip"), key=_parse_timestep)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints. Launching BeamNG...")

    out_csv = LOG_DIR / "checkpoints_eval.csv"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env: BeamNGRacingEnv | None = None
    results = []

    try:
        env = BeamNGRacingEnv(
            CENTRELINE_PATH,
            use_mock=False,
            launch_beamng=True,
            live_spawn_mode="bootstrap",
            vehicle_id="ego_vehicle",
            steps_per_action=15,
            max_episode_steps=500,
            max_lateral_error_m=10.0,
        )

        for ckpt in checkpoints:
            timestep = _parse_timestep(ckpt)
            print(f"\nEvaluating {ckpt.name} (t={timestep})...")
            model = PPO.load(str(ckpt))
            progress_m, reason, jumps, _ = _run_rollout(env, model, args.steps)
            print(f"  progress={progress_m:.1f}m  reason={reason}  jumps={jumps}")
            results.append({
                "checkpoint": str(ckpt),
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
        print(f"\nBest checkpoint:")
        print(f"  {best['checkpoint']}")
        print(f"  progress={best['progress_m']:.1f}m  at timestep={best['timestep']}")
        print(f"\nTo visualise: python scripts/watch_model.py \"{best['checkpoint']}\"")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
