"""Evaluate all saved checkpoints for a run and find the best one.

Run this after a training session to compare checkpoint performance and
identify which model to visualise with watch_model.py.

Usage:
    python scripts/eval_checkpoints.py                      # latest run
    python scripts/eval_checkpoints.py --run v2_smoother   # specific run
    python scripts/eval_checkpoints.py --steps 300         # shorter rollouts
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
from beamng_rl.training.run_manager import RunManager

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
DEFAULT_EVAL_STEPS = 500

CSV_FIELDS = ["checkpoint", "timestep", "progress_m", "termination_reason", "progress_jumps"]


def _run_rollout(env: BeamNGRacingEnv, model: PPO, n_steps: int) -> tuple[float, str, int]:
    obs, info = env.reset()
    progress_m = 0.0
    reason = "none"
    jumps = 0

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        reward_info = info.get("reward", {})
        if reward_info.get("progress_jump_detected", False):
            jumps += 1

        if terminated or truncated:
            progress_m = float(info.get("episode_progress_m", 0.0))
            reason = str(info.get("termination_reason", "unknown"))
            break

    return progress_m, reason, jumps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BeamNG PPO checkpoints.")
    parser.add_argument("--run", metavar="NAME", help="Run name to evaluate.")
    parser.add_argument("--steps", type=int, default=DEFAULT_EVAL_STEPS,
                        help="Max rollout steps per checkpoint.")
    args = parser.parse_args()

    rm = RunManager()

    run_name = args.run or rm.find_latest_run()
    if run_name is None:
        sys.exit("No runs found. Train a model first.")
    if not rm.exists(run_name):
        sys.exit(f"Run '{run_name}' not found.")

    checkpoints = rm.find_all_checkpoints(run_name)
    if not checkpoints:
        sys.exit(f"No checkpoints found in run '{run_name}'.")

    out_csv = rm.log_dir(run_name) / "checkpoints_eval.csv"
    rm.log_dir(run_name).mkdir(parents=True, exist_ok=True)

    print(f"Run      : {run_name}")
    print(f"Found    : {len(checkpoints)} checkpoints")
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
            steps_per_action=15,
            max_episode_steps=500,
            max_lateral_error_m=10.0,
        )

        for ckpt in checkpoints:
            from beamng_rl.training.run_manager import _parse_timestep
            timestep = _parse_timestep(ckpt)
            print(f"\nEvaluating {ckpt.name} (t={timestep})...")
            model = PPO.load(str(ckpt))
            progress_m, reason, jumps = _run_rollout(env, model, args.steps)
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
        print(f"  {Path(best['checkpoint']).name}")
        print(f"  progress={best['progress_m']:.1f} m  at timestep={best['timestep']}")
        print(f"\nTo visualise:")
        print(f"  python scripts/watch_model.py --run {run_name}")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
