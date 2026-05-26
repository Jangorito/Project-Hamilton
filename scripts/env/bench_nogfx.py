"""Benchmark live BeamNG step throughput with and without nogfx.

Runs a fixed number of random-action steps in each mode and reports
steps/sec so you can estimate real training wall time before committing
to a long run.

Usage:
    python scripts/bench_nogfx.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv

CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"

# Number of policy steps to time. Enough to get a stable rate; not so many
# that you're sitting here for 10 minutes per mode.
BENCH_STEPS = 200

# Match the live training script so the comparison is meaningful.
STEPS_PER_ACTION = 15
MAX_EPISODE_STEPS = 500


def kill_beamng() -> None:
    """Kill any stale BeamNG processes before launching a fresh instance."""
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "BeamNG.tech.x64.exe"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("Killed stale BeamNG process.")
        time.sleep(2)  # give OS time to release the port


def run_bench(nogfx: bool) -> float:
    """Return policy steps/sec for one timed episode in the given mode."""
    label = "nogfx=True " if nogfx else "nogfx=False"
    print(f"\n--- {label} ---")
    kill_beamng()
    print("Launching BeamNG... (this may take 1-3 min for Hirochi to load)")

    env = BeamNGRacingEnv(
        CENTRELINE_PATH,
        use_mock=False,
        launch_beamng=True,
        live_spawn_mode="bootstrap",
        vehicle_id="ego_vehicle",
        steps_per_action=STEPS_PER_ACTION,
        max_episode_steps=MAX_EPISODE_STEPS,
        nogfx=nogfx,
    )

    try:
        env.reset()
        print(f"Running {BENCH_STEPS} steps...")

        t0 = time.perf_counter()
        for i in range(BENCH_STEPS):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - t0

        sps = BENCH_STEPS / elapsed
        print(f"  {BENCH_STEPS} policy steps in {elapsed:.1f}s  =>  {sps:.1f} steps/sec")
        print(f"  100k steps estimate: {100_000 / sps / 3600:.1f} hours")
        return sps

    finally:
        env.close()
        print("BeamNG closed.")


def main() -> None:
    results: dict[str, float] = {}

    for nogfx in (False, True):
        results["nogfx=True" if nogfx else "nogfx=False"] = run_bench(nogfx)

    print("\n=== Summary ===")
    for label, sps in results.items():
        print(f"  {label}: {sps:.1f} steps/sec  ({100_000 / sps / 3600:.1f}h for 100k steps)")

    sps_off = results.get("nogfx=False", 1)
    sps_on  = results.get("nogfx=True", 1)
    print(f"\n  Speedup: {sps_on / sps_off:.2f}x")


if __name__ == "__main__":
    main()
