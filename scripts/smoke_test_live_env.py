from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


def main() -> None:
    """Launch BeamNG.tech and take a few gentle live environment steps."""

    centreline_path = (
        REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(
            centreline_path,
            use_mock=False,
            launch_beamng=True,
            vehicle_id="ego_vehicle",
        )
        observation, info = env.reset()
        print(f"Initial observation shape: {observation.shape}")
        print(
            "Initial progress: "
            f"{info['progress_m']:.3f} m ({info['progress_ratio']:.4f})"
        )

        action = np.asarray([0.0, 0.25], dtype=np.float32)
        for step_index in range(1, 11):
            observation, reward, terminated, truncated, info = env.step(action)
            reward_info = info["reward"]
            print(
                "Step "
                f"{step_index}: obs_shape={observation.shape}, "
                f"progress={info['progress_m']:.3f} m, "
                f"reward={reward:.4f}, "
                f"progress_delta={reward_info['progress_delta_m']:.4f}, "
                f"speed_reward={reward_info['speed_reward']:.4f}, "
                f"heading_penalty={reward_info['heading_penalty']:.4f}, "
                f"lateral_penalty={reward_info['lateral_penalty']:.4f}, "
                f"termination={info['termination_reason']}"
            )

            if terminated or truncated:
                break
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
