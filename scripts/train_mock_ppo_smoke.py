from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from stable_baselines3 import PPO
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required to run this mock PPO smoke test. "
        "Install project requirements with:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        "or install it directly with:\n"
        "  .\\venv\\Scripts\\python.exe -m pip install \"stable-baselines3>=2.3.0\""
    ) from exc

from beamng_rl.envs.beamng_racing_env import BeamNGRacingEnv


def main() -> None:
    centreline_path = (
        REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )
    model_path = REPO_ROOT / "models" / "mock_ppo_smoke.zip"

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(centreline_path, use_mock=True)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=64,
            batch_size=32,
            gamma=0.99,
        )

        model.learn(total_timesteps=256)

        model_path.parent.mkdir(exist_ok=True)
        model.save(model_path)
        print(f"Saved mock PPO smoke model to: {model_path}")

        obs, info = env.reset()
        for step_index in range(1, 11):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_info = info["reward"]

            print(
                f"step={step_index} "
                f"reward={reward:.4f} "
                f"episode_progress_m={info['episode_progress_m']:.4f} "
                f"progress_delta_m={reward_info['progress_delta_m']:.4f} "
                f"termination_reason={info['termination_reason']}"
            )

            if terminated or truncated:
                obs, info = env.reset()

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
