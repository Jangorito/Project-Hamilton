from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from stable_baselines3.common.env_checker import check_env
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required to run this compatibility check. "
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

    env: BeamNGRacingEnv | None = None
    try:
        env = BeamNGRacingEnv(centreline_path, use_mock=True)
        check_env(env, warn=True)
        print("Stable-Baselines3 environment compatibility check passed.")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
