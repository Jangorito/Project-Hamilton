"""Custom SB3 callbacks for BeamNG RL training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

# Reward component keys logged under reward/ prefix in TensorBoard.
_REWARD_KEYS = (
    "progress_reward",
    "speed_reward",
    "heading_penalty",
    "lateral_penalty",
    "off_track_penalty",
    "reverse_progress_penalty",
    "stuck_penalty",
    "smoothness_penalty",
    "curvature_overspeed_penalty",
    "progress_jump_penalty",
    "total_reward",
)

# Numeric fields inside info["reward"] logged under env/ prefix.
_REWARD_INFO_ENV_KEYS = (
    "forward_speed_mps",
    "lateral_error_m",
    "heading_error_rad",
    "max_curvature_ahead",
    "curvature_target_speed",
)

# Top-level info keys logged under env/ prefix.
_TOP_LEVEL_ENV_KEYS = (
    "episode_progress_m",
    "vehicle_damage",
    "stuck_steps",
)


class RewardComponentLogger(BaseCallback):
    """Logs individual reward components and env metrics to TensorBoard.

    Accumulates values over each rollout (n_steps environment steps) then
    writes per-component means at rollout end. This gives you separate
    TensorBoard curves for progress_reward, curvature_overspeed_penalty,
    smoothness_penalty, etc. so you can see which terms are dominating and
    whether the braking/smoothness penalties are actually firing.

    Termination reasons are logged as per-step rates under env/term_<reason>.

    Add to model.learn():
        model.learn(..., callback=[checkpoint_cb, RewardComponentLogger()])
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._sums: dict[str, float] = defaultdict(float)
        self._term_counts: dict[str, int] = defaultdict(int)
        self._n: int = 0

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            reward_info = info.get("reward", {})

            for key in _REWARD_KEYS:
                val = reward_info.get(key)
                if val is not None:
                    self._sums[f"reward/{key}"] += float(val)

            for key in _REWARD_INFO_ENV_KEYS:
                val = reward_info.get(key)
                if val is not None:
                    self._sums[f"env/{key}"] += float(val)

            for key in _TOP_LEVEL_ENV_KEYS:
                val = info.get(key)
                if val is not None:
                    self._sums[f"env/{key}"] += float(val)

            reason = info.get("termination_reason", "none")
            if reason and reason != "none":
                self._term_counts[reason] += 1

            self._n += 1

        return True

    def _on_rollout_end(self) -> None:
        if self._n == 0:
            return

        for key, total in self._sums.items():
            self.logger.record(key, total / self._n)

        for reason, count in self._term_counts.items():
            self.logger.record(f"env/term_{reason}_rate", count / self._n)

        self._sums = defaultdict(float)
        self._term_counts = defaultdict(int)
        self._n = 0
