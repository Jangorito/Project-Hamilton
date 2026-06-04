"""Custom SB3 callbacks for BeamNG RL training."""

from __future__ import annotations

import math
import time
from collections import deque
from collections import defaultdict
from pathlib import Path
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
    "raceline_lateral_penalty",
    "raceline_overspeed_penalty",
    "raceline_slow_speed_penalty",
    "raceline_baseline_speed_bonus",
    "total_reward",
)

# Numeric fields inside info["reward"] logged under env/ prefix.
_REWARD_INFO_ENV_KEYS = (
    "forward_speed_mps",
    "lateral_error_m",
    "heading_error_rad",
    "max_curvature_ahead",
    "curvature_target_speed",
    "raceline_lateral_error_m",
    "raceline_lateral_excess_m",
    "raceline_target_speed_mps",
    "raceline_speed_gate",
    "raceline_heading_gate",
    "raceline_decay_scale",
    "raceline_effective_scale",
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


class StopSignalCallback(BaseCallback):
    """Gracefully stops training when the launcher writes a stop signal file.

    Returning False from _on_step tells SB3 to stop after the current step,
    letting model.learn() return normally so the training script can save the
    model and run the deterministic rollout before exiting.
    """

    def __init__(self, signal_file: Path | str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._signal_file = Path(signal_file)

    def _on_step(self) -> bool:
        return not self._signal_file.exists()


class StepProgressWriter(BaseCallback):
    """Writes num_timesteps to a file after each rollout so the launcher UI shows real progress.

    The file contains a single integer (the current timestep count). The launcher
    reads it via /api/status instead of inferring progress from checkpoint filenames,
    which only update every CHECKPOINT_EVERY steps and miss the final steps entirely.
    """

    def __init__(self, path: Path | str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._path = Path(path)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(str(self.num_timesteps), encoding="utf-8")
        except OSError:
            pass


class BeamNGTrainingHudCallback(BaseCallback):
    """Push live PPO training metrics into the BeamNG UI training HUD."""

    def __init__(
        self,
        *,
        hud: Any,
        total_timesteps: int,
        run_name: str,
        vehicle_model: str,
        vehicle_label: str,
        reward_config: str,
        speed_factor: int | None,
        timing_profile: str,
        max_episode_steps: int,
        full_lap_m: float,
        update_every_steps: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._hud = hud
        self._total_timesteps = int(total_timesteps)
        self._run_name = run_name
        self._vehicle_model = vehicle_model
        self._vehicle_label = vehicle_label
        self._reward_config = reward_config
        self._speed_factor = int(speed_factor or 1)
        self._timing_profile = timing_profile
        self._max_episode_steps = int(max_episode_steps)
        self._full_lap_m = max(1.0, float(full_lap_m))
        self._update_every_steps = max(1, int(update_every_steps))
        # Re-assert the layout every N steps so it survives scenario reloads
        # between episodes.  BeamNG wipes the UI layout on each reset.
        self._layout_reassert_every = 60
        self._last_layout_step = -1
        self._started_at = 0.0
        self._last_sent_step = -1
        self._episodes_completed = 0
        self._terminations_total = 0
        self._termination_counts: dict[str, int] = defaultdict(int)
        self._latest_termination = "none"
        self._best_episode_progress_m = 0.0
        self._progress_jumps = 0
        self._reward_window: deque[float] = deque(maxlen=128)

    def _on_training_start(self) -> None:
        self._started_at = time.monotonic()
        show_layout = getattr(self._hud, "show_layout", None)
        if callable(show_layout):
            show_layout()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        if not infos:
            return True

        rewards = list(self.locals.get("rewards", []))
        dones = list(self.locals.get("dones", []))

        reward_value = _safe_float(rewards[-1] if rewards else 0.0)
        self._reward_window.append(reward_value)

        latest_info = infos[-1]
        latest_done = False
        for index, info in enumerate(infos):
            done = bool(dones[index]) if index < len(dones) else False
            latest_done = latest_done or done

            if info.get("progress_jump_detected", False):
                self._progress_jumps += 1

            progress_m = _safe_float(info.get("episode_progress_m"))
            self._best_episode_progress_m = max(self._best_episode_progress_m, progress_m)

            if done:
                reason = str(info.get("termination_reason", "none") or "none")
                if reason == "none":
                    reason = "episode_end"
                self._latest_termination = reason
                self._termination_counts[reason] += 1
                self._terminations_total += 1
                self._episodes_completed += 1

        # Re-assert the HUD layout periodically so it survives BeamNG
        # scenario reloads that happen on each episode reset.
        if self.num_timesteps - self._last_layout_step >= self._layout_reassert_every:
            show_layout = getattr(self._hud, "show_layout", None)
            if callable(show_layout):
                show_layout()
            self._last_layout_step = int(self.num_timesteps)

        if (
            self.num_timesteps - self._last_sent_step < self._update_every_steps
            and not latest_done
        ):
            return True

        self._last_sent_step = int(self.num_timesteps)
        self._send_payload(latest_info, reward_value)
        return True

    def _on_training_end(self) -> None:
        send_status = getattr(self._hud, "send_status", None)
        if callable(send_status):
            send_status("ended", active=False)

    def _send_payload(self, info: dict[str, Any], reward_value: float) -> None:
        reward_info = info.get("reward", {})
        if not isinstance(reward_info, dict):
            reward_info = {}
        control = info.get("control", {})
        if not isinstance(control, dict):
            control = {}

        elapsed = max(0.001, time.monotonic() - self._started_at)
        current_step = int(self.num_timesteps)
        steps_per_second = current_step / elapsed
        remaining_steps = max(0, self._total_timesteps - current_step)
        eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0.0

        progress_m = _safe_float(info.get("episode_progress_m"))
        lap_progress_percent = max(0.0, min(100.0, progress_m / self._full_lap_m * 100.0))
        heading_rad = _safe_float(
            info.get("heading_error_rad", reward_info.get("heading_error_rad"))
        )
        speed_mps = _safe_float(reward_info.get("forward_speed_mps"))
        reward_mean = (
            sum(self._reward_window) / len(self._reward_window)
            if self._reward_window
            else reward_value
        )

        payload = {
            "has_data": True,
            "active": True,
            "status": "training",
            "run_name": self._run_name,
            "vehicle_model": self._vehicle_model,
            "vehicle_label": self._vehicle_label,
            "reward_config": self._reward_config,
            "speed_factor": self._speed_factor,
            "timing_profile": self._timing_profile,
            "current_step": current_step,
            "total_steps": self._total_timesteps,
            "step_percent": _percent(current_step, self._total_timesteps),
            "steps_per_second": steps_per_second,
            "eta_seconds": eta_seconds,
            "episode_number": self._episodes_completed + 1,
            "episodes_completed": self._episodes_completed,
            "episode_step": int(_safe_float(info.get("step"))),
            "max_episode_steps": self._max_episode_steps,
            "episode_progress_m": progress_m,
            "full_lap_m": self._full_lap_m,
            "lap_progress_percent": lap_progress_percent,
            "best_episode_progress_m": self._best_episode_progress_m,
            "reward": reward_value,
            "reward_mean": reward_mean,
            "progress_reward": _safe_float(reward_info.get("progress_reward")),
            "speed_reward": _safe_float(reward_info.get("speed_reward")),
            "progress_delta_m": _safe_float(reward_info.get("progress_delta_m")),
            "speed_mps": speed_mps,
            "speed_kph": speed_mps * 3.6,
            "lateral_error_m": _safe_float(
                info.get("lateral_error_m", reward_info.get("lateral_error_m"))
            ),
            "heading_error_deg": math.degrees(heading_rad),
            "vehicle_damage": _safe_float(info.get("vehicle_damage")),
            "stuck_steps": int(_safe_float(info.get("stuck_steps"))),
            "progress_jumps": self._progress_jumps,
            "steering": _safe_float(control.get("steering")),
            "throttle": _safe_float(control.get("throttle")),
            "brake": _safe_float(control.get("brake")),
            "terminations_total": self._terminations_total,
            "latest_termination": self._latest_termination,
            "term_counts": {
                "off_track": self._termination_counts.get("off_track", 0),
                "damage": self._termination_counts.get("damage", 0),
                "stuck": self._termination_counts.get("stuck", 0),
                "wall_bash": self._termination_counts.get("wall_bash", 0),
                "max_episode_steps": self._termination_counts.get("max_episode_steps", 0),
                "lap_completed": self._termination_counts.get("lap_completed", 0),
            },
            # V2.1 physics reward diagnostics — None when using v1/v2 configs
            "traction_budget_penalty": _safe_float(reward_info.get("traction_budget_penalty")) or None,
            "braking_shortfall_penalty": _safe_float(reward_info.get("braking_shortfall_penalty")) or None,
            "lateral_risk_ratio": _safe_float(reward_info.get("lateral_risk_ratio")) or None,
            "physics_target_speed_mps": _safe_float(reward_info.get("physics_target_speed_mps")) or None,
        }

        send = getattr(self._hud, "send", None)
        if callable(send):
            send(payload)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _percent(value: float, total: float) -> float:
    total = max(1.0, float(total))
    return max(0.0, min(100.0, float(value) / total * 100.0))
