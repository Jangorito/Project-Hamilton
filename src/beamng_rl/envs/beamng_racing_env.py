"""Minimal Gymnasium racing environment for BeamNG RL experiments.

This module creates the Gymnasium-compatible wrapper around the track-query and
observation-building code that already exists in the project. Mock mode remains
the safe default for unit and smoke testing; live mode uses the manual
bootstrap's Hirochi/SBR setup and defaults to that known-good spawn pose while
allowing centreline-aligned reset poses for RL experiments.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

try:
    from beamng_rl.bootstrap.beamng_setup import (
        HOST,
        PORT,
        SPAWN_POS,
        SPAWN_ROT,
        apply_low_graphics_preset,
        apply_shadow_disabling,
        build_hirochi_etkc_scenario,
        build_hirochi_sbr_scenario,
        resolve_beamng_home_from_path_or_env,
    )
    from beamng_rl.envs.observation_builder import ObservationBuilder
    from beamng_rl.track.query_utils import TrackCentreline, TrackQueryResult
except ModuleNotFoundError as exc:
    # Running this file directly with
    #   python src/beamng_rl/envs/beamng_racing_env.py
    # puts ``src/beamng_rl/envs`` on sys.path, not the repository ``src``
    # directory. This fallback keeps normal package imports unchanged while
    # making the repository-root smoke test convenient.
    if exc.name != "beamng_rl":
        raise

    import sys

    src_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src_root))
    from beamng_rl.bootstrap.beamng_setup import (
        HOST,
        PORT,
        SPAWN_POS,
        SPAWN_ROT,
        apply_low_graphics_preset,
        apply_shadow_disabling,
        build_hirochi_etkc_scenario,
        build_hirochi_sbr_scenario,
        resolve_beamng_home_from_path_or_env,
    )
    from beamng_rl.envs.observation_builder import ObservationBuilder
    from beamng_rl.track.query_utils import TrackCentreline, TrackQueryResult


@dataclass
class RewardConfig:
    """Tunable constants for the first inspectable racing reward."""

    progress_weight: float = 1.0
    speed_weight: float = 0.05
    heading_error_weight: float = 0.2
    lateral_error_weight: float = 0.1
    off_track_penalty: float = 10.0
    reverse_progress_penalty: float = 2.0
    stuck_step_penalty: float = 0.05
    action_smoothness_weight: float = 0.1  # penalises |action_t - action_{t-1}| per dimension

    # Curvature-aware speed target. Penalises carrying excess speed into corners.
    # target_speed = clamp(base - scale * max_curvature_ahead, min, base)
    # penalty = weight * max(0, forward_speed - target_speed)
    curvature_overspeed_weight: float = 0.15    # turn it up if it's still not braking enough
    curvature_target_speed_base: float = 35.0   # m/s — target on a straight
    curvature_target_speed_min: float = 12.0    # m/s — floor for tight hairpins
    curvature_speed_scale: float = 700.0        # m/s per (1/m) of curvature


# ---------------------------------------------------------------------------
# Reward config registry — named presets for controlled experiments.
#
# Both V1 and V2 run the same car (etkc) on the same track. Selecting by key
# rather than hand-editing values ensures both conditions are always
# launchable from the same codebase with a single flag change, with no risk
# of accidentally running the wrong reward between experiment arms.
#
# Usage:  reward_config = REWARD_CONFIGS["v1"]
# ---------------------------------------------------------------------------
REWARD_CONFIGS: dict[str, RewardConfig] = {
    # V1 — baseline reward, matches the RewardConfig dataclass defaults.
    # Used as the control condition in the V1/V2 experiment.
    "v1": RewardConfig(
        progress_weight=1.0,
        speed_weight=0.05,            # flat speed bonus — encourages forward velocity but subsidises corner entry speed
        heading_error_weight=0.2,
        lateral_error_weight=0.1,     # full centreline constraint
        off_track_penalty=10.0,
        reverse_progress_penalty=2.0,
        stuck_step_penalty=0.05,
        action_smoothness_weight=0.1,    # weak smoothness — insufficient to prevent steering oscillation
        curvature_overspeed_weight=0.15, # partially cancelled by speed_weight subsidy
        curvature_target_speed_base=35.0,
        curvature_target_speed_min=12.0,
        curvature_speed_scale=700.0,
    ),
    # V2 — literature-informed tweaks. Four targeted changes from V1:
    #
    # 1. speed_weight removed (0.05 -> 0.0):
    #    GT Sport RL paper (Fuchs et al.) used progress-only reward successfully.
    #    Progress implicitly rewards speed; an explicit speed bonus creates a
    #    subsidy that partially cancels the curvature overspeed penalty on corner
    #    entry.
    #
    # 2. action_smoothness_weight raised (0.1 -> 0.4):
    #    TORCS-based racing RL literature (Guckiran & Bolat) identifies fast
    #    left-right steering oscillation (slaloming) as a reward function problem.
    #    SBR4 rollout data confirms full-lock steering swings every 1-3 steps;
    #    the 0.1 penalty costs ~0.2 reward vs ~4-6 from progress — effectively
    #    ignored. Raising to 0.4 makes jerk meaningfully costly.
    #
    # 3. lateral_error_weight halved (0.1 -> 0.05):
    #    Racing RL literature notes that strict centreline constraint can prevent
    #    the agent from discovering wider out-in-out racing lines. Relaxing this
    #    allows more lateral freedom while the heading penalty still discourages
    #    spinning.
    #
    # 4. curvature_overspeed_weight raised (0.15 -> 0.3):
    #    With speed_weight removed the curvature penalty no longer fights a
    #    subsidy, so it can be stronger without over-penalising straight-line
    #    speed.
    "v2": RewardConfig(
        progress_weight=1.0,
        speed_weight=0.0,
        heading_error_weight=0.2,
        lateral_error_weight=0.05,
        off_track_penalty=10.0,
        reverse_progress_penalty=2.0,
        stuck_step_penalty=0.05,
        action_smoothness_weight=0.4,
        curvature_overspeed_weight=0.3,
        curvature_target_speed_base=35.0,
        curvature_target_speed_min=12.0,
        curvature_speed_scale=700.0,
    ),
}


@dataclass
class TerminationResult:
    """Inspectable outcome of the environment termination checks."""

    terminated: bool
    truncated: bool
    reason: str
    off_track: bool
    stuck: bool
    max_steps_reached: bool
    lap_completed: bool = False
    damage: bool = False
    wall_bash: bool = False


class BeamNGRacingEnv(gym.Env):
    """A small continuous-control racing environment shell.

    The environment already exposes the Gymnasium API expected by RL libraries:
    ``reset()``, ``step()``, ``observation_space``, and ``action_space``. The
    simulator-facing methods stay deliberately separated so live BeamNGpy setup
    can evolve without rewriting reward or observation logic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track_json_path: str | Path,
        max_episode_steps: int = 2000,
        max_lateral_error_m: float = 8.0,
        stuck_steps_limit: int = 100,
        min_progress_delta_m: float = 0.05,
        max_progress_delta_m: float = 50.0,
        progress_jump_penalty: float = 5.0,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        use_mock: bool = True,
        beamng_home: str | Path | None = None,
        launch_beamng: bool = True,
        apply_low_graphics: bool = False,
        disable_shadows: bool = False,
        beamng_host: str = HOST,
        beamng_port: int = PORT,
        scenario_name: str = "hirochi_raceway",
        vehicle_id: str = "ego_vehicle",
        steps_per_action: int = 3,
        live_spawn_mode: str = "bootstrap",
        live_spawn_progress_m: float = 0.0,
        live_spawn_lateral_offset_m: float = 0.0,
        live_spawn_z_offset_m: float = 0.5,
        # Set nogfx=True to run BeamNG with a null graphics backend (no GPU
        # rendering). Speeds up physics throughput significantly — benchmark
        # with True vs False before committing to a long training run.
        # Camera-based sensors are unavailable in nogfx mode, but the 12-feature
        # obs uses only the vehicle state sensor so nothing is lost here.
        nogfx: bool = False,
        max_damage: float = 500.0,
        wall_bash_steps_limit: int = 30,
        vehicle_model: str = "etkc",
        speed_factor: int | None = None,
    ) -> None:
        super().__init__()

        live_spawn_mode_value = str(live_spawn_mode)
        if live_spawn_mode_value not in ("bootstrap", "centreline"):
            raise ValueError(
                "live_spawn_mode must be one of 'bootstrap' or 'centreline', "
                f"got {live_spawn_mode!r}"
            )

        # TrackCentreline owns the geometric track queries. Keeping it as a
        # separate object makes the environment easy to test without BeamNG.
        self.track = TrackCentreline.from_json(track_json_path, closed_loop=True)

        # ObservationBuilder converts raw vehicle-state dictionaries into the
        # fixed 12-feature vector already smoke-tested elsewhere in the project.
        self.observation_builder = ObservationBuilder(self.track)

        # Reuse the builder's bounds so the Gymnasium space stays in sync with
        # the observation implementation if feature limits change later.
        self.observation_space = gym.spaces.Box(
            low=self.observation_builder.low(),
            high=self.observation_builder.high(),
            dtype=np.float32,
        )

        # Continuous action format:
        #   action[0] = steering in [-1, 1]
        #   action[1] = combined throttle/brake in [-1, 1]
        # Positive throttle/brake accelerates; negative throttle/brake brakes.
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode and reward settings. These are environment-level because they
        # describe failure conditions and training-horizon limits.
        self.max_episode_steps = int(max_episode_steps)
        self.max_lateral_error_m = float(max_lateral_error_m)
        self.stuck_steps_limit = int(stuck_steps_limit)
        self.min_progress_delta_m = float(min_progress_delta_m)
        self.max_progress_delta_m = float(max_progress_delta_m)
        self.progress_jump_penalty = float(progress_jump_penalty)
        self.render_mode = render_mode
        self.reward_config = reward_config or RewardConfig()

        # Simulation backend selection. The mock backend is the default because
        # it keeps imports, CI-style smoke tests, and reward debugging possible
        # on machines where BeamNG.tech is not running.
        self.use_mock = bool(use_mock)
        self.beamng_home = beamng_home
        self.launch_beamng = bool(launch_beamng)
        self.apply_low_graphics = bool(apply_low_graphics)
        self.disable_shadows = bool(disable_shadows)
        self.beamng_host = str(beamng_host)
        self.beamng_port = int(beamng_port)
        self.live_spawn_mode = live_spawn_mode_value
        self.live_spawn_progress_m = float(live_spawn_progress_m)
        self.live_spawn_lateral_offset_m = float(live_spawn_lateral_offset_m)
        self.live_spawn_z_offset_m = float(live_spawn_z_offset_m)
        # Kept as a public knob for future tracks. The current live setup uses
        # the shared bootstrap Hirochi scenario to avoid drift between debug
        # launches and training launches.
        self.scenario_name = str(scenario_name)
        self.vehicle_id = str(vehicle_id)
        self.steps_per_action = int(steps_per_action)
        if self.steps_per_action <= 0:
            raise ValueError(
                f"steps_per_action must be a positive integer, got {steps_per_action!r}"
            )
        self.nogfx = bool(nogfx)
        self.max_damage = float(max_damage)
        self.wall_bash_steps_limit = int(wall_bash_steps_limit)
        vehicle_model_value = str(vehicle_model).strip().lower()
        if vehicle_model_value not in ("sbr", "etkc"):
            raise ValueError(
                "vehicle_model must be one of 'sbr' or 'etkc', "
                f"got {vehicle_model!r}"
            )
        self.vehicle_model = vehicle_model_value
        self.speed_factor = int(speed_factor) if speed_factor is not None else None
        if self.speed_factor is not None and self.speed_factor not in (1, 2, 4, 8, 16, 32):
            raise ValueError(
                "speed_factor must be one of 1, 2, 4, 8, 16, 32, "
                f"got {speed_factor!r}"
            )

        # Runtime counters are reset in reset(), but initial values keep the
        # object inspectable immediately after construction.
        self.current_step = 0
        self.previous_progress_m = 0.0
        self.previous_raw_progress_m = 0.0
        self.episode_start_progress_m = 0.0
        self.episode_progress_m = 0.0
        self.stuck_steps = 0
        self._wall_bash_steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)
        self.last_observation: np.ndarray | None = None
        self.last_info: dict[str, Any] = {}

        # BeamNGpy integration placeholders. Later these can hold the live
        # BeamNGpy connection and Vehicle object, but importing BeamNGpy here is
        # intentionally avoided so the environment can be imported in tests.
        self.beamng = None
        self.scenario = None
        self.vehicle = None
        self._resolved_beamng_home: Path | None = None

        # Mock-state variables used only while no live BeamNG vehicle is
        # connected. They let step() move around the track during smoke tests.
        self._mock_dt_s = 0.1
        self._mock_progress_m = 0.0
        self._mock_lateral_offset_m = 0.0
        self._mock_heading_error_rad = 0.0
        self._mock_forward_speed_mps = 8.0
        self._mock_vehicle_state: dict[str, Any] | None = None
        self._last_control = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

        if not self.use_mock:
            self._connect_beamng()

    def _split_action(self, action: np.ndarray) -> dict[str, float]:
        """Convert the policy action into BeamNG-style control values."""

        # Clip first so a noisy policy or caller cannot send controls outside
        # the declared Gymnasium action space.
        clipped = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        steering = float(clipped[0])
        throttle_brake = float(clipped[1])

        # Throttle and brake share one axis in the baseline environment because
        # acceleration and braking are mutually exclusive for this first control
        # setup. That reduces exploration complexity. Later, a clean ablation
        # can compare this against separate throttle and brake dimensions.
        if throttle_brake >= 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        return {"steering": steering, "throttle": throttle, "brake": brake}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode state and return the first observation."""

        super().reset(seed=seed)

        self.current_step = 0
        self.previous_progress_m = 0.0
        self.previous_raw_progress_m = 0.0
        self.episode_start_progress_m = 0.0
        self.episode_progress_m = 0.0
        self.stuck_steps = 0
        self._wall_bash_steps = 0
        self._last_action = np.zeros(2, dtype=np.float32)

        # This returns a backend-normalised vehicle-state dictionary. Mock mode
        # uses the safe toy dynamics; live mode polls the BeamNGpy vehicle.
        vehicle_state = self._reset_simulation(options=options)
        observation = self.observation_builder.build(vehicle_state)

        # Store the starting progress so the first reward uses progress made
        # after reset rather than distance from zero on the lap.
        heading_rad = self._heading_from_state(vehicle_state)
        query = self.track.query(vehicle_state["pos"], vehicle_heading_rad=heading_rad)
        raw_start_progress_m = float(query.lap_progress)
        raw_start_progress_ratio = float(query.lap_progress_ratio)
        self.episode_start_progress_m = raw_start_progress_m
        self.previous_progress_m = raw_start_progress_m
        self.previous_raw_progress_m = raw_start_progress_m
        self.episode_progress_m = 0.0

        # progress_m/progress_ratio are raw centreline coordinates. They may
        # start near the wrap boundary in live bootstrap mode. episode_progress_m
        # is the training-facing distance travelled since this reset.
        info: dict[str, Any] = {
            "progress_m": raw_start_progress_m,
            "progress_ratio": raw_start_progress_ratio,
            "raw_progress_m": raw_start_progress_m,
            "raw_progress_ratio": raw_start_progress_ratio,
            "episode_start_progress_m": self.episode_start_progress_m,
            "episode_progress_m": self.episode_progress_m,
            "lateral_error_m": float(query.signed_lateral_error),
            "heading_error_rad": (
                float(query.heading_error_rad)
                if query.heading_error_rad is not None
                else 0.0
            ),
            "mock_simulation": self.use_mock,
            "spawn_mode": self.live_spawn_mode,
            "near_progress_wrap": (
                raw_start_progress_ratio > 0.95
                or raw_start_progress_ratio < 0.05
            ),
            "spawn_lateral_error_abs_m": abs(float(query.signed_lateral_error)),
        }

        self.last_observation = observation
        self.last_info = info
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action, advance the simulation, and return one RL step."""

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_delta = action - self._last_action

        control = self._split_action(action)
        self._apply_action(control)
        self._advance_simulation()

        vehicle_state = self._get_vehicle_state()
        observation = self.observation_builder.build(vehicle_state)

        heading_rad = self._heading_from_state(vehicle_state)
        query = self.track.query(vehicle_state["pos"], vehicle_heading_rad=heading_rad)

        reward, reward_info = self._compute_reward(query, vehicle_state, action_delta=action_delta)

        self._last_action = action

        self.current_step += 1

        termination = self._check_termination(
            query_result=query,
            progress_delta_m=reward_info["progress_delta_m"],
            vehicle_state=vehicle_state,
        )
        terminated = termination.terminated
        truncated = termination.truncated

        self.episode_progress_m += float(reward_info["progress_delta_m"])
        reward_info["episode_progress_m"] = float(self.episode_progress_m)

        # progress_m/progress_ratio are raw centreline coordinates retained for
        # backwards compatibility. episode_progress_m is the reset-relative
        # distance accumulated for training and future lap completion logic.
        info: dict[str, Any] = {
            "step": self.current_step,
            "control": control,
            "progress_m": float(query.lap_progress),
            "progress_ratio": float(query.lap_progress_ratio),
            "raw_progress_m": float(query.lap_progress),
            "raw_progress_ratio": float(query.lap_progress_ratio),
            "episode_start_progress_m": self.episode_start_progress_m,
            "episode_progress_m": self.episode_progress_m,
            "lateral_error_m": float(query.signed_lateral_error),
            "heading_error_rad": (
                float(query.heading_error_rad)
                if query.heading_error_rad is not None
                else 0.0
            ),
            "stuck_steps": self.stuck_steps,
            "progress_jump_detected": reward_info["progress_jump_detected"],
            "termination_reason": termination.reason,
            "termination": {
                "off_track": termination.off_track,
                "stuck": termination.stuck,
                "max_steps_reached": termination.max_steps_reached,
                "lap_completed": termination.lap_completed,
                "damage": termination.damage,
                "wall_bash": termination.wall_bash,
            },
            "vehicle_damage": float(vehicle_state.get("damage", 0.0)),
            "mock_simulation": self.use_mock,
            "vehicle_pos": vehicle_state.get("pos"),
            "vehicle_velocity": vehicle_state.get("velocity"),
            "vehicle_state": {
                "pos": vehicle_state.get("pos"),
                "velocity": vehicle_state.get("velocity"),
                "heading_rad": vehicle_state.get("heading_rad"),
            },
            "reward": reward_info,
        }

        # These updates happen after reward calculation so the next call can
        # measure progress relative to this step's centreline projection.
        self.previous_progress_m = float(query.lap_progress)
        self.previous_raw_progress_m = float(query.lap_progress)
        self.last_observation = observation
        self.last_info = info

        return observation, float(reward), terminated, truncated, info

    def _check_termination(
        self,
        query_result: TrackQueryResult,
        progress_delta_m: float,
        vehicle_state: dict[str, Any] | None = None,
    ) -> TerminationResult:
        """Update episode counters and report why an episode should end."""

        # Stuck detection is based on low progress over repeated steps. The
        # reward function reports the per-step stuck penalty; this counter turns
        # repeated low-progress behaviour into an episode termination.
        low_progress = progress_delta_m < self.min_progress_delta_m
        if low_progress:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        lateral_error_abs = abs(float(query_result.signed_lateral_error))
        off_track = lateral_error_abs > self.max_lateral_error_m
        stuck = self.stuck_steps >= self.stuck_steps_limit
        max_steps_reached = self.current_step >= self.max_episode_steps

        # Wall-bash: car pinned near the track boundary with no forward progress.
        near_wall = lateral_error_abs > self.max_lateral_error_m * 0.7
        if near_wall and low_progress:
            self._wall_bash_steps += 1
        else:
            self._wall_bash_steps = 0
        wall_bash = self._wall_bash_steps >= self.wall_bash_steps_limit

        # Damage termination: only meaningful in live mode; mock has no physics.
        damage_val = float((vehicle_state or {}).get("damage", 0.0))
        damage_exceeded = (not self.use_mock) and (damage_val > self.max_damage)

        # TODO: Add lap completion based on episode_progress_m reaching
        # track.total_lap_length. Do not use raw progress_ratio for this because
        # the recommended live bootstrap spawn starts near the centreline wrap
        # boundary rather than raw progress zero.
        lap_completed = False

        terminated = off_track or stuck or lap_completed or damage_exceeded or wall_bash
        truncated = max_steps_reached

        reason = "none"
        if off_track:
            reason = "off_track"
        elif damage_exceeded:
            reason = "damage"
        elif wall_bash:
            reason = "wall_bash"
        elif stuck:
            reason = "stuck"
        elif lap_completed:
            reason = "lap_completed"
        elif max_steps_reached:
            reason = "max_episode_steps"

        return TerminationResult(
            terminated=bool(terminated),
            truncated=bool(truncated),
            reason=reason,
            off_track=bool(off_track),
            stuck=bool(stuck),
            max_steps_reached=bool(max_steps_reached),
            lap_completed=bool(lap_completed),
            damage=bool(damage_exceeded),
            wall_bash=bool(wall_bash),
        )

    def _compute_reward(
        self,
        query_result: TrackQueryResult,
        vehicle_state: dict[str, Any],
        action_delta: np.ndarray | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Compute the first dense, inspectable racing reward.

        The reward uses raw centreline progress delta as the main signal, then
        adds small shaping terms for speed, heading alignment, lateral control,
        and obvious failure modes. Every component is returned in ``reward_info``
        so training logs can explain exactly where reward came from.
        """

        config = self.reward_config
        current_raw_progress_m = float(query_result.lap_progress)
        progress_delta_m = current_raw_progress_m - self.previous_raw_progress_m
        total_lap_length = float(self.track.total_lap_length)

        # Closed-loop tracks jump from total_lap_length back to 0 at the start
        # line. If that happens while moving forward, convert the large negative
        # delta into the small positive distance actually travelled.
        if progress_delta_m < -0.5 * total_lap_length:
            progress_delta_m += total_lap_length

        original_progress_delta_m = progress_delta_m
        progress_jump_detected = abs(progress_delta_m) > self.max_progress_delta_m
        if progress_jump_detected:
            progress_delta_m = 0.0

        progress_reward = progress_delta_m * config.progress_weight

        forward_speed_mps = self._forward_speed_from_state(vehicle_state)
        speed_reward = forward_speed_mps * config.speed_weight

        heading_error_rad = (
            float(query_result.heading_error_rad)
            if query_result.heading_error_rad is not None
            else 0.0
        )
        heading_penalty = abs(heading_error_rad) * config.heading_error_weight

        lateral_error_m = float(query_result.signed_lateral_error)
        lateral_penalty = abs(lateral_error_m) * config.lateral_error_weight

        off_track_penalty = 0.0
        if abs(lateral_error_m) > self.max_lateral_error_m:
            off_track_penalty = config.off_track_penalty

        reverse_progress_penalty = 0.0
        if progress_delta_m < 0.0:
            reverse_progress_penalty = config.reverse_progress_penalty

        stuck_penalty = 0.0
        if progress_delta_m < self.min_progress_delta_m:
            stuck_penalty = config.stuck_step_penalty

        progress_jump_penalty_value = (
            self.progress_jump_penalty if progress_jump_detected else 0.0
        )

        smoothness_penalty = (
            float(np.sum(np.abs(action_delta))) * config.action_smoothness_weight
            if action_delta is not None
            else 0.0
        )

        lap_progress = float(query_result.lap_progress)
        max_curvature_ahead = max(
            self.track.curvature_at(lap_progress + d) for d in (20.0, 40.0, 80.0)
        )
        target_speed = max(
            config.curvature_target_speed_min,
            config.curvature_target_speed_base - config.curvature_speed_scale * max_curvature_ahead,
        )
        curvature_overspeed_penalty = (
            max(0.0, forward_speed_mps - target_speed) * config.curvature_overspeed_weight
        )

        reward = (
            progress_reward
            + speed_reward
            - heading_penalty
            - lateral_penalty
            - off_track_penalty
            - reverse_progress_penalty
            - stuck_penalty
            - progress_jump_penalty_value
            - smoothness_penalty
            - curvature_overspeed_penalty
        )

        reward_info = {
            "raw_progress_m": float(current_raw_progress_m),
            "episode_start_progress_m": float(self.episode_start_progress_m),
            "episode_progress_m": float(self.episode_progress_m + progress_delta_m),
            "progress_delta_m": float(progress_delta_m),
            "progress_jump_detected": bool(progress_jump_detected),
            "original_progress_delta_m": float(original_progress_delta_m),
            "max_progress_delta_m": float(self.max_progress_delta_m),
            "progress_reward": float(progress_reward),
            "forward_speed_mps": float(forward_speed_mps),
            "speed_reward": float(speed_reward),
            "heading_error_rad": float(heading_error_rad),
            "heading_penalty": float(heading_penalty),
            "lateral_error_m": float(lateral_error_m),
            "lateral_penalty": float(lateral_penalty),
            "off_track_penalty": float(off_track_penalty),
            "reverse_progress_penalty": float(reverse_progress_penalty),
            "stuck_penalty": float(stuck_penalty),
            "progress_jump_penalty": float(progress_jump_penalty_value),
            "smoothness_penalty": float(smoothness_penalty),
            "max_curvature_ahead": float(max_curvature_ahead),
            "curvature_target_speed": float(target_speed),
            "curvature_overspeed_penalty": float(curvature_overspeed_penalty),
            "total_reward": float(reward),
        }
        return float(reward), reward_info

    def _reset_simulation(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset BeamNG or initialise the mock vehicle state."""

        if not self.use_mock:
            self._ensure_live_connected()

            # Keep the car neutral during reset. BeamNG keeps the last control
            # command until changed, so this prevents a stale throttle/brake
            # input from affecting the first frame after reset.
            self._last_control = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
            self._apply_action(self._last_control)

            # Scenario restart is the cleanest reset when BeamNGpy owns the
            # scenario. If it is unavailable, fall back to teleport/recover
            # below so live-mode experimentation can still proceed.
            self._try_restart_live_scenario()
            self._try_place_live_vehicle_at_start()

            # Step a tiny amount after teleport/restart so the state sensor has
            # a fresh frame to report. Deterministic stepping is configured in
            # _connect_beamng() when BeamNG accepts that setting.
            self._advance_simulation()
            return self._get_vehicle_state()

        options = options or {}
        self._mock_progress_m = float(options.get("mock_progress_m", 0.0))
        self._mock_lateral_offset_m = float(options.get("mock_lateral_offset_m", 0.0))
        self._mock_heading_error_rad = float(options.get("mock_heading_error_rad", 0.0))
        self._mock_forward_speed_mps = float(options.get("mock_speed_mps", 8.0))
        self._last_control = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}
        self._mock_vehicle_state = self._mock_state_from_progress()
        return self._mock_vehicle_state

    def _apply_action(self, control: dict[str, float]) -> None:
        """Send a control command to BeamNG, or store it for the mock state."""

        # Store the most recent command in both modes. In mock mode this drives
        # the toy dynamics; in live mode it gives info/debug code one consistent
        # place to inspect the last action sent to BeamNG.
        self._last_control = {
            "steering": float(control["steering"]),
            "throttle": float(control["throttle"]),
            "brake": float(control["brake"]),
        }

        if not self.use_mock:
            self._ensure_live_connected()
            try:
                self.vehicle.control(
                    steering=self._last_control["steering"],
                    throttle=self._last_control["throttle"],
                    brake=self._last_control["brake"],
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to send control command to BeamNG vehicle "
                    f"{self.vehicle_id!r}: {exc!r}"
                ) from exc

    def _advance_simulation(self) -> None:
        """Advance BeamNG one control step, or move the mock vehicle."""

        if not self.use_mock:
            self._ensure_live_connected()
            try:
                # BeamNGpy's deterministic step API assumes the simulator is
                # paused. _connect_beamng() asks BeamNG for deterministic mode
                # and pauses the sim; if a user changes that outside the env,
                # this call may become realtime-ish again.
                #
                # TODO: expose the deterministic steps-per-second setting once
                # training needs tighter control over policy frequency.
                self.beamng.step(self.steps_per_action, wait=True)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to advance BeamNG simulation by "
                    f"{self.steps_per_action} step(s): {exc!r}"
                ) from exc
            return

        # The mock dynamics are intentionally simple; they only exist to prove
        # the Gymnasium loop, observation builder, and reward accounting work.
        throttle = self._last_control["throttle"]
        brake = self._last_control["brake"]
        steering = self._last_control["steering"]

        # A tiny toy speed model: throttle accelerates, brake decelerates, and
        # light drag keeps speed bounded. This is not meant to be vehicle physics.
        acceleration_mps2 = 3.0 * throttle - 5.0 * brake - 0.05 * self._mock_forward_speed_mps
        self._mock_forward_speed_mps = float(
            np.clip(
                self._mock_forward_speed_mps + acceleration_mps2 * self._mock_dt_s,
                0.0,
                35.0,
            )
        )

        self._mock_progress_m = (
            self._mock_progress_m + self._mock_forward_speed_mps * self._mock_dt_s
        ) % self.track.total_lap_length

        # Steering nudges the mock car sideways and changes its heading error so
        # the observation and reward terms visibly respond during smoke tests.
        self._mock_lateral_offset_m = float(
            np.clip(self._mock_lateral_offset_m + steering * 0.05, -3.0, 3.0)
        )
        self._mock_heading_error_rad = float(np.clip(steering * 0.25, -0.35, 0.35))
        self._mock_vehicle_state = self._mock_state_from_progress()

    def _get_vehicle_state(self) -> dict[str, Any]:
        """Read the current BeamNG vehicle state, or return mock state."""

        if not self.use_mock:
            self._ensure_live_connected()
            try:
                # Every BeamNGpy Vehicle attaches a "state" sensor by default.
                # Polling it refreshes vehicle.state with pos/dir/vel/rotation.
                self.vehicle.sensors.poll("state", "damage")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to poll BeamNG vehicle state for {self.vehicle_id!r}: {exc!r}"
                ) from exc

            try:
                damage_val = float(self.vehicle.sensors["damage"].get("damage", 0.0))
            except Exception:
                damage_val = 0.0

            raw_state = dict(getattr(self.vehicle, "state", {}) or {})
            raw_sensors = getattr(getattr(self.vehicle, "sensors", None), "data", {})
            raw_payload: dict[str, Any] = {
                "state": raw_state,
                "sensors": raw_sensors,
                **raw_state,
            }

            # BeamNGpy's state sensor normally provides:
            #   pos: world position
            #   dir: vehicle forward direction vector
            #   vel: world velocity
            # The extra nested paths keep this robust if a future sensor stack
            # returns a wrapped dict such as {"sensors": {"state": {...}}}.
            position_xyz = self._extract_live_xyz(
                raw_payload,
                (
                    ("pos",),
                    ("position",),
                    ("state", "pos"),
                    ("state", "position"),
                    ("sensors", "state", "pos"),
                    ("sensors", "state", "position"),
                ),
                name="position",
            )
            if position_xyz is None:
                raise RuntimeError(
                    "BeamNG live vehicle state did not include a usable position. "
                    f"Available state keys: {sorted(raw_state.keys())!r}"
                )

            velocity_xyz = self._extract_live_xyz(
                raw_payload,
                (
                    ("vel",),
                    ("velocity",),
                    ("state", "vel"),
                    ("state", "velocity"),
                    ("sensors", "state", "vel"),
                    ("sensors", "state", "velocity"),
                    ("sensors", "electrics", "vel"),
                    ("sensors", "electrics", "velocity"),
                    ("sensors", "electrics", "state", "vel"),
                    ("sensors", "electrics", "state", "velocity"),
                ),
                name="velocity",
                default=(0.0, 0.0, 0.0),
            )

            heading_rad = self._extract_live_float(
                raw_payload,
                (
                    ("heading_rad",),
                    ("state", "heading_rad"),
                    ("sensors", "state", "heading_rad"),
                ),
            )
            if heading_rad is None:
                direction_xyz = self._extract_live_xyz(
                    raw_payload,
                    (
                        ("dir",),
                        ("direction",),
                        ("state", "dir"),
                        ("state", "direction"),
                        ("sensors", "state", "dir"),
                        ("sensors", "state", "direction"),
                    ),
                    name="direction",
                    default=None,
                )
                if direction_xyz is not None:
                    heading_rad = _heading_from_direction(direction_xyz)

            if heading_rad is None:
                # Last-resort fallback for early frames that have velocity but
                # no direction vector. ObservationBuilder can also infer this,
                # but live mode returns heading_rad explicitly to keep the env's
                # state shape stable.
                velocity_xy_norm = float(np.linalg.norm(np.asarray(velocity_xyz[:2])))
                if velocity_xy_norm > 1.0e-9:
                    heading_rad = _heading_from_direction(velocity_xyz)
                else:
                    # Neutral heading for a completely stationary/uninitialised
                    # frame. This should be rare once the state sensor is live.
                    heading_rad = 0.0

            return {
                "pos": position_xyz,
                "velocity": velocity_xyz,
                "heading_rad": float(heading_rad),
                "damage": damage_val,
            }

        if self._mock_vehicle_state is None:
            self._mock_vehicle_state = self._mock_state_from_progress()
        return self._mock_vehicle_state

    def close(self) -> None:
        """Close any BeamNG connection and clear local handles."""

        # This remains safe even if live setup failed halfway through. When
        # launch_beamng=False, BeamNGpy is created with quit_on_close=False so
        # close() disconnects Python without shutting down the user's process.
        beamng_close = getattr(self.beamng, "close", None)
        if callable(beamng_close):
            beamng_close()

        self.beamng = None
        self.scenario = None
        self.vehicle = None
        self._resolved_beamng_home = None

    def _connect_beamng(self) -> None:
        """Connect to BeamNG.tech and prepare the shared Hirochi/SBR scenario.

        BeamNGpy is imported here, not at module import time. That keeps mock
        smoke tests and simple imports working on machines without BeamNGpy.
        """

        try:
            from beamngpy import BeamNGpy
        except ImportError as exc:
            raise ImportError(
                "BeamNGpy must be installed to use BeamNGRacingEnv(use_mock=False). "
                "Install the beamngpy package and run BeamNG.tech, or keep "
                "use_mock=True for local smoke tests."
            ) from exc

        beamng = None
        try:
            if self.launch_beamng:
                self._resolved_beamng_home = resolve_beamng_home_from_path_or_env(
                    self.beamng_home
                )
                beamng = BeamNGpy(
                    self.beamng_host,
                    self.beamng_port,
                    home=str(self._resolved_beamng_home),
                    # nogpu=True disables GPU rendering (beamngpy 1.35 API).
                    # Controlled by nogfx=True on BeamNGRacingEnv — change it there.
                    nogpu=self.nogfx,
                )
                beamng.open(launch=True)
            else:
                # This mode is useful when BeamNG.tech is already open. Keep
                # quit_on_close disabled so env.close() only disconnects Python.
                beamng = BeamNGpy(
                    self.beamng_host,
                    self.beamng_port,
                    quit_on_close=False,
                )
                beamng.open(launch=False)

            # Deterministic stepping plus pause mirrors beamng_bootstrap.py and
            # makes beamng.step(...) the clock source for RL actions.
            try:
                desired_speed = self.speed_factor or 1
                # BeamNG.tech's deterministic API stores speedup as powers of
                # two: 0 -> 1x, 1 -> 2x, 2 -> 4x, etc. The launcher exposes
                # user-facing multipliers, so convert before calling BeamNGpy.
                beamng_speed_factor = int(math.log2(desired_speed))
                beamng.settings.set_nondeterministic()
                beamng.settings.remove_step_limit()
                beamng.settings.set_deterministic(60, speed_factor=beamng_speed_factor)
                print(
                    "BeamNG deterministic mode set: "
                    f"steps_per_second=60, requested_speed={desired_speed}x, "
                    f"beamng_speed_factor={beamng_speed_factor}"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to set BeamNG deterministic timing. "
                    "Restart BeamNG and verify the selected speed factor."
                ) from exc
            try:
                beamng.pause()
            except Exception:
                pass

            # The live env deliberately shares the bootstrap scenario setup so
            # manual debugging and training start from the same level, vehicle,
            # and part config. scenario_name is kept as a public constructor
            # parameter, but the shared setup is currently Hirochi.
            scenario_builders = {
                "sbr": build_hirochi_sbr_scenario,
                "etkc": build_hirochi_etkc_scenario,
            }
            print(f"Building Hirochi scenario for vehicle_model={self.vehicle_model}")
            _scenario_builder = scenario_builders[self.vehicle_model]
            scenario_instance_name = (
                f"beamng_rl_{self.vehicle_model}_{self.vehicle_id}_{os.getpid()}"
            )
            print(f"Generated scenario instance: {scenario_instance_name}")
            scenario, vehicle = _scenario_builder(
                beamng,
                vehicle_id=self.vehicle_id,
                scenario_instance_name=scenario_instance_name,
            )

            from beamngpy.sensors import Damage
            vehicle.attach_sensor("damage", Damage())

            beamng.scenario.load(scenario)
            self._try_hide_live_hud(beamng)
            beamng.scenario.start()

            # Re-pause after scenario start so beamng.step(...) controls the
            # simulation clock as tightly as this minimal integration allows.
            try:
                beamng.pause()
            except Exception:
                pass
            self._try_hide_live_hud(beamng)
            self._try_apply_live_graphics(beamng)

            self.beamng = beamng
            self.scenario = scenario
            self.vehicle = vehicle

            # Prime the state sensor. This catches a bad vehicle connection
            # immediately instead of failing later inside the first RL step.
            self.vehicle.sensors.poll("state")

        except Exception as exc:
            if beamng is not None:
                close = getattr(beamng, "close", None)
                if callable(close):
                    close()
            self.beamng = None
            self.scenario = None
            self.vehicle = None
            self._resolved_beamng_home = None
            connection_mode = (
                "launch BeamNG.tech"
                if self.launch_beamng
                else "connect to an already-running BeamNG.tech process"
            )
            raise RuntimeError(
                "Failed to initialise live BeamNG mode at "
                f"{self.beamng_host}:{self.beamng_port} while trying to "
                f"{connection_mode}. Confirm the shared Hirochi/SBR scenario "
                "is available, or use use_mock=True. "
                f"Original error: {exc!r}"
            ) from exc

    def _try_hide_live_hud(self, beamng: Any) -> None:
        """Best-effort HUD hiding; visual setup should not fail an RL reset."""

        hide_hud = getattr(getattr(beamng, "ui", None), "hide_hud", None)
        if not callable(hide_hud):
            return

        try:
            hide_hud()
        except Exception:
            pass

    def _try_apply_live_graphics(self, beamng: Any) -> None:
        """Apply optional graphics settings without making them training-critical."""

        if self.disable_shadows:
            try:
                apply_shadow_disabling(beamng)
            except Exception:
                # TODO: collect graphics-setting failures in info/logging once
                # the live training harness has a central diagnostics channel.
                pass

        if self.apply_low_graphics:
            try:
                apply_low_graphics_preset(beamng)
            except Exception:
                # Low graphics is a convenience for local performance, not a
                # condition for the environment to be usable.
                pass

    def _ensure_live_connected(self) -> None:
        """Reconnect if needed, then raise a clear error if BeamNG is unavailable."""

        if self.beamng is None or self.vehicle is None:
            if not self.use_mock:
                self._connect_beamng()

        if self.beamng is None or self.vehicle is None:
            raise RuntimeError(
                "BeamNG live mode is not connected. Construct the environment "
                "with use_mock=False and ensure _connect_beamng() completes."
            )

    def _try_restart_live_scenario(self) -> None:
        """Best-effort live reset via BeamNGpy scenario restart."""

        self._ensure_live_connected()
        restart = getattr(getattr(self.beamng, "scenario", None), "restart", None)
        if not callable(restart):
            return

        try:
            restart()
            try:
                self.beamng.pause()
            except Exception:
                pass
        except Exception:
            # Scenario restart can fail if the user manually replaced/stopped
            # the scenario. Teleport/recover below is still useful, so reset()
            # does not fail solely because restart was unavailable.
            pass

    def _try_place_live_vehicle_at_start(self) -> None:
        """Best-effort placement of the live vehicle at the configured spawn."""

        self._ensure_live_connected()
        spawn_pos, spawn_rot_quat = self._select_live_spawn_pose()

        # Prefer Vehicle.teleport because it resets velocity and keeps the call
        # tied to the connected vehicle object. BeamNGpy exposes an equivalent
        # beamng.vehicles.teleport(...) path, so try that as a fallback.
        try:
            self.vehicle.teleport(
                spawn_pos,
                rot_quat=spawn_rot_quat,
                reset=True,
            )
            return
        except Exception:
            pass

        try:
            self.beamng.vehicles.teleport(
                self.vehicle,
                spawn_pos,
                rot_quat=spawn_rot_quat,
                reset=True,
            )
            return
        except Exception:
            pass

        recover = getattr(self.vehicle, "recover", None)
        if callable(recover):
            try:
                recover()
            except Exception:
                pass

    def _mock_state_from_progress(self) -> dict[str, Any]:
        """Create a BeamNG-like state dictionary from mock lap progress."""

        centre_point = self.track.point_at_progress(self._mock_progress_m)
        base_query = self.track.query(centre_point)
        tangent_xy = np.asarray(base_query.track_tangent_xy, dtype=float)

        # Positive lateral error is to the left of the centreline. This normal
        # vector matches the sign convention used by TrackCentreline.query().
        left_normal_xy = np.asarray([-tangent_xy[1], tangent_xy[0]], dtype=float)
        position_xyz = centre_point.copy()
        position_xyz[:2] += left_normal_xy * self._mock_lateral_offset_m

        heading_rad = float(base_query.track_heading_rad + self._mock_heading_error_rad)
        velocity_xyz = np.asarray(
            [
                math.cos(heading_rad) * self._mock_forward_speed_mps,
                math.sin(heading_rad) * self._mock_forward_speed_mps,
                0.0,
            ],
            dtype=float,
        )

        return {
            "pos": position_xyz.tolist(),
            "velocity": velocity_xyz.tolist(),
            "heading_rad": heading_rad,
        }

    def _select_live_spawn_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return the live BeamNG spawn pose selected by live_spawn_mode."""

        if self.live_spawn_mode == "bootstrap":
            # Bootstrap mode uses the hand-verified Hirochi/SBR pose from
            # beamng_bootstrap.py because it is known to open the scenario
            # correctly and places the car on the BeamNG start-marker pose.
            # This is the recommended mode for live training.
            return (
                tuple(float(value) for value in SPAWN_POS),
                tuple(float(value) for value in SPAWN_ROT),
            )

        if self.live_spawn_mode == "centreline":
            # Centreline mode is experimental. It aligns with the track JSON,
            # but may not match the actual BeamNG road surface, vehicle heading,
            # terrain height, or painted start markers.
            centre_point = np.asarray(
                self.track.point_at_progress(self.live_spawn_progress_m),
                dtype=float,
            ).reshape(-1)
            if centre_point.shape[0] < 3:
                centre_point = np.asarray(
                    [centre_point[0], centre_point[1], 0.0],
                    dtype=float,
                )

            base_query = self.track.query(centre_point[:3])
            tangent_xy = np.asarray(base_query.track_tangent_xy, dtype=float)
            left_normal_xy = np.asarray([-tangent_xy[1], tangent_xy[0]], dtype=float)

            position_xyz = centre_point[:3].copy()
            position_xyz[:2] += left_normal_xy * self.live_spawn_lateral_offset_m
            position_xyz[2] += self.live_spawn_z_offset_m

            heading_rad = float(base_query.track_heading_rad)
            half_yaw = 0.5 * heading_rad
            rot_quat = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
            return (
                tuple(float(value) for value in position_xyz[:3]),
                tuple(float(value) for value in rot_quat),
            )

        # Constructor validation should make this unreachable, but keeping this
        # branch gives a clear error if the attribute is changed at runtime.
        raise ValueError(
            "live_spawn_mode must be one of 'bootstrap' or 'centreline', "
            f"got {self.live_spawn_mode!r}"
        )

    def _default_spawn_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Backward-compatible alias for the selected live spawn pose."""

        return self._select_live_spawn_pose()

    @staticmethod
    def _extract_live_xyz(
        payload: Mapping[str, Any],
        paths: tuple[tuple[str, ...], ...],
        *,
        name: str,
        default: tuple[float, float, float] | None = None,
    ) -> list[float] | None:
        """Extract a live BeamNG vector from several possible nested paths."""

        for path in paths:
            value = _nested_get(payload, path)
            if value is None:
                continue

            try:
                return _coerce_xyz_list(value, name=name)
            except ValueError:
                continue

        if default is None:
            return None
        return [float(default[0]), float(default[1]), float(default[2])]

    @staticmethod
    def _extract_live_float(
        payload: Mapping[str, Any],
        paths: tuple[tuple[str, ...], ...],
    ) -> float | None:
        """Extract a finite float from several possible nested paths."""

        for path in paths:
            value = _nested_get(payload, path)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
        return None

    def _heading_from_state(self, vehicle_state: dict[str, Any]) -> float | None:
        """Extract a heading consistently with ObservationBuilder."""

        try:
            velocity_xyz = self.observation_builder._extract_velocity_xyz(vehicle_state)
            return self.observation_builder._extract_heading_rad(vehicle_state, velocity_xyz)
        except (KeyError, TypeError, ValueError):
            return None

    def _forward_speed_from_state(self, vehicle_state: dict[str, Any]) -> float:
        """Extract forward speed for reward calculation, defaulting safely."""

        try:
            velocity_xyz = self.observation_builder._extract_velocity_xyz(vehicle_state)
            heading_rad = self.observation_builder._extract_heading_rad(
                vehicle_state,
                velocity_xyz,
            )
        except (KeyError, TypeError, ValueError):
            return 0.0

        if heading_rad is None:
            return 0.0

        forward_xy = np.asarray(
            [math.cos(float(heading_rad)), math.sin(float(heading_rad))],
            dtype=float,
        )
        return float(np.dot(velocity_xyz[:2], forward_xy))


def _smoke_test() -> None:
    """Run a small environment loop without launching BeamNG."""

    repo_root = Path(__file__).resolve().parents[3]
    centreline_path = (
        repo_root / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )

    # Mock mode is the default and intentionally remains the smoke-test path.
    # To try live mode manually with the shared bootstrap setup, instantiate:
    #   env = BeamNGRacingEnv(centreline_path, use_mock=False, launch_beamng=True)
    env = BeamNGRacingEnv(centreline_path)
    print(f"Loaded track: {centreline_path}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}")

    for step_index in range(1, 6):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        reward_info = info["reward"]

        print(
            "Step "
            f"{step_index}: reward={reward:.4f}, "
            f"terminated={terminated}, truncated={truncated}, "
            f"progress_delta_m={reward_info['progress_delta_m']:.4f}, "
            f"progress_reward={reward_info['progress_reward']:.4f}, "
            f"speed_reward={reward_info['speed_reward']:.4f}, "
            f"heading_penalty={reward_info['heading_penalty']:.4f}, "
            f"lateral_penalty={reward_info['lateral_penalty']:.4f}, "
            f"total_reward={reward_info['total_reward']:.4f}"
        )

        if terminated or truncated:
            break

    env.close()


def _heading_from_direction(direction: Any) -> float:
    """Convert a BeamNG-style forward vector into an XY-plane heading angle."""

    vector = np.asarray(direction, dtype=float).reshape(-1)
    if vector.shape[0] < 2:
        raise ValueError(f"direction must contain at least x and y, got {direction!r}")
    return float(math.atan2(vector[1], vector[0]))


def _nested_get(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
    """Read a nested mapping path, returning None when any key is absent."""

    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _coerce_xyz_list(value: Any, *, name: str) -> list[float]:
    """Normalise BeamNG vector variants into a finite [x, y, z] list."""

    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise ValueError(f"{name} mapping must contain at least x and y")
        raw_values = [value["x"], value["y"], value.get("z", 0.0)]
    else:
        try:
            raw_values = list(np.asarray(value, dtype=float).reshape(-1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector") from exc
        if len(raw_values) == 2:
            raw_values.append(0.0)

    vector = np.asarray(raw_values[:3], dtype=float).reshape(-1)
    if vector.shape[0] != 3:
        raise ValueError(f"{name} must contain 2 or 3 values, got {vector.shape[0]}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values, got {value!r}")
    return [float(vector[0]), float(vector[1]), float(vector[2])]


if __name__ == "__main__":
    _smoke_test()
