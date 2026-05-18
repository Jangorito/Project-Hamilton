"""Minimal Gymnasium racing environment for BeamNG RL experiments.

This module creates the Gymnasium-compatible wrapper around the track-query and
observation-building code that already exists in the project. Mock mode remains
the safe default for unit and smoke testing; live mode reuses the manual
bootstrap's Hirochi/SBR setup so simulator experiments start from the same
scenario, vehicle, and spawn pose.
"""

from __future__ import annotations

import math
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
        build_hirochi_sbr_scenario,
        resolve_beamng_home_from_path_or_env,
    )
    from beamng_rl.envs.observation_builder import ObservationBuilder
    from beamng_rl.track.query_utils import TrackCentreline, TrackQueryResult


@dataclass
class RewardConfig:
    """Tunable constants for the first inspectable racing reward."""

    progress_weight: float = 1.0
    speed_weight: float = 0.02
    heading_error_weight: float = 0.2
    lateral_error_weight: float = 0.1
    off_track_penalty: float = 10.0
    reverse_progress_penalty: float = 2.0
    stuck_step_penalty: float = 0.05


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
    ) -> None:
        super().__init__()

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

        # Runtime counters are reset in reset(), but initial values keep the
        # object inspectable immediately after construction.
        self.current_step = 0
        self.previous_progress_m = 0.0
        self.stuck_steps = 0
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

        # Live BeamNG mode uses the same hand-verified spawn pose as
        # beamng_bootstrap.py. The centreline-derived pose remains only as a
        # fallback helper for future non-Hirochi experiments.
        self._live_spawn_pos = tuple(float(value) for value in SPAWN_POS)
        self._live_spawn_rot_quat = tuple(float(value) for value in SPAWN_ROT)

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
        self.stuck_steps = 0

        # This returns a backend-normalised vehicle-state dictionary. Mock mode
        # uses the safe toy dynamics; live mode polls the BeamNGpy vehicle.
        vehicle_state = self._reset_simulation(options=options)
        observation = self.observation_builder.build(vehicle_state)

        # Store the starting progress so the first reward uses progress made
        # after reset rather than distance from zero on the lap.
        heading_rad = self._heading_from_state(vehicle_state)
        query = self.track.query(vehicle_state["pos"], vehicle_heading_rad=heading_rad)
        self.previous_progress_m = float(query.lap_progress)

        info: dict[str, Any] = {
            "progress_m": float(query.lap_progress),
            "progress_ratio": float(query.lap_progress_ratio),
            "lateral_error_m": float(query.signed_lateral_error),
            "heading_error_rad": (
                float(query.heading_error_rad)
                if query.heading_error_rad is not None
                else 0.0
            ),
            "mock_simulation": self.use_mock,
        }

        self.last_observation = observation
        self.last_info = info
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action, advance the simulation, and return one RL step."""

        control = self._split_action(action)
        self._apply_action(control)
        self._advance_simulation()

        vehicle_state = self._get_vehicle_state()
        observation = self.observation_builder.build(vehicle_state)

        heading_rad = self._heading_from_state(vehicle_state)
        query = self.track.query(vehicle_state["pos"], vehicle_heading_rad=heading_rad)

        reward, reward_info = self._compute_reward(query, vehicle_state)

        self.current_step += 1

        termination = self._check_termination(
            query_result=query,
            progress_delta_m=reward_info["progress_delta_m"],
        )
        terminated = termination.terminated
        truncated = termination.truncated

        info: dict[str, Any] = {
            "step": self.current_step,
            "control": control,
            "progress_m": float(query.lap_progress),
            "progress_ratio": float(query.lap_progress_ratio),
            "lateral_error_m": float(query.signed_lateral_error),
            "heading_error_rad": (
                float(query.heading_error_rad)
                if query.heading_error_rad is not None
                else 0.0
            ),
            "stuck_steps": self.stuck_steps,
            "termination_reason": termination.reason,
            "termination": {
                "off_track": termination.off_track,
                "stuck": termination.stuck,
                "max_steps_reached": termination.max_steps_reached,
                "lap_completed": termination.lap_completed,
            },
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
        self.last_observation = observation
        self.last_info = info

        return observation, float(reward), terminated, truncated, info

    def _check_termination(
        self,
        query_result: TrackQueryResult,
        progress_delta_m: float,
    ) -> TerminationResult:
        """Update episode counters and report why an episode should end."""

        # Stuck detection is based on low progress over repeated steps. The
        # reward function reports the per-step stuck penalty; this counter turns
        # repeated low-progress behaviour into an episode termination.
        if progress_delta_m < self.min_progress_delta_m:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        off_track = abs(float(query_result.signed_lateral_error)) > self.max_lateral_error_m
        stuck = self.stuck_steps >= self.stuck_steps_limit
        max_steps_reached = self.current_step >= self.max_episode_steps

        # TODO: Add lap completion once reset/progress wrapping is robust enough
        # to distinguish a completed lap from start-line initialisation.
        lap_completed = False

        terminated = off_track or stuck or lap_completed
        truncated = max_steps_reached

        reason = "none"
        if off_track:
            reason = "off_track"
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
        )

    def _compute_reward(
        self,
        query_result: TrackQueryResult,
        vehicle_state: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Compute the first dense, inspectable racing reward.

        The reward uses centreline progress as the main signal, then adds small
        shaping terms for speed, heading alignment, lateral control, and obvious
        failure modes. Every component is returned in ``reward_info`` so training
        logs can explain exactly where reward came from.
        """

        config = self.reward_config
        current_progress_m = float(query_result.lap_progress)
        progress_delta_m = current_progress_m - self.previous_progress_m
        total_lap_length = float(self.track.total_lap_length)

        # Closed-loop tracks jump from total_lap_length back to 0 at the start
        # line. If that happens while moving forward, convert the large negative
        # delta into the small positive distance actually travelled.
        if progress_delta_m < -0.5 * total_lap_length:
            progress_delta_m += total_lap_length

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

        reward = (
            progress_reward
            + speed_reward
            - heading_penalty
            - lateral_penalty
            - off_track_penalty
            - reverse_progress_penalty
            - stuck_penalty
        )

        reward_info = {
            "progress_delta_m": float(progress_delta_m),
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
                self.vehicle.sensors.poll("state")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to poll BeamNG vehicle state for {self.vehicle_id!r}: {exc!r}"
                ) from exc

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
                beamng.settings.set_deterministic(60)
                beamng.pause()
            except Exception:
                pass

            # The live env deliberately shares the bootstrap scenario setup so
            # manual debugging and training start from the same level, vehicle,
            # part config, and spawn pose. scenario_name is kept as a public
            # constructor parameter, but the shared setup is currently Hirochi.
            scenario, vehicle = build_hirochi_sbr_scenario(
                beamng,
                vehicle_id=self.vehicle_id,
                scenario_instance_name=f"beamng_rl_{self.vehicle_id}",
            )
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
        """Best-effort placement of the live vehicle at the bootstrap spawn."""

        self._ensure_live_connected()

        # Prefer Vehicle.teleport because it resets velocity and keeps the call
        # tied to the connected vehicle object. BeamNGpy exposes an equivalent
        # beamng.vehicles.teleport(...) path, so try that as a fallback.
        try:
            self.vehicle.teleport(
                self._live_spawn_pos,
                rot_quat=self._live_spawn_rot_quat,
                reset=True,
            )
            return
        except Exception:
            pass

        try:
            self.beamng.vehicles.teleport(
                self.vehicle,
                self._live_spawn_pos,
                rot_quat=self._live_spawn_rot_quat,
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

    def _default_spawn_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Fallback spawn pose for future tracks without a hand-verified pose."""

        start_point = np.asarray(
            self.track.point_at_progress(0.0),
            dtype=float,
        ).reshape(-1)
        if start_point.shape[0] < 3:
            start_point = np.asarray([start_point[0], start_point[1], 0.0], dtype=float)

        start_query = self.track.query(start_point[:3])
        heading_rad = float(start_query.track_heading_rad)

        # BeamNGpy wants quaternions as (x, y, z, w). This is a flat yaw-only
        # quaternion, good enough for a generated scenario on a mostly level
        # start segment. Track-authored scenarios should eventually provide an
        # exact spawn quaternion.
        half_yaw = 0.5 * heading_rad
        rot_quat = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
        pos = tuple(float(value) for value in start_point[:3])
        return pos, tuple(float(value) for value in rot_quat)

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
