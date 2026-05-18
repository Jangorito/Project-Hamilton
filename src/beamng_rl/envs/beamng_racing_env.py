"""Minimal Gymnasium racing environment for BeamNG RL experiments.

This module creates the first real Gymnasium-compatible wrapper around the
track-query and observation-building code that already exists in the project.
The BeamNG-specific methods are intentionally small placeholders for now: they
make the integration points obvious while still allowing a smoke test to run
without launching BeamNG.tech.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

try:
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
    simulator-facing methods are deliberately separated so they can later be
    replaced with BeamNGpy calls without rewriting reward or observation logic.
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
        self.vehicle = None

        # Mock-state variables used only while no live BeamNG vehicle is
        # connected. They let step() move around the track during smoke tests.
        self._mock_dt_s = 0.1
        self._mock_progress_m = 0.0
        self._mock_lateral_offset_m = 0.0
        self._mock_heading_error_rad = 0.0
        self._mock_forward_speed_mps = 8.0
        self._mock_vehicle_state: dict[str, Any] | None = None
        self._last_control = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

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

        # This currently returns a mock vehicle state when no BeamNG vehicle is
        # connected. The method boundary is the future BeamNGpy reset hook.
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
            "mock_simulation": self.vehicle is None,
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
            "mock_simulation": self.vehicle is None,
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
        """Reset BeamNG or, for now, initialise the mock vehicle state."""

        # Future BeamNGpy integration should reset/reload the scenario here,
        # place the vehicle at the start pose, and return the first live state.
        # Until self.vehicle is connected, this mock branch keeps development
        # and smoke testing independent of a running simulator.
        if self.vehicle is not None:
            # Placeholder for future live BeamNG reset logic.
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

        # Future BeamNGpy integration point:
        #   self.vehicle.control(
        #       steering=control["steering"],
        #       throttle=control["throttle"],
        #       brake=control["brake"],
        #   )
        # For the current mock path, store the command so _advance_simulation()
        # can make the fake vehicle respond in a simple, deterministic way.
        self._last_control = {
            "steering": float(control["steering"]),
            "throttle": float(control["throttle"]),
            "brake": float(control["brake"]),
        }

    def _advance_simulation(self) -> None:
        """Advance BeamNG one control step, or move the mock vehicle."""

        # Future BeamNGpy integration point:
        #   self.beamng.step(1)
        # The mock dynamics are intentionally simple; they only exist to prove
        # the Gymnasium loop, observation builder, and reward accounting work.
        if self.vehicle is not None:
            return

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

        # Future BeamNGpy integration should poll sensors/electrics here and
        # normalise the result into the dictionary shape expected by
        # ObservationBuilder: at minimum position plus optional velocity/heading.
        if self.vehicle is not None:
            # Placeholder until the live vehicle-state schema is connected.
            return self._mock_vehicle_state or self._mock_state_from_progress()

        if self._mock_vehicle_state is None:
            self._mock_vehicle_state = self._mock_state_from_progress()
        return self._mock_vehicle_state

    def close(self) -> None:
        """Close any future BeamNG connection and clear local handles."""

        # The placeholders are defensive: once BeamNGpy is wired in, close()
        # should remain safe even if setup failed halfway through.
        beamng_close = getattr(self.beamng, "close", None)
        if callable(beamng_close):
            beamng_close()

        self.beamng = None
        self.vehicle = None

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


if __name__ == "__main__":
    _smoke_test()
