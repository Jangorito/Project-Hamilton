"""
Build fixed-size RL observations from BeamNG vehicle state dictionaries.
The track geometry stays in ``beamng_rl.track.query_utils`` and the 
future Gym environment can call this builder without needing to know the
details of centreline projection, lookahead points, or BeamNG's 
slightly variable state field names.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from beamng_rl.track.query_utils import TrackCentreline
except ModuleNotFoundError as exc:
    # Running this file directly with
    #   python src/beamng_rl/envs/observation_builder.py
    # puts ``src/beamng_rl/envs`` on sys.path, not the repository ``src``
    # directory. This fallback is only for the temporary smoke test and keeps
    # normal package imports unchanged.
    if exc.name != "beamng_rl":
        raise

    import sys

    src_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(src_root))
    from beamng_rl.track.query_utils import TrackCentreline


# Base observation vector has 12 features.  obs_v2 appends 2 prev_action
# features (prev_steer, prev_throttle_brake) for a 14-feature vector.
_OBSERVATION_SIZE_V1 = 12
_OBSERVATION_SIZE_V2 = 14
_LOOKAHEAD_COUNT = 3

# Registry of named observation configs — mirrors the REWARD_CONFIGS pattern so
# every run records exactly which obs space it used in run_config.json.
OBS_CONFIGS: dict[str, int] = {
    "v1": _OBSERVATION_SIZE_V1,   # 12-feature baseline (all existing runs)
    "v2": _OBSERVATION_SIZE_V2,   # v1 + prev_steer + prev_throttle_brake
}

# A tiny threshold is used checking if a vector is basically zero length
_MIN_VECTOR_NORM = 1.0e-9

# Handling any discrepancies in naming conventions
_POSITION_KEYS = ("pos", "position")
_VELOCITY_KEYS = ("vel", "velocity")
_HEADING_RAD_KEYS = ("heading_rad",)
_DIRECTION_KEYS = ("dir", "direction")


@dataclass(frozen=True) # prevents later object mutation
class ObservationConfig:
    """
    Normalisation settings for the observation vector.

    The defaults are intentionally conservative rather than track-perfect:
    they keep typical Hirochi race speeds, off-centre distances, and upcoming
    curvature values inside useful neural-network ranges while still clipping
    outliers.
    """

    # Forward speed is signed to account for reversing. Keeping it separate
    # from lateral speed lets the policy distinguish productive motion from
    # sliding.
    max_forward_speed_mps: float = 60.0

    # Lateral speed is also signed in the vehicle frame. Large values indicate
    # slip or loss of control.
    max_lateral_speed_mps: float = 20.0

    # Vertical speed because the vehicle can crest, jump, or settle
    # after suspension movement. It is signed and normally smaller
    # than horizontal speed.
    max_vertical_speed_mps: float = 10.0

    # Lateral error is signed by TrackCentreline (+left, -right), so it is
    # scaled symmetrically into [-1, 1].
    max_lateral_error_m: float = 10.0

    # :class:`beamng_rl.track.query_utils.curvature_at` currently returns unsigned 
    # curvature magnitude. We use signed clipping [-1, 1] so the observation can preserve 
    # sign automatically if query_utils later grows signed curvature.
    max_curvature: float = 0.05

    # Three lookahead distances give the policy near/mid/far track context:
    # 20 m for immediate steering, 40 m for corner entry, and 80 m for planning.
    lookahead_distances_m: tuple[float, float, float] = (20.0, 40.0, 80.0)

    def __post_init__(self) -> None:
        # value validations for config parameters
        _require_positive_finite(self.max_forward_speed_mps, "max_forward_speed_mps")
        _require_positive_finite(self.max_lateral_speed_mps, "max_lateral_speed_mps")
        _require_positive_finite(self.max_vertical_speed_mps, "max_vertical_speed_mps")
        _require_positive_finite(self.max_lateral_error_m, "max_lateral_error_m")
        _require_positive_finite(self.max_curvature, "max_curvature")
        if len(self.lookahead_distances_m) != _LOOKAHEAD_COUNT:
            raise ValueError(
                "lookahead_distances_m must contain exactly "
                f"{_LOOKAHEAD_COUNT} distances for the fixed-size observation"
            )
        for index, distance_m in enumerate(self.lookahead_distances_m):
            _require_positive_finite(distance_m, f"lookahead_distances_m[{index}]")


class ObservationBuilder:
    """Convert BeamNG vehicle state dictionaries into observation vectors."""

    def __init__(
        self,
        track: TrackCentreline,
        config: ObservationConfig | None = None,
        obs_config_key: str = "v1",
    ) -> None:
        if obs_config_key not in OBS_CONFIGS:
            raise ValueError(
                f"obs_config_key must be one of {sorted(OBS_CONFIGS)}, "
                f"got {obs_config_key!r}"
            )
        # The builder depends on the public TrackCentreline API instead of raw
        # centreline arrays, separating track math and observation construction
        self.track = track
        self.config = config if config is not None else ObservationConfig()
        self.obs_config_key = obs_config_key

    def build(
        self,
        vehicle_state: Mapping[str, Any],
        prev_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the ``np.float32`` observation vector.

        Observation order (obs_v1, 12 features):
            [forward_speed_norm, lateral_speed_norm, vertical_speed_norm,
             heading_error_norm, lateral_error_norm,
             lookahead_heading_error_20m_norm, ..._40m_norm, ..._80m_norm,
             curvature_20m_norm, curvature_40m_norm, curvature_80m_norm,
             progress_ratio]

        obs_v2 appends 2 additional features (14 total):
            [...obs_v1..., prev_steer_norm, prev_throttle_brake_norm]

        Args:
            vehicle_state: Raw BeamNG state dict.
            prev_action: The action taken in the previous step as a length-2
                array [steer, throttle_brake] in [-1, 1]. Pass None (or omit)
                to use zeros — correct for the first step after a reset.
        """

        if not isinstance(vehicle_state, Mapping):
            raise TypeError(
                "vehicle_state must be a mapping/dictionary-like object; "
                f"got {type(vehicle_state).__name__}"
            )

        position_xyz = self._extract_position_xyz(vehicle_state)
        velocity_xyz = self._extract_velocity_xyz(vehicle_state)
        heading_rad = self._extract_heading_rad(vehicle_state, velocity_xyz)

        #querying the track, bridge between raw BeamNG state and RL features
        query = self.track.query(position_xyz, vehicle_heading_rad=heading_rad)

        # calculating forward, lateral and vertical speeds using heading
        # If heading is unavailable, it defaults to 0 radians, which 
        # means neutral speeds
        speed_heading_rad = heading_rad if heading_rad is not None else 0.0

        # converting world velocity into vehicle-relative velocity
        forward_speed, lateral_speed, vertical_speed = self._vehicle_frame_speeds(
            velocity_xyz=velocity_xyz,
            heading_rad=speed_heading_rad,)
        
        # Normalising speeds into roughly [-1, 1]
        forward_speed_norm = _clip_signed(forward_speed/self.config.max_forward_speed_mps)
        lateral_speed_norm = _clip_signed(lateral_speed/self.config.max_lateral_speed_mps)
        vertical_speed_norm = _clip_signed(vertical_speed/self.config.max_vertical_speed_mps)

        # use query_utils.py heading or 0
        heading_error_rad = (
            float(query.heading_error_rad)
            if query.heading_error_rad is not None
            else 0.0
        )
        # normalise heading error to [-1, 1] by dividing by pi and clipping
        heading_error_norm = _clip_signed(heading_error_rad / math.pi)
        # normalise lateral error to [-1, 1]
        lateral_error_norm = _clip_signed(
            float(query.signed_lateral_error) / self.config.max_lateral_error_m
        )

        # The current TrackCentreline.query API returns the current centreline
        # projection. Lookahead points and curvature are exposed as separate
        # public helpers, so the builder evaluates them at future lap progress
        # values without changing query_utils.py.

        # lists to store lookahead heading errors and curvature values for each 
        # lookahead distance
        lookahead_heading_error_norms: list[float] = []
        curvature_norms: list[float] = []

        for lookahead_m in self.config.lookahead_distances_m:
            """
            For each distance, calculate:
                - future progress along the lap
                - future centreline point
                - heading error to that future point
                - curvature at that future progress
            """
            lookahead_progress = float(query.lap_progress) + float(lookahead_m)
            lookahead_point_xyz = self.track.lookahead_point(
                                    query.lap_progress,
                                    lookahead_m,
                                    )
            # current heading error asks:
            #  "am I aligned with the track here?"
            # lookahead heading error asks:
            # -----------------------
            # "am I pointing toward where the track is going soon?"
            lookahead_heading_error_norms.append(
                self._lookahead_heading_error_norm(
                    position_xyz=position_xyz,
                    heading_rad=heading_rad,
                    lookahead_point_xyz=lookahead_point_xyz,
                )
            )

            # "how curved is the track at the future progress point?"
            curvature = self.track.curvature_at(lookahead_progress)
            curvature_norms.append(
                _clip_signed(float(curvature) / self.config.max_curvature)
            )

        # lap progress ratio
        progress_ratio = _clip_unit(float(query.lap_progress_ratio))

        base = np.asarray(
            [
                forward_speed_norm,
                lateral_speed_norm,
                vertical_speed_norm,
                heading_error_norm,
                lateral_error_norm,
                *lookahead_heading_error_norms,
                *curvature_norms,
                progress_ratio,
            ],
            dtype=np.float32,
        )

        if self.obs_config_key == "v2":
            if prev_action is not None:
                prev_steer = float(np.clip(prev_action[0], -1.0, 1.0))
                prev_throttle = float(np.clip(prev_action[1], -1.0, 1.0))
            else:
                prev_steer, prev_throttle = 0.0, 0.0
            return np.append(base, [prev_steer, prev_throttle]).astype(np.float32)

        return base

    def __call__(self, vehicle_state: Mapping[str, Any]) -> np.ndarray:
        """Allows the builder instance to be called directly to build an observation."""

        return self.build(vehicle_state)

    def observation_shape(self) -> tuple[int]:
        """Return the observation shape for the active obs config."""

        return (OBS_CONFIGS[self.obs_config_key],)

    def low(self) -> np.ndarray:
        """Return per-feature lower bounds for a Gym/Gymnasium Box space."""

        # All features are signed and clipped to [-1, 1] except progress_ratio
        # (lap fraction in [0, 1]) and prev_action features (also [-1, 1]).
        base = np.asarray(
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
            dtype=np.float32,
        )
        if self.obs_config_key == "v2":
            return np.append(base, [-1.0, -1.0]).astype(np.float32)
        return base

    def high(self) -> np.ndarray:
        """Return per-feature upper bounds for a Gym/Gymnasium Box space."""

        return np.ones(self.observation_shape(), dtype=np.float32)

    def feature_names(self) -> list[str]:
        """Return feature names in the exact order used by ``build``."""

        distance_labels = [
            _distance_label(distance_m)
            for distance_m in self.config.lookahead_distances_m
        ]
        names = [
            "forward_speed_norm",
            "lateral_speed_norm",
            "vertical_speed_norm",
            "heading_error_norm",
            "lateral_error_norm",
            *(f"lookahead_heading_error_{label}_norm" for label in distance_labels),
            *(f"curvature_{label}_norm" for label in distance_labels),
            "progress_ratio",
        ]
        if self.obs_config_key == "v2":
            names += ["prev_steer_norm", "prev_throttle_brake_norm"]
        return names

    def _extract_position_xyz(self, vehicle_state: Mapping[str, Any]) -> np.ndarray:
        # checks the possible position keys. 
        key, value = _first_present(vehicle_state, _POSITION_KEYS)
        if key is None:
            raise KeyError(
                "vehicle_state must contain one of "
                f"{_POSITION_KEYS!r} for vehicle position"
            )

        return _coerce_xyz_vector(value, name=key)

    def _extract_velocity_xyz(self, vehicle_state: Mapping[str, Any]) -> np.ndarray:
        # checks the possible velocity keys, can be missing or None if the vehicle is 
        # stationary at the start of a lap
        key, value = _first_present(vehicle_state, _VELOCITY_KEYS)
        if key is None:
            return np.zeros(3, dtype=float)

        return _coerce_xyz_vector(value, name=key)

    def _extract_heading_rad(
        self,
        vehicle_state: Mapping[str, Any],
        velocity_xyz: np.ndarray,
    ) -> float | None:
        # Priority 1: explicit heading:
        # Try to get the car’s heading angle in radians.
        # Prefer an explicit heading because it is the least ambiguous signal:
        # it describes where the car is pointing, not where it is sliding.
        key, value = _first_present(vehicle_state, _HEADING_RAD_KEYS)
        if key is not None: # If found, use it
            if value is None:
                raise ValueError(f"{key} must be a finite heading in radians")
            try:
                heading_rad = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a finite heading in radians") from exc

            if not math.isfinite(heading_rad):
                raise ValueError(f"{key} must be finite, got {value!r}")
            return heading_rad

        # Priority 2: direction vector:
        key, value = _first_present(vehicle_state, _DIRECTION_KEYS)
        if key is not None:
            direction_xyz = _coerce_xyz_vector(value, name=key)
            direction_norm_xy = float(np.linalg.norm(direction_xyz[:2]))
            if direction_norm_xy > _MIN_VECTOR_NORM:
                return float(math.atan2(direction_xyz[1], direction_xyz[0]))

        # Priority 3: velocity direction:
        # As a last useful fallback, infer heading from velocity direction.
        # This is less correct during drifting/reversing, so it is deliberately
        # lower priority than heading_rad or dir/direction.
        velocity_norm_xy = float(np.linalg.norm(velocity_xyz[:2]))
        if velocity_norm_xy > _MIN_VECTOR_NORM:
            return float(math.atan2(velocity_xyz[1], velocity_xyz[0]))

        # Returning None allows the caller to keep a neutral heading feature for
        # reset frames where neither heading nor movement is available.
        return None

    def _lookahead_heading_error_norm(
        self,
        *,
        position_xyz: np.ndarray,
        heading_rad: float | None,
        lookahead_point_xyz: np.ndarray,
    ) -> float:
        """
        This calculates a normalised angle error between:
            - where the car is facing
              &
            - the direction from the car to a future centreline point
        """

        # If there is no heading there is no meaningful angle error.
        if heading_rad is None:
            return 0.0

        # This creates a vector from the car to the future track point.
        to_lookahead_xy = np.asarray(lookahead_point_xyz[:2], dtype=float) - position_xyz[:2]
        # if the lookahead point is basically on top of the car, the heading error 0
        if float(np.linalg.norm(to_lookahead_xy)) <= _MIN_VECTOR_NORM:
            return 0.0

        # Convert the vector to the lookahead point into an angle.
        lookahead_heading_rad = float(math.atan2(to_lookahead_xy[1], to_lookahead_xy[0]))
        
        # Calculate the wrapped angle difference.
        lookahead_error_rad = self.track.heading_error(heading_rad, lookahead_heading_rad)
        # return clipped normalised error 
        return _clip_signed(lookahead_error_rad / math.pi)

    def _vehicle_frame_speeds(
        self,
        *,
        velocity_xyz: np.ndarray,
        heading_rad: float,
    ) -> tuple[float, float, float]:
        """This converts world velocity into vehicle-relative speeds."""

        # This creates a unit vector pointing in the direction the car is facing.
        forward_vector_xy = np.asarray(
            [math.cos(heading_rad), math.sin(heading_rad)], dtype=float,)

        # The requested lateral axis is [-sin(theta), cos(theta)]. In the
        # coordinate convention used by query_utils, this is the perpendicular
        # vehicle-frame axis used to measure sideways motion.
        right_vector_xy = np.asarray(
            [-math.sin(heading_rad), math.cos(heading_rad)],
            dtype=float,
        )

        # Take only X and Y velocity for horizontal movement.
        velocity_xy = velocity_xyz[:2]
        # dot products show how much of the velocity is in the forward and lateral directions
        # basically speed in the direction the car is facing, and speed perpendicular to 
        # that direction
        forward_speed = float(np.dot(velocity_xy, forward_vector_xy))
        lateral_speed = float(np.dot(velocity_xy, right_vector_xy))
        # z is already vertical
        vertical_speed = float(velocity_xyz[2])

        return forward_speed, lateral_speed, vertical_speed


def _first_present(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[str | None, Any | None]:
    """Helper function that searches a dictionary for the first usable key."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return key, value
    return None, None


def _coerce_xyz_vector(value: Any, *, name: str) -> np.ndarray:
    """Coerce common BeamNG vector shapes into a finite length-3 array."""

    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise ValueError(f"{name} mapping must contain at least x and y keys")
        raw_values = [value["x"], value["y"], value.get("z", 0.0)]
    else:
        try:
            raw_values = list(np.asarray(value, dtype=float).reshape(-1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric vector") from exc

        if len(raw_values) == 2:
            # TrackCentreline accepts 2D positions, but this builder stores
            # internal vectors as XYZ so speed and lookahead calculations have
            # one predictable shape.
            raw_values.append(0.0)

    # Convert raw values to flat NumPy array
    vector = np.asarray(raw_values, dtype=float).reshape(-1)
    if vector.shape[0] != 3:
        raise ValueError(f"{name} must contain 2 or 3 values, got {vector.shape[0]}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values, got {value!r}")

    return vector


def _require_positive_finite(value: float, name: str) -> None:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite value, got {value!r}")


def _clip_signed(value: float) -> float:
    # Signed normalised features use the common neural-network-friendly range.
    return float(np.clip(float(value), -1.0, 1.0))


def _clip_unit(value: float) -> float:
    # clipping progress ratio to [0, 1]
    return float(np.clip(float(value), 0.0, 1.0))


def _distance_label(distance_m: float) -> str:
    # Feature names are easier to read as "20m" than "20.0m". If a custom
    # config uses a non-integer distance, keep the decimal value in the name so
    # logs remain honest.
    distance = float(distance_m)
    if distance.is_integer():
        return f"{int(distance)}m"
    return f"{distance:g}m"


def _smoke_test() -> None:
    """Temporary manual smoke test for repository-root execution."""

    repo_root = Path(__file__).resolve().parents[3]
    centreline_path = (
        repo_root / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    )

    track = TrackCentreline.from_json(centreline_path, closed_loop=True)
    builder = ObservationBuilder(track)

    # Pick a real centreline point and nudge it sideways so the fake state
    # exercises lateral error, heading error, lookahead, curvature, and progress
    # in one simple example.
    base_position = track.points_xyz[25] + np.asarray([1.5, -0.75, 0.0], dtype=float)
    base_query = track.query(base_position)
    fake_heading_rad = float(base_query.track_heading_rad + 0.2)
    fake_speed_mps = 25.0

    fake_vehicle_state = {
        "pos": base_position.tolist(),
        "velocity": [
            math.cos(fake_heading_rad) * fake_speed_mps,
            math.sin(fake_heading_rad) * fake_speed_mps,
            0.5,
        ],
        "heading_rad": fake_heading_rad,
    }

    observation = builder.build(fake_vehicle_state)

    print(f"Loaded: {centreline_path}")
    print("Observation features:")
    for name, value in zip(builder.feature_names(), observation):
        print(f"  {name}: {value:.4f}")
    print(f"Observation shape: {builder.observation_shape()}")
    print(f"Low: {np.array2string(builder.low(), precision=1)}")
    print(f"High: {np.array2string(builder.high(), precision=1)}")


if __name__ == "__main__":
    _smoke_test()
