"""Reusable centreline query utilities for BeamNG reinforcement learning.

The main class in this module, :class:`TrackCentreline`, loads an ordered
centreline and answers common questions an RL environment usually needs:

* Where is the vehicle relative to the track centreline?
* How far around the lap has it progressed?
* What is the local track heading?
* Where is a point some metres ahead?
* How curved is the centreline near the vehicle?

BeamNG track points are stored as XYZ coordinates because the road has
elevation. For most driving-control geometry, however, we intentionally work
in the XY plane: steering, lateral error, heading, and progress are naturally
defined on the map/ground plane. Z is preserved and interpolated for returned
points, but it does not affect nearest-segment projection or heading.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# small threshold to disregard negligible segment lengths
_MIN_SEGMENT_LENGTH_M = 1.0e-9


@dataclass
class TrackQueryResult:
    """

    Object that stores result of centreline query

    """

    nearest_point_xyz: np.ndarray
    nearest_point_xy: np.ndarray
    segment_index: int
    segment_progress: float
    distance_to_centerline: float
    signed_lateral_error: float
    lap_progress: float
    lap_progress_ratio: float
    track_heading_rad: float
    track_tangent_xy: np.ndarray    # road direction as an [x, y] arrow


def heading_error(vehicle_heading_rad: float, track_heading_rad: float) -> float:
    """
    - Comparing the car's heading with the track's heading, returning:
      vehicle heading - track heading 
    - Angles are wrapped around +/-pi to account for difference between 
      +179 degrees & -179 degrees
    """

    # float conversion
    vehicle_heading = float(vehicle_heading_rad)
    track_heading = float(track_heading_rad)

    # error handling
    if not math.isfinite(vehicle_heading):
        raise ValueError(f"vehicle_heading_rad must be finite, got {vehicle_heading_rad!r}")
    if not math.isfinite(track_heading):
        raise ValueError(f"track_heading_rad must be finite, got {track_heading_rad!r}")

    raw_error = vehicle_heading - track_heading
    return math.atan2(math.sin(raw_error), math.cos(raw_error))


class TrackCentreline:
    """An ordered track centreline with fast projection/query helpers.

    Parameters
    ----------
    centreline_points:
        Ordered centreline points as an ``(N, 3)`` array-like object.
    closed_loop:
        If true, the final point connects back to the first point. This is the
        normal setting for a racing circuit.
    metadata:
        Optional JSON metadata retained for debugging or logging.
    """

    # static method so it can be used without TrackCentreline instance
    heading_error = staticmethod(heading_error)

    def __init__(
        self,
        centreline_points: np.ndarray | Sequence[Sequence[float]],
        *,
        
        closed_loop: bool = True,
        metadata: Mapping[str, Any] | None = None, 
        ) -> None:
        

        self.closed_loop = bool(closed_loop)
        self.metadata: dict[str, Any] = dict(metadata or {})

        points = self.json_points_to_NumPy_array(centreline_points)
        self._validate_point_count(points, self.closed_loop)


        # copy of centreline
        self.points_xyz = points.copy()

        # centreline without z
        self.points_xy = self.points_xyz[:, :2]

        if self.closed_loop: # closed loop, every point connects to next
            self.segment_start_points_xyz = self.points_xyz
            # last point connect to first
            self.segment_end_points_xyz = np.roll(self.points_xyz, shift=-1, axis=0)  
           
        else: # open loop, final point is end of last segment
            self.segment_start_points_xyz = self.points_xyz[:-1]
            self.segment_end_points_xyz = self.points_xyz[1:]

        self.segment_start_points_xy = self.segment_start_points_xyz[:, :2]
        self.segment_end_points_xy = self.segment_end_points_xyz[:, :2]


        # Directional vectors from segment start to end
        self.segment_vectors_xyz = self.segment_end_points_xyz - self.segment_start_points_xyz
        self.segment_vectors_xy = self.segment_end_points_xy - self.segment_start_points_xy

        # Length of each segment
        self.segment_lengths = np.linalg.norm(self.segment_vectors_xy, axis=1) # vector length
        self._validate_segment_lengths()

        self.segment_count = int(self.segment_lengths.shape[0])


        # cumulative_arc_lengths stores segment boundary distances:
        #   [0, length(seg0), length(seg0)+length(seg1), ..., total]
        # The start distance for segment i is cumulative_arc_lengths[i]. Query
        # progress is therefore:
        #   start_distance_of_segment + t_along_segment * segment_length

        # running total of segment lengths, final value = total lap length
        self.cumulative_arc_lengths = np.concatenate(
            ([0.0], np.cumsum(self.segment_lengths, dtype=float)))
        
        self.segment_start_progress_m = self.cumulative_arc_lengths[:-1]
        self.total_lap_length = float(self.cumulative_arc_lengths[-1])

        if self.total_lap_length <= _MIN_SEGMENT_LENGTH_M:
            raise ValueError("centreline total length must be positive")

        # local track directions in XY
        self.segment_track_direction_xy = self.segment_vectors_xy / self.segment_lengths[:, None]


    @classmethod
    def from_json(
        cls,
        path: str | Path,
        closed_loop: bool = True,
        drop_duplicate_endpoint: bool = True,
    ) -> "TrackCentreline":
        """

        Create a TrackCentreline object from a JSON file.

        """

        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        # Error handling
        if not isinstance(data, Mapping):
            raise ValueError(f"{json_path} must contain a JSON object")
        if "points" not in data:
            raise ValueError(f"{json_path} is missing required key 'points'")

        points = cls._points_from_json(data["points"], source_path=json_path)

        if closed_loop and drop_duplicate_endpoint and len(points) >= 2:
            spacing_m = cls._positive_float_or_none(data.get("spacing"))
            duplicate_threshold_m = cls._duplicate_endpoint_threshold(points, spacing_m)
            closing_gap_m = float(np.linalg.norm(points[-1, :2] - points[0, :2]))

            # If the final point is very close to the first point, it's a duplicate
            if closing_gap_m <= duplicate_threshold_m:
                points = points[:-1]

        metadata = {key: value for key, value in data.items() if key != "points"}
        metadata["path"] = str(json_path)

        return cls(points, closed_loop=closed_loop, metadata=metadata)


    def query(
        self,
        position_xyz: Sequence[float] | np.ndarray,
        vehicle_heading_rad: float | None = None,
    ) -> TrackQueryResult:
        """
        Project a vehicle position onto the centreline.

        - find the nearest point on the track centreline to the vehicle
        - compute relevant RL geometry:
            - distance to centreline
            - signed lateral error (+left, -right)
            - progress around the lap (metres & ratio)
            - track heading at the nearest point
        """

        vehicle_xy = self._coerce_position_xy(position_xyz)

        # Vector from each segment start to the vehicle
        segment_start_to_vehicle = vehicle_xy[None, :] - self.segment_start_points_xy

        # Project segment_start_to_vehicle onto each segment vector. Dividing by 
        # the squared length converts the dot product into the segment parameter t.
        # t=0 means nearest at the start point, t=1 at the end point, and
        # values between them are interpolated points along the segment.

        """
        calculate the projection of the vehicle position onto each segment vector
        using the dot product. The result is a progression parameter indicates how far 
        along the segment the projection falls. 
        We then clip this parameter to the range [0, 1] to ensure that the projection 
        stays within the segment bounds.
        """
        segment_length_sq = self.segment_lengths * self.segment_lengths
        raw_segment_progression_projection = np.einsum("ij,ij->i", 
            segment_start_to_vehicle, self.segment_vectors_xy) / segment_length_sq
        segment_progression_projection = np.clip(raw_segment_progression_projection, 0.0, 1.0)

        # TODO: add comments from here onwards...
        nearest_points_xy = (
            self.segment_start_points_xy + segment_progression_projection[:, None] * self.segment_vectors_xy
        )
        vehicle_offsets_xy = vehicle_xy[None, :] - nearest_points_xy
        distance_sq = np.einsum("ij,ij->i", vehicle_offsets_xy, vehicle_offsets_xy)

        best_segment_index = int(np.argmin(distance_sq))
        segment_t = float(segment_progression_projection[best_segment_index])

        nearest_point_xy = nearest_points_xy[best_segment_index].copy()
        nearest_point_xyz = (
            self.segment_start_points_xyz[best_segment_index]
            + segment_t * self.segment_vectors_xyz[best_segment_index]
        )

        distance_to_centerline = float(math.sqrt(float(distance_sq[best_segment_index])))

        tangent_xy = self.segment_track_direction_xy[best_segment_index].copy()
        vehicle_offset_xy = vehicle_xy - nearest_point_xy

        # Signed lateral error uses the 2D cross product:
        #   tangent_x * offset_y - tangent_y * offset_x
        # Because tangent_xy has unit length, the result is the perpendicular
        # distance from the centreline tangent to the vehicle. Positive means
        # the vehicle is to the left of the ordered track direction; negative
        # means it is to the right. Near segment endpoints, this signed value
        # describes side-of-track, while distance_to_centerline is still the
        # true Euclidean distance to the nearest centreline point.
        signed_lateral_error = float(
            tangent_xy[0] * vehicle_offset_xy[1] - tangent_xy[1] * vehicle_offset_xy[0]
        )

        # Progress is the distance at the segment start plus the fraction of
        # the current segment reached by the projection. For a closed lap we
        # wrap total_length back to 0, so progress always means "where am I on
        # this lap?" rather than "how many metres have I ever driven?".
        raw_progress_m = (
            float(self.segment_start_progress_m[best_segment_index])
            + segment_t * float(self.segment_lengths[best_segment_index])
        )
        progress_m = self._normalise_progress(raw_progress_m)
        progress_ratio = progress_m / self.total_lap_length

        track_heading_rad = float(math.atan2(tangent_xy[1], tangent_xy[0]))

        if vehicle_heading_rad is not None:
            # Validate the optional heading and make the intended companion API
            # obvious. The result dataclass stays focused on track geometry;
            # environments can compute this only when they actually need it:
            #   heading_error(vehicle_heading, result.track_heading_rad)
            heading_error(vehicle_heading_rad, track_heading_rad)

        return TrackQueryResult(
            nearest_point_xyz=nearest_point_xyz.copy(),
            nearest_point_xy=nearest_point_xy,
            segment_index=best_segment_index,
            segment_progress=segment_t,
            distance_to_centerline=distance_to_centerline,
            signed_lateral_error=signed_lateral_error,
            lap_progress=progress_m,
            lap_progress_ratio=progress_ratio,
            track_heading_rad=track_heading_rad,
            track_tangent_xy=tangent_xy,
        )

    def point_at_progress(self, progress_m: float) -> np.ndarray:
        """Interpolate an XYZ centreline point at a lap progress distance.

        For closed tracks, progress wraps with modulo total length. For example,
        asking for total_length + 10 m returns the point 10 m after the start.
        This is what makes lookahead work smoothly across the start/finish line.

        For open tracks, progress clamps to the available range so callers get
        the first or final point instead of an out-of-bounds error.
        """

        progress = self._normalise_progress(progress_m)
        segment_index = self._segment_index_at_progress(progress)
        segment_start_progress = float(self.cumulative_arc_lengths[segment_index])
        segment_length = float(self.segment_lengths[segment_index])

        segment_t = (progress - segment_start_progress) / segment_length
        segment_t = float(np.clip(segment_t, 0.0, 1.0))

        return (
            self.segment_start_points_xyz[segment_index]
            + segment_t * self.segment_vectors_xyz[segment_index]
        ).copy()

    def lookahead_point(self, progress_m: float, lookahead_m: float) -> np.ndarray:
        """Return an XYZ point ``lookahead_m`` metres after ``progress_m``.

        On a closed loop this simply adds the lookahead distance and lets
        :meth:`point_at_progress` wrap around modulo lap length. That means a
        policy near the end of the lap can request, say, a 20 m lookahead and
        receive a point just after the start/finish line instead of falling off
        the end of the array.
        """

        lookahead = float(lookahead_m)
        if not math.isfinite(lookahead):
            raise ValueError(f"lookahead_m must be finite, got {lookahead_m!r}")

        return self.point_at_progress(float(progress_m) + lookahead)

    def curvature_at(self, progress_m: float, sample_distance_m: float = 5.0) -> float:
        """Estimate unsigned XY curvature near ``progress_m``.

        We sample three centreline points: one before the progress, one at the
        progress, and one after it. Those three points define a triangle. If the
        triangle has a stable, non-zero area, the circumcircle through the
        points approximates the local path curve, and curvature is 1/radius:

            curvature = 4 * triangle_area / (side_a * side_b * side_c)

        Straight or nearly straight samples have almost zero triangle area, and
        duplicate/clamped samples at open-track ends can make side lengths
        degenerate. In those cases the safest curvature estimate is 0.0.
        """

        sample_distance = float(sample_distance_m)
        if not math.isfinite(sample_distance) or sample_distance <= 0.0:
            raise ValueError(
                f"sample_distance_m must be a positive finite value, got {sample_distance_m!r}"
            )

        prev_xy = self.point_at_progress(float(progress_m) - sample_distance)[:2]
        curr_xy = self.point_at_progress(progress_m)[:2]
        next_xy = self.point_at_progress(float(progress_m) + sample_distance)[:2]

        side_a = float(np.linalg.norm(curr_xy - prev_xy))
        side_b = float(np.linalg.norm(next_xy - curr_xy))
        side_c = float(np.linalg.norm(next_xy - prev_xy))

        if (
            side_a <= _MIN_SEGMENT_LENGTH_M
            or side_b <= _MIN_SEGMENT_LENGTH_M
            or side_c <= _MIN_SEGMENT_LENGTH_M
        ):
            return 0.0

        vec_ab = curr_xy - prev_xy
        vec_ac = next_xy - prev_xy
        twice_area = abs(float(vec_ab[0] * vec_ac[1] - vec_ab[1] * vec_ac[0]))

        if twice_area <= _MIN_SEGMENT_LENGTH_M:
            return 0.0

        denominator = side_a * side_b * side_c
        if denominator <= _MIN_SEGMENT_LENGTH_M:
            return 0.0

        curvature = 2.0 * twice_area / denominator
        if not math.isfinite(curvature) or curvature <= _MIN_SEGMENT_LENGTH_M:
            return 0.0

        return float(curvature)

    def _normalise_progress(self, progress_m: float) -> float:
        """Wrap or clamp a progress value according to track topology."""

        progress = float(progress_m)
        if not math.isfinite(progress):
            raise ValueError(f"progress_m must be finite, got {progress_m!r}")

        if self.closed_loop:
            return progress % self.total_lap_length

        return float(np.clip(progress, 0.0, self.total_lap_length))

    def _segment_index_at_progress(self, progress_m: float) -> int:
        """Find the segment containing a normalised progress value."""

        segment_index = int(np.searchsorted(self.cumulative_arc_lengths, progress_m, side="right") - 1)
        return int(np.clip(segment_index, 0, self.segment_count - 1))

    def _validate_segment_lengths(self) -> None:
        bad_indices = np.flatnonzero(self.segment_lengths <= _MIN_SEGMENT_LENGTH_M)
        if bad_indices.size == 0:
            return

        preview = ", ".join(str(int(index)) for index in bad_indices[:8])
        if bad_indices.size > 8:
            preview += ", ..."

        loop_note = (
            " For a closed loop, this can happen if the final point duplicates the first "
            "point and drop_duplicate_endpoint=False was used."
        )
        raise ValueError(f"centreline contains zero-length XY segment(s) at index {preview}.{loop_note}")

    @staticmethod
    def json_points_to_NumPy_array(points_xyz: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=float)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "centreline points must be a numeric array with shape (N, 3); "
                f"got shape {points.shape}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("centreline points must all be finite numbers")

        return points

    @staticmethod
    def _validate_point_count(points: np.ndarray, closed_loop: bool) -> None:
        min_count = 3 if closed_loop else 2
        if points.shape[0] < min_count:
            topology = "closed-loop" if closed_loop else "open"
            raise ValueError(
                f"{topology} centreline requires at least {min_count} points; "
                f"got {points.shape[0]}"
            )

    @staticmethod
    def _coerce_position_xy(position_xyz: Sequence[float] | np.ndarray) -> np.ndarray:
        position = np.asarray(position_xyz, dtype=float).reshape(-1)

        if position.shape[0] not in (2, 3):
            raise ValueError(
                "position_xyz must contain either (x, y) or (x, y, z); "
                f"got {position.shape[0]} value(s)"
            )
        if not np.all(np.isfinite(position)):
            raise ValueError(f"position_xyz must contain finite numbers, got {position_xyz!r}")

        return position[:2].copy()

    @staticmethod
    def _points_from_json(points_json: Any, *, source_path: Path) -> np.ndarray:
        if not isinstance(points_json, Sequence) or isinstance(points_json, (str, bytes)):
            raise ValueError(f"{source_path} key 'points' must be a list of point objects")

        rows: list[tuple[float, float, float]] = []
        for index, point in enumerate(points_json):
            if isinstance(point, Mapping):
                missing = [axis for axis in ("x", "y", "z") if axis not in point]
                if missing:
                    raise ValueError(
                        f"{source_path} point {index} is missing required key(s): {missing}"
                    )
                row = (point["x"], point["y"], point["z"])
            else:
                try:
                    values = list(point)
                except TypeError as exc:
                    raise ValueError(
                        f"{source_path} point {index} must be an object with x/y/z "
                        "or a sequence of at least three numbers"
                    ) from exc

                if len(values) < 3:
                    raise ValueError(
                        f"{source_path} point {index} must contain at least three values"
                    )
                row = (values[0], values[1], values[2])

            try:
                rows.append((float(row[0]), float(row[1]), float(row[2])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source_path} point {index} contains non-numeric coordinate(s): {row!r}"
                ) from exc

        return TrackCentreline.json_points_to_NumPy_array(rows)

    @staticmethod
    def _positive_float_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if math.isfinite(number) and number > 0.0:
            return number
        return None

    @staticmethod
    def _duplicate_endpoint_threshold(points_xyz: np.ndarray, spacing_m: float | None) -> float:
        """Choose a practical XY threshold for removing a duplicate endpoint."""

        if spacing_m is not None:
            # The resampled file says points are about spacing_m apart, so a
            # closing gap under half that spacing is a short duplicate tail
            # rather than a meaningful extra segment.
            return max(_MIN_SEGMENT_LENGTH_M, 0.5 * spacing_m)

        # If the JSON has no spacing metadata, infer the typical sample spacing
        # from neighbouring XY distances.
        step_lengths = np.linalg.norm(np.diff(points_xyz[:, :2], axis=0), axis=1)
        usable_lengths = step_lengths[step_lengths > _MIN_SEGMENT_LENGTH_M]
        if usable_lengths.size == 0:
            return _MIN_SEGMENT_LENGTH_M

        return max(_MIN_SEGMENT_LENGTH_M, 0.5 * float(np.median(usable_lengths)))


if __name__ == "__main__":
    # Small smoke-test / usage example. Running:
    #   python src/beamng_rl/track/query_utils.py
    # from the repository root loads the Hirochi centreline and prints the
    # values an RL environment would typically place into an observation.
    repo_root = Path(__file__).resolve().parents[3]
    centreline_path = repo_root / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"

    track = TrackCentreline.from_json(centreline_path, closed_loop=True)

    # Fake a vehicle position by taking a known centreline point and nudging it
    # in XY. A real environment would pass the BeamNG vehicle position here.
    fake_vehicle_position = track.points_xyz[25] + np.array([1.5, -0.75, 0.0])
    result = track.query(fake_vehicle_position)

    lookahead = track.lookahead_point(result.lap_progress, lookahead_m=20.0)
    curvature = track.curvature_at(result.lap_progress)

    print(f"Loaded: {centreline_path}")
    print(f"Total lap length: {track.total_lap_length:.2f} m")
    print(f"Nearest point XYZ: {np.array2string(result.nearest_point_xyz, precision=3)}")
    print(f"Progress: {result.lap_progress:.2f} m")
    print(f"Progress ratio: {result.lap_progress_ratio:.4f}")
    print(f"Signed lateral error: {result.signed_lateral_error:.3f} m")
    print(f"Track heading: {result.track_heading_rad:.3f} rad")
    print(f"Lookahead point XYZ: {np.array2string(lookahead, precision=3)}")
    print(f"Curvature: {curvature:.6f} 1/m")
