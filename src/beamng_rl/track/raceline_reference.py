"""Racing-line reference for the v22 reward.

Loads a precomputed racing line (physics_raceline.json or an AI raceline export)
and exposes, indexed by *centreline* arc-length progress:

    - the signed lateral offset of the racing line from the centreline, and
    - the target speed from the racing line's speed profile.

The racing line has its own arc-length parameterisation, so on load we project
every centreline point onto the nearest racing-line point and store the offset
and target speed per centreline index. Reward-time lookups are then an O(1)
index-by-progress, matching how the reward already consumes the centreline.

The offset is recomputed here (rather than read from the JSON's
``static_lateral_error_m``) so its sign matches query_utils' ``signed_lateral_error``
(+ = left of centreline), keeping the reward's raceline-lateral term consistent
with the env's measured lateral error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from beamng_rl.track.query_utils import TrackCentreline
except ModuleNotFoundError as exc:  # pragma: no cover - direct-execution fallback
    if exc.name != "beamng_rl":
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from beamng_rl.track.query_utils import TrackCentreline


class RaceLineReference:
    """Maps centreline progress → (raceline lateral offset, target speed)."""

    def __init__(
        self,
        raceline_xy: np.ndarray,
        speed_profile_mps: np.ndarray,
        centreline: TrackCentreline,
    ) -> None:
        raceline_xy = np.asarray(raceline_xy, dtype=float).reshape(-1, 2)
        speed_profile_mps = np.asarray(speed_profile_mps, dtype=float).reshape(-1)
        if raceline_xy.shape[0] != speed_profile_mps.shape[0]:
            raise ValueError(
                "raceline_xy and speed_profile_mps must have the same length, got "
                f"{raceline_xy.shape[0]} and {speed_profile_mps.shape[0]}"
            )

        self.centreline = centreline
        centre_xy = centreline.points_xyz[:, :2]
        # Per-point forward tangent: reuse precomputed segment directions; the
        # final point mirrors the last segment so counts match.
        tangents = centreline.segment_track_direction_xy
        tangents = np.vstack([tangents, tangents[-1]])

        n_centre = centre_xy.shape[0]
        self._offset_m = np.zeros(n_centre, dtype=float)
        self._target_speed_mps = np.zeros(n_centre, dtype=float)

        for i in range(n_centre):
            deltas = raceline_xy - centre_xy[i]
            dist_sq = np.einsum("ij,ij->i", deltas, deltas)
            nearest = int(np.argmin(dist_sq))
            offset_vec = raceline_xy[nearest] - centre_xy[i]
            tx, ty = tangents[i]
            # 2D cross product → signed offset (+ = left of centreline).
            self._offset_m[i] = tx * offset_vec[1] - ty * offset_vec[0]
            self._target_speed_mps[i] = speed_profile_mps[nearest]

        self._lap_length = float(centreline.total_lap_length)
        self._n = n_centre

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        raceline_json_path: str | Path,
        centreline: TrackCentreline,
    ) -> "RaceLineReference":
        path = Path(raceline_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        # physics_raceline.json / ai_*_raceline_*.json store a "points" list,
        # each with x, y and speed_mps (plus progress_m, static_lateral_error_m).
        points = data.get("points")
        if not points:
            raise ValueError(
                f"{path} must contain a non-empty 'points' list "
                "(physics_raceline.json format)"
            )

        raceline_xy = np.asarray(
            [[float(p["x"]), float(p["y"])] for p in points], dtype=float
        )
        speed_profile = np.asarray(
            [float(p.get("speed_mps", p.get("speed", 0.0))) for p in points],
            dtype=float,
        )
        return cls(raceline_xy, speed_profile, centreline)

    # ------------------------------------------------------------------
    # Reward-time lookup
    # ------------------------------------------------------------------

    def _index_at_progress(self, lap_progress_m: float) -> int:
        progress = float(lap_progress_m) % self._lap_length
        idx = int(np.searchsorted(self.centreline.cumulative_arc_lengths, progress))
        return min(max(idx, 0), self._n - 1)

    def offset_at(self, lap_progress_m: float) -> float:
        """Signed lateral offset (m) of the racing line from the centreline."""
        return float(self._offset_m[self._index_at_progress(lap_progress_m)])

    def target_speed_at(self, lap_progress_m: float) -> float:
        """Target speed (m/s) from the racing line speed profile."""
        return float(self._target_speed_mps[self._index_at_progress(lap_progress_m)])

    def query(self, lap_progress_m: float) -> tuple[float, float]:
        """Return (signed_offset_m, target_speed_mps) at the given progress."""
        idx = self._index_at_progress(lap_progress_m)
        return float(self._offset_m[idx]), float(self._target_speed_mps[idx])


def _smoke_test() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    centre_path = repo_root / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"
    raceline_path = repo_root / "data" / "hirochi_track" / "physics_raceline.json"

    centreline = TrackCentreline.from_json(centre_path, closed_loop=True)
    ref = RaceLineReference.from_json(raceline_path, centreline)

    print(f"Centreline points : {ref._n}")
    print(f"Lap length        : {ref._lap_length:.1f} m")
    print(f"Offset range      : {ref._offset_m.min():+.2f} .. {ref._offset_m.max():+.2f} m")
    print(f"Target speed range: {ref._target_speed_mps.min():.1f} .. {ref._target_speed_mps.max():.1f} m/s")
    for p in (0.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0):
        off, spd = ref.query(p)
        print(f"  progress {p:>7.1f} m -> offset {off:+6.2f} m, target {spd:5.1f} m/s ({spd*3.6:5.1f} km/h)")


if __name__ == "__main__":
    _smoke_test()
