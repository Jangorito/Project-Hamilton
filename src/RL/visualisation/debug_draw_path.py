from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


Float3 = tuple[float, float, float]
Color = tuple[float, float, float, float]


@dataclass(slots=True)
class DebugPathHandles:
    """Stores BeamNG debug-object IDs so they can be removed later."""

    polyline_id: int | None = None
    sphere_ids: list[int] = field(default_factory=list)
    text_ids: list[int] = field(default_factory=list)

    def all_empty(self) -> bool:
        return self.polyline_id is None and not self.sphere_ids and not self.text_ids


class DebugPathDrawer:
    """
    Draws a path in BeamNG using the built-in debug API.

    This uses the BeamNGpy debug API rather than procedural meshes because the
    debug polyline/sphere primitives are lightweight, easy to update/remove, and
    ideal for validating waypoint / centre-line geometry during development.
    """

    def __init__(self, beamng) -> None:
        self.beamng = beamng
        self.debug = beamng.debug
        self._handles: list[DebugPathHandles] = []

    def draw_path(
        self,
        coordinates: Sequence[Sequence[float]],
        *,
        line_color: Color = (0.1, 0.8, 1.0, 1.0),
        sphere_color: Color = (1.0, 0.45, 0.1, 1.0),
        text_color: Color = (1.0, 1.0, 1.0, 1.0),
        cling: bool = False,
        offset: float = 0.0,
        sphere_radius: float = 0.35,
        sphere_every: int = 25,
        label_every: int | None = None,
        start_end_markers: bool = True,
    ) -> DebugPathHandles:
        """
        Draw a path and optional marker spheres/text labels in the simulator.

        Parameters
        ----------
        coordinates:
            Iterable of ``(x, y, z)`` points.
        cling / offset:
            Passed straight through to BeamNG debug objects. Use ``cling=True``
            if your input points are 2D plan-view coordinates and you want the
            path aligned to terrain height.
        sphere_every:
            Place a marker every N points. Set to 0 or a negative value to
            disable intermediate spheres.
        label_every:
            Place an index label every N points. ``None`` disables labels.
        start_end_markers:
            Always mark the first and last point.
        """
        pts = _normalize_points(coordinates)
        if len(pts) < 2:
            raise ValueError("At least two coordinates are required to draw a path.")

        handles = DebugPathHandles()
        handles.polyline_id = self.debug.add_polyline(
            coordinates=pts,
            rgba_color=line_color,
            cling=cling,
            offset=offset,
        )

        marker_points: list[Float3] = []
        marker_radii: list[float] = []
        marker_colors: list[Color] = []

        if sphere_every and sphere_every > 0:
            for i in range(0, len(pts), sphere_every):
                marker_points.append(pts[i])
                marker_radii.append(sphere_radius)
                marker_colors.append(sphere_color)

        if start_end_markers:
            endpoints = [pts[0], pts[-1]]
            endpoint_colors = [(0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
            for point, color in zip(endpoints, endpoint_colors, strict=True):
                if point not in marker_points:
                    marker_points.append(point)
                    marker_radii.append(sphere_radius * 1.4)
                    marker_colors.append(color)

        if marker_points:
            handles.sphere_ids = self.debug.add_spheres(
                coordinates=marker_points,
                radii=marker_radii,
                rgba_colors=marker_colors,
                cling=cling,
                offset=offset,
            )

        if label_every is not None and label_every > 0:
            for i in range(0, len(pts), label_every):
                text_id = self.debug.add_text(
                    origin=pts[i],
                    content=str(i),
                    rgba_color=text_color,
                    cling=cling,
                    offset=offset + max(sphere_radius * 2.5, 0.75),
                )
                handles.text_ids.append(text_id)

        self._handles.append(handles)
        return handles

    def clear(self, handles: DebugPathHandles | None = None) -> None:
        """Remove one previously drawn path, or all if ``handles`` is None."""
        if handles is None:
            for stored in list(self._handles):
                self._remove_handles(stored)
            self._handles.clear()
            return

        self._remove_handles(handles)
        self._handles = [h for h in self._handles if h is not handles]

    def _remove_handles(self, handles: DebugPathHandles) -> None:
        if handles.polyline_id is not None:
            self.debug.remove_polyline(handles.polyline_id)
        if handles.sphere_ids:
            self.debug.remove_spheres(handles.sphere_ids)
        for text_id in handles.text_ids:
            self.debug.remove_text(text_id)


def load_path_points(path: str | Path) -> list[Float3]:
    """
    Load path coordinates from a simple file format.

    Supported formats
    -----------------
    - ``.json``: ``[[x, y, z], ...]`` or ``[{"x":..., "y":..., "z":...}, ...]``
    - ``.csv``: headers ``x,y,z`` (or uppercase variants)
    - ``.txt`` / ``.xyz``: whitespace- or comma-separated rows: ``x y z``
    - ``.npy``: ``Nx3`` array (requires NumPy)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return _points_from_json_like(data)

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return _points_from_dict_rows(rows)

    if suffix in {".txt", ".xyz"}:
        points: list[Float3] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [p for p in stripped.replace(",", " ").split() if p]
            if len(parts) < 3:
                raise ValueError(f"Invalid point row: {line!r}")
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
        return _normalize_points(points)

    if suffix == ".npy":
        if np is None:
            raise ImportError("NumPy is required to load .npy files.")
        arr = np.load(file_path)
        return _normalize_points(arr.tolist())

    raise ValueError(
        f"Unsupported file extension '{suffix}'. Use .json, .csv, .txt, .xyz, or .npy."
    )


def _points_from_json_like(data: object) -> list[Float3]:
    if isinstance(data, dict):
        if "points" in data:
            return _points_from_json_like(data["points"])
        raise ValueError("JSON object must contain a 'points' field or be a list of points.")

    if not isinstance(data, list):
        raise ValueError("JSON content must be a list of points.")

    if not data:
        return []

    first = data[0]
    if isinstance(first, dict):
        return _points_from_dict_rows(data)

    return _normalize_points(data)


def _points_from_dict_rows(rows: Sequence[dict]) -> list[Float3]:
    aliases = {
        "x": ("x", "X"),
        "y": ("y", "Y"),
        "z": ("z", "Z"),
    }

    points: list[Float3] = []
    for row in rows:
        x = _get_first_present(row, aliases["x"])
        y = _get_first_present(row, aliases["y"])
        z = _get_first_present(row, aliases["z"])
        points.append((float(x), float(y), float(z)))
    return _normalize_points(points)


def _get_first_present(row: dict, keys: Iterable[str]) -> object:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"Could not find any of {tuple(keys)!r} in row: {row!r}")


def _normalize_points(coordinates: Sequence[Sequence[float]]) -> list[Float3]:
    pts: list[Float3] = []
    for point in coordinates:
        if len(point) < 3:
            raise ValueError(f"Point must contain at least 3 values, got: {point!r}")
        pts.append((float(point[0]), float(point[1]), float(point[2])))
    return pts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw a path in BeamNG using debug primitives.")
    parser.add_argument("path_file", type=Path, help="Path to a JSON/CSV/TXT/XYZ/NPY file of points.")
    parser.add_argument("--host", default="127.0.0.1", help="BeamNG host.")
    parser.add_argument("--port", type=int, default=25252, help="BeamNG port.")
    parser.add_argument("--sphere-every", type=int, default=25, help="Add a sphere every N points.")
    parser.add_argument("--label-every", type=int, default=None, help="Add a text label every N points.")
    parser.add_argument("--sphere-radius", type=float, default=0.35, help="Sphere radius in metres.")
    parser.add_argument("--cling", action="store_true", help="Project path markers onto terrain height.")
    parser.add_argument("--offset", type=float, default=0.0, help="Vertical offset used with --cling.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Import lazily so the module can be imported/tested without BeamNGpy installed.
    from beamngpy import BeamNGpy

    points = load_path_points(args.path_file)

    with BeamNGpy(args.host, args.port) as bng:
        drawer = DebugPathDrawer(bng)
        drawer.draw_path(
            points,
            cling=args.cling,
            offset=args.offset,
            sphere_every=args.sphere_every,
            label_every=args.label_every,
            sphere_radius=args.sphere_radius,
        )
        print(f"Drew path with {len(points)} points.")
        print("Press Enter to clear the debug objects and exit...")
        input()
        drawer.clear()


if __name__ == "__main__":
    main()
