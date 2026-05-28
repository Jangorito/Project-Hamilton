from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

# Allow running as a script directly (python bootstrap/beamng_bootstrap.py)
# without setting PYTHONPATH — module-style runs (python -m beamng_rl.bootstrap...)
# don't need this.
_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from beamngpy import BeamNGpy, set_up_simple_logging

# Keep these names available from beamng_bootstrap.py for older manual/debug
# snippets while the live env imports the side-effect-free setup module.
from beamng_rl.bootstrap.beamng_setup import (
    BEAMNG_HOME_CANDIDATES,
    BEAMNG_HOME_ENV_VAR,
    HOST,
    JANGO_BEAMNG_HOME,
    PORT,
    SPAWN_POS,
    SPAWN_ROT,
    STUDENT_BEAMNG_HOME,
    apply_low_graphics_preset,
    apply_shadow_disabling,
    build_hirochi_etkc_scenario,
    is_student_beamng_home,
    resolve_beamng_home_from_path_or_env,
)
from beamng_rl.bootstrap.graphics_settings import DEFAULT_LOW_GRAPHICS_SETTINGS_FILE
from beamng_rl.track.geometry import resample_polyline
from beamng_rl.track.physics_raceline import compute_track_boundaries
from beamng_rl.track.query_utils import TrackCentreline
from beamng_rl.track.raceline_loader import derive_race_path_from_files
from beamng_rl.visualisation.debug_draw_path import DebugPathDrawer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACK_DATA_DIR_ENV_VAR = "BEAMNG_TRACK_DATA_DIR"
STUDENT_TRACK_DATA_DIR = Path(r"C:\Users\Student\Workspace\Project-Hamilton\data\hirochi_track")
JANGO_TRACK_DATA_DIR = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track")
TRACK_DATA_DIR_CANDIDATES = (STUDENT_TRACK_DATA_DIR, JANGO_TRACK_DATA_DIR)
RACE_FILE_NAME = "race.race.json"
ITEMS_FILE_NAME = "items.level.json"

HIGHLIGHT_IDS = [30, 12, 10, 17]


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------
Float3 = Tuple[float, float, float]


def derive_path(track_data_dir: Path) -> list[Float3]:
    return derive_race_path_from_files(
        track_data_dir / RACE_FILE_NAME,
        track_data_dir / ITEMS_FILE_NAME,
        HIGHLIGHT_IDS,
    )


def export_path(
    points: list[Float3],
    output_file: Path,
    *,
    track_name: str = "hirochi_raceway",
    spacing: float | None = None,
    source: str = "derived_from_race_and_items",
) -> None:
    payload = {
        "track": track_name,
        "source": source,
        "point_count": len(points),
        "points": [{"x": x, "y": y, "z": z} for x, y, z in points],
    }

    if spacing is not None:
        payload["spacing"] = spacing

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def build_scenario(bng: BeamNGpy) -> tuple[object, object]:
    """Build the shared Hirochi/ETK K-Series scenario used by bootstrap and live env."""

    return build_hirochi_etkc_scenario(bng)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BeamNG ETK K-Series bootstrap")

    parser.add_argument(
        "--beamng-home",
        type=Path,
        default=None,
        help=(
            "BeamNG.tech install directory. Overrides the BEAMNG_HOME environment "
            "variable and known local paths."
        ),
    )
    parser.add_argument(
        "--track-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing race.race.json and items.level.json. Overrides the "
            f"{TRACK_DATA_DIR_ENV_VAR} environment variable and known local paths."
        ),
    )
    parser.add_argument(
        "--draw-path",
        action="store_true",
        help="Derive and draw the race path in-sim using debug primitives.",
    )
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        help=(
            "Student-machine smoke-test flag: after BeamNG starts, ask BeamNGpy "
            "to set GraphicDisableShadows to 2 and apply graphics settings."
        ),
    )
    parser.add_argument(
        "--low-graphics",
        action="store_true",
        help=(
            "Student-machine flag: apply the editable low-graphics settings "
            "file after BeamNG starts."
        ),
    )
    parser.add_argument(
        "--low-graphics-settings-file",
        type=Path,
        default=DEFAULT_LOW_GRAPHICS_SETTINGS_FILE,
        help=(
            "INI file containing ordered low-graphics settings for --low-graphics "
            f"(default: {DEFAULT_LOW_GRAPHICS_SETTINGS_FILE})."
        ),
    )
    parser.add_argument(
        "--sphere-every",
        type=int,
        default=25,
        help="Draw a waypoint sphere every N path points (default: 25).",
    )
    parser.add_argument(
        "--label-every",
        type=int,
        default=None,
        help="Draw an index label every N path points (default: off).",
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=2.0,
        help="Resample the derived path to uniform spacing in metres before drawing/exporting. "
             "Set to 0 or a negative value to disable resampling.",
    )
    parser.add_argument(
        "--no-raceline",
        action="store_true",
        help="Skip drawing the physics racing line even if physics_raceline.json exists.",
    )
    parser.add_argument(
        "--no-limits",
        action="store_true",
        help="Skip drawing left/right track-limit boundary lines.",
    )

    return parser


def resolve_beamng_home(args: argparse.Namespace) -> Path:
    return resolve_beamng_home_from_path_or_env(args.beamng_home)


def resolve_track_data_dir(args: argparse.Namespace) -> Path:
    track_data_dir = args.track_data_dir

    if track_data_dir is None:
        env_track_data_dir = os.environ.get(TRACK_DATA_DIR_ENV_VAR)
        if env_track_data_dir:
            track_data_dir = Path(env_track_data_dir)
        else:
            track_data_dir = next(
                (candidate for candidate in TRACK_DATA_DIR_CANDIDATES if candidate.is_dir()),
                STUDENT_TRACK_DATA_DIR,
            )

    track_data_dir = track_data_dir.expanduser()

    if not track_data_dir.exists():
        raise FileNotFoundError(
            f"Track data directory does not exist: {track_data_dir}. "
            f"Set it with --track-data-dir or the {TRACK_DATA_DIR_ENV_VAR} environment variable."
        )
    if not track_data_dir.is_dir():
        raise NotADirectoryError(f"Track data path is not a directory: {track_data_dir}")

    missing_files = [
        track_data_dir / file_name
        for file_name in (RACE_FILE_NAME, ITEMS_FILE_NAME)
        if not (track_data_dir / file_name).is_file()
    ]
    if missing_files:
        missing_list = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required track data file(s): {missing_list}")

    return track_data_dir


def main() -> None:
    args = build_arg_parser().parse_args()
    beamng_home = resolve_beamng_home(args)

    set_up_simple_logging()

    print(f"Using BeamNG home: {beamng_home}")

    bng = BeamNGpy(HOST, PORT, home=str(beamng_home))
    drawer: DebugPathDrawer | None = None

    try:
        bng.open(launch=True)

        if args.low_graphics:
            if is_student_beamng_home(beamng_home):
                apply_low_graphics_preset(bng, args.low_graphics_settings_file)
            else:
                print(
                    "WARNING: --low-graphics is currently limited to the "
                    f"Student BeamNG install ({STUDENT_BEAMNG_HOME}). "
                    f"Resolved BeamNG home was {beamng_home}; leaving graphics unchanged."
                )

        if args.disable_shadows:
            if is_student_beamng_home(beamng_home):
                apply_shadow_disabling(bng)
            else:
                print(
                    "WARNING: --disable-shadows is currently limited to the "
                    f"Student BeamNG install ({STUDENT_BEAMNG_HOME}). "
                    f"Resolved BeamNG home was {beamng_home}; leaving graphics unchanged."
                )

        scenario, vehicle = build_scenario(bng)

        bng.settings.set_deterministic(60)
        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()

        print("Ready.")

        if args.draw_path:
            track_data_dir = resolve_track_data_dir(args)

            print(f"Using track data directory: {track_data_dir}")
            print("Deriving race path...")
            raw_path = derive_path(track_data_dir)
            print(f"Derived raw path with {len(raw_path)} points.")

            raw_output = track_data_dir / "centreline_raw.json"
            export_path(raw_path, raw_output)
            print(f"Exported raw path to: {raw_output}")

            path_to_draw = raw_path

            if args.resample_spacing > 0:
                path_to_draw = resample_polyline(raw_path, spacing=args.resample_spacing)
                print(
                    f"Resampled path to {len(path_to_draw)} points "
                    f"at {args.resample_spacing:.2f} m spacing."
                )

                spacing_label = str(args.resample_spacing).replace(".", "_")
                resampled_output = track_data_dir / f"centreline_resampled_{spacing_label}m.json"
                export_path(
                    path_to_draw,
                    resampled_output,
                    spacing=args.resample_spacing,
                )
                print(f"Exported resampled path to: {resampled_output}")

            print(f"Drawing {len(path_to_draw)} path points in-sim...")
            drawer = DebugPathDrawer(bng)

            # Centreline — cyan
            drawer.draw_path(
                path_to_draw,
                sphere_every=args.sphere_every,
                label_every=args.label_every,
                line_color=(0.1, 0.8, 1.0, 1.0),
                sphere_color=(1.0, 0.45, 0.1, 1.0),
                cling=False,
                offset=0.5,
            )
            print("Centreline drawn.")

            # Track limits — left boundary (red) and right boundary (green)
            if not args.no_limits:
                spacing_label = str(args.resample_spacing).replace(".", "_")
                resampled_json = track_data_dir / f"centreline_resampled_{spacing_label}m.json"
                # Prefer the just-written resampled file; fall back to the default 2_0m one
                if not resampled_json.exists():
                    resampled_json = track_data_dir / "centreline_resampled_2_0m.json"
                if resampled_json.exists():
                    cl = TrackCentreline.from_json(resampled_json)
                    _left, _right, _hw, _normals = compute_track_boundaries(
                        cl, track_data_dir / ITEMS_FILE_NAME, HIGHLIGHT_IDS
                    )
                    left_pts = [(float(p[0]), float(p[1]), float(p[2])) for p in _left]
                    right_pts = [(float(p[0]), float(p[1]), float(p[2])) for p in _right]
                    # Close the loops
                    left_pts.append(left_pts[0])
                    right_pts.append(right_pts[0])
                    drawer.draw_path(
                        left_pts,
                        line_color=(1.0, 0.15, 0.15, 1.0),
                        sphere_every=0,
                        start_end_markers=False,
                        cling=False,
                        offset=0.5,
                    )
                    drawer.draw_path(
                        right_pts,
                        line_color=(0.15, 1.0, 0.15, 1.0),
                        sphere_every=0,
                        start_end_markers=False,
                        cling=False,
                        offset=0.5,
                    )
                    print("Track limits drawn (red=left, green=right).")
                else:
                    print(f"Track limits: skipped (resampled centreline not found at {resampled_json}).")

            # Physics racing line — gold/yellow
            if not args.no_raceline:
                raceline_json = track_data_dir / "physics_raceline.json"
                if raceline_json.exists():
                    rl_data = __import__("json").loads(raceline_json.read_text(encoding="utf-8"))
                    rl_pts = [
                        (float(p["x"]), float(p["y"]), float(p["z"]))
                        for p in rl_data["points"]
                    ]
                    rl_pts.append(rl_pts[0])  # close the loop
                    drawer.draw_path(
                        rl_pts,
                        line_color=(1.0, 0.85, 0.0, 1.0),
                        sphere_color=(1.0, 0.85, 0.0, 1.0),
                        sphere_every=args.sphere_every,
                        start_end_markers=False,
                        cling=False,
                        offset=0.5,
                    )
                    print("Physics racing line drawn (yellow).")
                else:
                    print(f"Racing line: skipped ({raceline_json} not found).")

        input("Press Enter to exit...")

    finally:
        if drawer is not None:
            drawer.clear()
        bng.disconnect()


if __name__ == "__main__":
    main()

r"""
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

    default 2m resampling:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path

    Without resampling:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --resample-spacing 0

    With denser debug spheres:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --sphere-every 10
"""
