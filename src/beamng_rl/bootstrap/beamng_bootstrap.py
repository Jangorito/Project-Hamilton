from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

from beamng_rl.track.geometry import resample_polyline
from beamng_rl.track.raceline_loader import derive_race_path_from_files
from beamng_rl.visualisation.debug_draw_path import DebugPathDrawer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEAMNG_HOME_ENV_VAR = "BEAMNG_HOME"
STUDENT_BEAMNG_HOME = Path(r"C:\Users\Student\BeamNG")
JANGO_BEAMNG_HOME = Path(r"C:\Users\Jango\BeamNG.tech.v0.38.3.0")
BEAMNG_HOME_CANDIDATES = (STUDENT_BEAMNG_HOME, JANGO_BEAMNG_HOME)

TRACK_DATA_DIR_ENV_VAR = "BEAMNG_TRACK_DATA_DIR"
STUDENT_TRACK_DATA_DIR = Path(r"C:\Users\Student\Workspace\Project-Hamilton\data\hirochi_track")
JANGO_TRACK_DATA_DIR = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track")
TRACK_DATA_DIR_CANDIDATES = (STUDENT_TRACK_DATA_DIR, JANGO_TRACK_DATA_DIR)
RACE_FILE_NAME = "race.race.json"
ITEMS_FILE_NAME = "items.level.json"

HOST = "localhost"
PORT = 64256

SPAWN_POS = (-406.8015137, 257.4653625, 25.0089035)
SPAWN_ROT = (-0.0001524259429, 0.0005056571858, -0.2886135897, 0.9574455164)

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
def build_scenario(bng: BeamNGpy) -> tuple[Scenario, Vehicle]:
    scenario = Scenario(
        "hirochi_raceway",
        "sbr4_bootstrap",
        description="SBR4 Track bootstrap scenario",
    )

    vehicle = Vehicle(
        "ego_vehicle",
        model="sbr",
        part_config="vehicles/sbr/track.pc",
        license="JANGO",
        color="Blue",
    )

    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)
    return scenario, vehicle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BeamNG SBR4 bootstrap")

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

    return parser


def resolve_beamng_home(args: argparse.Namespace) -> Path:
    beamng_home = args.beamng_home

    if beamng_home is None:
        env_beamng_home = os.environ.get(BEAMNG_HOME_ENV_VAR)
        if env_beamng_home:
            beamng_home = Path(env_beamng_home)
        else:
            beamng_home = next(
                (candidate for candidate in BEAMNG_HOME_CANDIDATES if candidate.is_dir()),
                STUDENT_BEAMNG_HOME,
            )

    beamng_home = beamng_home.expanduser()

    if not beamng_home.exists():
        raise FileNotFoundError(
            f"BeamNG home path does not exist: {beamng_home}. "
            f"Set it with --beamng-home or the {BEAMNG_HOME_ENV_VAR} environment variable."
        )
    if not beamng_home.is_dir():
        raise NotADirectoryError(f"BeamNG home path is not a directory: {beamng_home}")

    return beamng_home


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
            drawer.draw_path(
                path_to_draw,
                sphere_every=args.sphere_every,
                label_every=args.label_every,
                cling=False,
                offset=0.5,
            )
            print("Path drawn.")

        input("Press Enter to exit...")

    finally:
        if drawer is not None:
            drawer.clear()
        bng.disconnect()


if __name__ == "__main__":
    main()

"""
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

    default 2m resampling:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path

    Without resampling:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --resample-spacing 0

    With denser debug spheres:
python -m beamng_rl.bootstrap.beamng_bootstrap --draw-path --sphere-every 10
"""
