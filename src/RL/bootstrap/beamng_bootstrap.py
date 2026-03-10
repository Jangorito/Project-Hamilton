from __future__ import annotations

import argparse
import json
from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

from RL.track.geometry import resample_polyline
from RL.track.raceline_loader import derive_race_path_from_files
from RL.visualisation.debug_draw_path import DebugPathDrawer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEAMNG_HOME = Path(r"C:\Users\Jango\BeamNG.tech.v0.38.3.0")
RACE_FILE = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\race.race.json")
ITEMS_FILE = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\items.level.json")
TRACK_DATA_DIR = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track")

HOST = "localhost"
PORT = 64256

SPAWN_POS = (-406.8015137, 257.4653625, 25.0089035)
SPAWN_ROT = (-0.0001524259429, 0.0005056571858, -0.2886135897, 0.9574455164)

HIGHLIGHT_IDS = [30, 12, 10, 17]


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------
Float3 = tuple[float, float, float]


def derive_path() -> list[Float3]:
    return derive_race_path_from_files(RACE_FILE, ITEMS_FILE, HIGHLIGHT_IDS)


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


def main() -> None:
    args = build_arg_parser().parse_args()

    set_up_simple_logging()

    bng = BeamNGpy(HOST, PORT, home=str(BEAMNG_HOME))
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
            print("Deriving race path...")
            raw_path = derive_path()
            print(f"Derived raw path with {len(raw_path)} points.")

            raw_output = TRACK_DATA_DIR / "centreline_raw.json"
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
                resampled_output = TRACK_DATA_DIR / f"centreline_resampled_{spacing_label}m.json"
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