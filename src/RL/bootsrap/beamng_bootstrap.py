from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEAMNG_HOME  = Path(r"C:\Users\Jango\BeamNG.tech.v0.38.3.0")
RACELINE_SCRIPT = Path(r"C:\Users\Jango\workspace\BeamNG\scripts\raceline_plot.py")
DEBUG_DRAW   = Path(r"C:\Users\Jango\workspace\BeamNG\src\RL\visualisation\debug_draw_path.py")
RACE_FILE    = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\race.race.json")
ITEMS_FILE   = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\items.level.json")

HOST = "localhost"
PORT = 64256

SPAWN_POS = (-406.8015137, 257.4653625, 25.0089035)
SPAWN_ROT = (-0.0001524259429, 0.0005056571858, -0.2886135897, 0.9574455164)

HIGHLIGHT_IDS = [30, 12, 10, 17]

# ---------------------------------------------------------------------------
# Import helpers from sibling scripts without installing them as packages
# ---------------------------------------------------------------------------

def _import_from(script_path: Path, module_name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def derive_path() -> list[tuple[float, float, float]]:
    """Load the raceline scripts and return the derived race path coordinates."""
    rl = _import_from(RACELINE_SCRIPT, "raceline_plot")

    data, pathnodes, segments, start_positions, node_by_id = rl.load_race_path(RACE_FILE)
    decal_roads = rl.load_items_lines(ITEMS_FILE)
    chain_slices = rl.build_sequential_chain(decal_roads, HIGHLIGHT_IDS)
    return rl.derive_race_path(decal_roads, HIGHLIGHT_IDS, chain_slices)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="BeamNG SBR4 bootstrap")
    parser.add_argument(
        "--draw-path",
        action="store_true",
        help="Derive and draw the race path in-sim using debug primitives.",
    )
    parser.add_argument(
        "--sphere-every", type=int, default=25,
        help="Draw a waypoint sphere every N path points (default: 25).",
    )
    parser.add_argument(
        "--label-every", type=int, default=None,
        help="Draw an index label every N path points (default: off).",
    )
    args = parser.parse_args()

    set_up_simple_logging()

    bng = BeamNGpy(HOST, PORT, home=str(BEAMNG_HOME))
    bng.open(launch=True)

    scenario, vehicle = build_scenario(bng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    print("Ready.")

    drawer = None
    if args.draw_path:
        debug_draw = _import_from(DEBUG_DRAW, "debug_draw_path")
        print("Deriving race path...")
        path_points = derive_path()
        print(f"Drawing {len(path_points)} path points in-sim...")
        drawer = debug_draw.DebugPathDrawer(bng)
        drawer.draw_path(
            path_points,
            sphere_every=args.sphere_every,
            label_every=args.label_every,
            cling=False,
            offset=0.5,          # float slightly above road surface
        )
        print("Path drawn.")

    input("Press Enter to exit...")

    if drawer is not None:
        drawer.clear()

    bng.disconnect()


if __name__ == "__main__":
    main()