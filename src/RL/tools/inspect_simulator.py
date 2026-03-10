from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beamngpy import BeamNGpy

OUTPUT_DIR = Path("sim_catalog")

_PRIMITIVES = (str, int, float, bool, type(None))


def _to_jsonable(value: Any, _seen: frozenset = frozenset(), _depth: int = 0) -> Any:
    """
    Recursively convert BeamNGpy objects / paths / tuples into JSON-friendly data.
    Guards against circular references and excessive recursion depth.
    """
    MAX_DEPTH = 20

    if _depth > MAX_DEPTH:
        return f"<max depth reached: {type(value).__name__}>"

    if isinstance(value, _PRIMITIVES):
        return value

    if isinstance(value, Path):
        return str(value)

    obj_id = id(value)
    if obj_id in _seen:
        return f"<circular ref: {type(value).__name__}>"
    _seen = _seen | {obj_id}  # immutable update — each branch gets its own copy

    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v, _seen, _depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v, _seen, _depth + 1) for v in value]

    if hasattr(value, "__dict__"):
        return {
            k: _to_jsonable(v, _seen, _depth + 1)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }

    # Fallback: try str(), otherwise note the type
    try:
        return str(value)
    except Exception:
        return f"<unserializable: {type(value).__name__}>"


def dump_simulator_catalog(
    host: str = "localhost",
    port: int = 64256,
    launch: bool = True,
    home: Path | str = r"C:\Users\Jango\BeamNG.tech.v0.38.3.0",
    output_dir: Path | str = OUTPUT_DIR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beamng = BeamNGpy(host, port, home=str(home))

    try:
        beamng.open(launch=launch)
        print("Connected to BeamNG.tech")

        vehicles = beamng.vehicles.get_available()
        vehicles_json = _to_jsonable(vehicles)

        levels, scenarios = beamng.scenario.get_levels_and_scenarios()
        levels_json = _to_jsonable(levels)
        scenarios_json = _to_jsonable(scenarios)

        (output_dir / "vehicles.json").write_text(
            json.dumps(vehicles_json, indent=2), encoding="utf-8"
        )
        (output_dir / "levels.json").write_text(
            json.dumps(levels_json, indent=2), encoding="utf-8"
        )
        (output_dir / "scenarios.json").write_text(
            json.dumps(scenarios_json, indent=2), encoding="utf-8"
        )

        summary = {
            "vehicle_models": len(vehicles_json),
            "levels": len(levels_json),
            "scenario_groups": len(scenarios_json),
            "files": [
                str(output_dir / "vehicles.json"),
                str(output_dir / "levels.json"),
                str(output_dir / "scenarios.json"),
            ],
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print("\nWrote:")
        for file_name in summary["files"]:
            print(f"  - {file_name}")
        print(f"  - {output_dir / 'summary.json'}")

    finally:
        beamng.close()
        print("\nClosed BeamNG connection.")


if __name__ == "__main__":
    dump_simulator_catalog()