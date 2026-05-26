from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project-specific BeamNG setup shared by the manual bootstrap and Gym env.
# ---------------------------------------------------------------------------
# beamng_bootstrap.py remains the manual/debug entry point. This module keeps
# the side-effect-free setup assumptions reusable so live RL episodes start in
# the same Hirochi/SBR scenario that the bootstrap script has already proven.
BEAMNG_HOME_ENV_VAR = "BEAMNG_HOME"
STUDENT_BEAMNG_HOME = Path(r"C:\Users\Student\BeamNG")
JANGO_BEAMNG_HOME = Path(r"C:\Users\Jango\BeamNG.tech.v0.38.3.0")
BEAMNG_HOME_CANDIDATES = (STUDENT_BEAMNG_HOME, JANGO_BEAMNG_HOME)

HOST = "localhost"
PORT = 25252  # below Windows dynamic port range (49152-65535) to avoid OS conflicts

SPAWN_POS = (-406.8015137, 257.4653625, 25.0089035)
SPAWN_ROT = (-0.0001524259429, 0.0005056571858, -0.2886135897, 0.9574455164)

SHADOWS_DISABLE_SETTING_KEY = "GraphicDisableShadows"
SHADOWS_DISABLE_ALL = "2"


def resolve_beamng_home_from_path_or_env(beamng_home: str | Path | None = None) -> Path:
    """Resolve BeamNG.tech home from an explicit path, env var, or local defaults."""

    resolved_home = Path(beamng_home) if beamng_home is not None else None

    if resolved_home is None:
        env_beamng_home = os.environ.get(BEAMNG_HOME_ENV_VAR)
        if env_beamng_home:
            resolved_home = Path(env_beamng_home)
        else:
            resolved_home = next(
                (candidate for candidate in BEAMNG_HOME_CANDIDATES if candidate.is_dir()),
                STUDENT_BEAMNG_HOME,
            )

    resolved_home = resolved_home.expanduser()

    if not resolved_home.exists():
        raise FileNotFoundError(
            f"BeamNG home path does not exist: {resolved_home}. "
            f"Set it with --beamng-home or the {BEAMNG_HOME_ENV_VAR} environment variable."
        )
    if not resolved_home.is_dir():
        raise NotADirectoryError(f"BeamNG home path is not a directory: {resolved_home}")

    return resolved_home


def build_hirochi_sbr_scenario(
    bng: Any,
    *,
    vehicle_id: str = "ego_vehicle",
    scenario_instance_name: str = "sbr4_bootstrap",
) -> tuple[Any, Any]:
    """Build the known-good Hirochi Raceway SBR track scenario."""

    # BeamNGpy is imported lazily so importing the Gym env in mock mode never
    # requires a BeamNG.tech installation or the beamngpy package.
    from beamngpy import Scenario, Vehicle

    scenario = Scenario(
        "hirochi_raceway",
        scenario_instance_name,
        description="SBR4 Track bootstrap scenario",
    )

    vehicle = Vehicle(
        vehicle_id,
        model="sbr",
        part_config="vehicles/sbr/track.pc",
        license="JANGO",
        color="Blue",
    )

    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    scenario.make(bng)
    return scenario, vehicle


def is_student_beamng_home(beamng_home: Path) -> bool:
    """Return whether the selected install is the Student-machine BeamNG path."""

    return os.path.normcase(str(beamng_home.resolve(strict=False))) == os.path.normcase(
        str(STUDENT_BEAMNG_HOME.resolve(strict=False))
    )


def apply_shadow_disabling(bng: Any) -> bool:
    """
    Try the first API-based low-graphics setting.

    This intentionally changes only shadows for now. If this BeamNGpy setting
    call works reliably, we can later add a fuller low-graphics preset without
    editing BeamNG's default settings file directly.
    """

    try:
        bng.settings.change(SHADOWS_DISABLE_SETTING_KEY, SHADOWS_DISABLE_ALL)
        bng.settings.apply_graphics()
    except Exception as exc:
        print(
            "WARNING: Could not apply BeamNG shadow-disabling setting "
            f"{SHADOWS_DISABLE_SETTING_KEY}={SHADOWS_DISABLE_ALL}: {exc!r}"
        )
        return False

    print(
        "Applied BeamNG shadow-disabling setting "
        f"{SHADOWS_DISABLE_SETTING_KEY}={SHADOWS_DISABLE_ALL}."
    )
    return True


def apply_low_graphics_preset(bng: Any, settings_file: Path | None = None) -> bool:
    """
    Apply the editable low-graphics preset.

    The graphics-settings module imports BeamNGpy, so it is loaded only when a
    live BeamNG connection is already being configured.
    """

    try:
        from beamng_rl.bootstrap.graphics_settings import (
            DEFAULT_LOW_GRAPHICS_SETTINGS_FILE,
            apply_graphics_settings,
            load_graphics_settings,
        )
    except Exception as exc:
        print(f"WARNING: Could not import low-graphics helpers: {exc!r}")
        return False

    settings_path = settings_file or DEFAULT_LOW_GRAPHICS_SETTINGS_FILE

    try:
        settings = load_graphics_settings(settings_path)
    except Exception as exc:
        print(f"WARNING: Could not load low-graphics settings from {settings_path}: {exc!r}")
        return False

    print(f"Applying low-graphics settings from: {settings_path}")
    return apply_graphics_settings(bng, settings, label="low graphics")
