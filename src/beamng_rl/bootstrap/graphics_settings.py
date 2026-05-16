from __future__ import annotations

import configparser
from pathlib import Path
from typing import Iterable, List, Tuple, Union, cast

from beamngpy import BeamNGpy


SettingValue = Union[bool, int, float, str]
GraphicsSetting = Tuple[str, SettingValue]

SETTINGS_SECTION = "settings"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOW_GRAPHICS_SETTINGS_FILE = REPO_ROOT / "config" / "beamng_low_graphics.ini"


def parse_setting_value(raw_value: str) -> SettingValue:
    """
    Convert an INI value into the type BeamNG expects.

    BeamNG's Lua settings layer expects real booleans for boolean settings:
    passing the string "false" can behave like true in Lua because non-empty
    strings are truthy. Quoted values are kept as strings for settings such as
    GraphicDisplayResolutions = "800 600" or GraphicDisableShadows = "2".
    """
    value = raw_value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_graphics_settings(settings_file: Path) -> List[GraphicsSetting]:
    """Load ordered BeamNG graphics settings from a commented INI file."""
    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";"),
        interpolation=None,
    )
    parser.optionxform = str

    files_read = parser.read(settings_file, encoding="utf-8")
    if not files_read:
        raise FileNotFoundError(f"Graphics settings file not found: {settings_file}")
    if not parser.has_section(SETTINGS_SECTION):
        raise ValueError(
            f"Graphics settings file {settings_file} must contain a "
            f"[{SETTINGS_SECTION}] section."
        )

    return [
        (key, parse_setting_value(value))
        for key, value in parser.items(SETTINGS_SECTION)
    ]


def apply_graphics_settings(
    bng: BeamNGpy,
    settings: Iterable[GraphicsSetting],
    *,
    label: str,
) -> bool:
    """
    Apply BeamNG graphics settings and refresh graphics once at the end.

    Individual failures are warnings rather than fatal errors so a bad tuning
    key does not crash a long training/bootstrap session.
    """
    applied_count = 0
    failed_count = 0

    for key, value in settings:
        try:
            # BeamNGpy's type hint says value is str, but its protocol sends JSON
            # through to BeamNG Lua. Real bools/numbers are important there.
            bng.settings.change(key, cast(str, value))
        except Exception as exc:
            failed_count += 1
            print(f"WARNING: Could not apply BeamNG setting {key}={value!r}: {exc!r}")
        else:
            applied_count += 1
            print(f"Applied BeamNG setting {key}={value!r}.")

    if applied_count == 0:
        print(f"WARNING: No BeamNG graphics settings were applied for {label}.")
        return False

    try:
        bng.settings.apply_graphics()
    except Exception as exc:
        print(f"WARNING: Could not apply BeamNG graphics refresh for {label}: {exc!r}")
        return False

    print(
        f"Applied {applied_count} BeamNG setting(s) for {label}; "
        f"{failed_count} failed."
    )
    return failed_count == 0
