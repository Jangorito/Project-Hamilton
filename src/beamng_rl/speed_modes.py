"""Speed-mode compatibility helpers.

Historically the launcher exposed BeamNG deterministic timing as raw
``speed_factor`` values such as 16x and 32x. Training logs show those values are
requested timing targets, not achieved end-to-end PPO speedups. Keep the legacy
integer config key for old runs and scripts, but expose a smaller set of user
facing modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SpeedMode:
    key: str
    requested_factor: int
    label: str
    short_label: str
    subtitle: str
    run_suffix: str | None
    eta_multiplier: float
    note: str = ""


SPEED_MODES: tuple[SpeedMode, ...] = (
    SpeedMode(
        key="realtime",
        requested_factor=1,
        label="Realtime",
        short_label="1x",
        subtitle="real-time",
        run_suffix=None,
        eta_multiplier=1.0,
    ),
    SpeedMode(
        key="fast2",
        requested_factor=2,
        label="Fast 2x",
        short_label="2x",
        subtitle="target 2x faster than real-time",
        run_suffix="2x",
        eta_multiplier=2.0,
    ),
    SpeedMode(
        key="fast4",
        requested_factor=4,
        label="Fast 4x",
        short_label="4x",
        subtitle="target 4x faster than real-time",
        run_suffix="4x",
        eta_multiplier=4.0,
    ),
    SpeedMode(
        key="turbo",
        requested_factor=16,
        label="Turbo",
        short_label="Turbo",
        subtitle="max request; observed about 2-3x end-to-end training speed",
        run_suffix="turbo",
        eta_multiplier=3.0,
        note=(
            "Compatibility value speed_factor=16 requests BeamNG's fastest "
            "deterministic timing. Historical 16x runs achieved about 2-3x "
            "end-to-end PPO training speed in event logs."
        ),
    ),
)

SPEED_MODES_BY_KEY = {mode.key: mode for mode in SPEED_MODES}
SPEED_MODES_BY_FACTOR = {mode.requested_factor: mode for mode in SPEED_MODES}

CANONICAL_SPEED_FACTORS = tuple(mode.requested_factor for mode in SPEED_MODES)
LEGACY_SPEED_FACTORS = (1, 2, 4, 8, 16, 32)
LEGACY_TURBO_FACTORS = (8, 16, 32)
LEGACY_SPEED_FACTOR_LABEL = ", ".join(str(value) for value in LEGACY_SPEED_FACTORS)


def speed_mode_from_factor(value: Any, *, default_factor: int = 1) -> SpeedMode:
    """Return the user-facing speed mode for a legacy integer value.

    Legacy 8/16/32 values all mean "Turbo" now. The canonical persisted
    compatibility value for Turbo is 16.
    """

    try:
        speed_factor = int(value)
    except (TypeError, ValueError):
        speed_factor = int(default_factor)

    if speed_factor in LEGACY_TURBO_FACTORS:
        return SPEED_MODES_BY_KEY["turbo"]
    return SPEED_MODES_BY_FACTOR.get(speed_factor, SPEED_MODES_BY_FACTOR[default_factor])


def speed_mode_from_key(value: Any, *, default_factor: int = 1) -> SpeedMode:
    key = str(value or "").strip().lower()
    if key in SPEED_MODES_BY_KEY:
        return SPEED_MODES_BY_KEY[key]
    return speed_mode_from_factor(default_factor)


def speed_mode_from_config(
    config: Mapping[str, Any],
    *,
    default_factor: int = 1,
) -> SpeedMode:
    """Read either new ``speed_mode`` or legacy ``speed_factor`` config."""

    speed_mode = str(config.get("speed_mode", "")).strip().lower()
    if speed_mode in SPEED_MODES_BY_KEY:
        return SPEED_MODES_BY_KEY[speed_mode]
    return speed_mode_from_factor(config.get("speed_factor", default_factor))


def is_supported_speed_factor(value: Any) -> bool:
    try:
        return int(value) in LEGACY_SPEED_FACTORS
    except (TypeError, ValueError):
        return False


def speed_mode_payload(mode: SpeedMode) -> dict[str, Any]:
    return {
        "speed_mode": mode.key,
        "speed_factor": mode.requested_factor,
        "speed_label": mode.short_label,
        "speed_display_label": mode.label,
        "speed_subtitle": mode.subtitle,
        "speed_note": mode.note,
    }


def existing_speed_tokens() -> set[str]:
    tokens = {f"{factor}x" for factor in LEGACY_SPEED_FACTORS}
    tokens.update(mode.run_suffix for mode in SPEED_MODES if mode.run_suffix)
    return {token for token in tokens if token}
