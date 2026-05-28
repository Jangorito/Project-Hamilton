"""BeamNG UI-app bridge for live training telemetry."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping


APP_NAME = "beamngRlTrainingHud"
APP_SOURCE_DIR = Path(__file__).resolve().parents[3] / "beamng_ui" / "BeamNGRLTrainingHUD"
APP_DEST_RELATIVE = Path("ui") / "modules" / "apps" / "BeamNGRLTrainingHUD"

DEFAULT_PLACEMENT = {
    "left": "16px",
    "top": "16px",
    "width": "430px",
    "height": "330px",
    "min-width": "360px",
    "min-height": "280px",
}


class BeamNGTrainingHud:
    """Installs a BeamNG UI app and pushes telemetry to it via guihooks."""

    def __init__(self, beamng: Any, *, app_source_dir: Path | None = None) -> None:
        self.beamng = beamng
        self.app_source_dir = app_source_dir or APP_SOURCE_DIR
        self.enabled = beamng is not None
        self._layout_loaded = False
        self._failure_count = 0

    def install(self) -> Path | None:
        """Copy the HUD app into BeamNG's current user folder."""

        if not self.enabled:
            return None

        user_with_version = getattr(self.beamng, "user_with_version", None)
        if not user_with_version:
            self.enabled = False
            return None

        destination = Path(user_with_version) / APP_DEST_RELATIVE
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.copytree(self.app_source_dir, destination)
        except OSError as exc:
            print(f"WARNING: could not install BeamNG training HUD: {exc!r}")
            self.enabled = False
            return None

        return destination

    def show_layout(self) -> None:
        """Load a minimal app layout containing just the training HUD."""

        if not self.enabled:
            return

        try:
            show_hud = getattr(getattr(self.beamng, "ui", None), "show_hud", None)
            if callable(show_hud):
                show_hud()
        except Exception:
            pass

        layout = {
            "type": "beamngrl_training",
            "title": "Project Hamilton Training",
            "apps": [
                {
                    "appName": APP_NAME,
                    "placement": DEFAULT_PLACEMENT,
                    "settings": {"noCockpit": False},
                }
            ],
        }
        layout_lua_json = _lua_string(json.dumps(layout, separators=(",", ":")))
        self._queue_lua(
            "\n".join(
                [
                    "if ui_apps and ui_apps.requestUIAppsData then ui_apps.requestUIAppsData() end",
                    "if guihooks then guihooks.trigger('ShowApps', true) end",
                    (
                        "if guihooks then guihooks.trigger('appContainer:loadLayoutByObject', "
                        f"jsonDecode({layout_lua_json})) end"
                    ),
                ]
            )
        )
        self._layout_loaded = True

    def send(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if not self._layout_loaded:
            self.show_layout()

        try:
            payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            payload_json = json.dumps(_json_safe(payload), separators=(",", ":"))

        self._queue_lua(
            (
                "local data = jsonDecode("
                f"{_lua_string(payload_json)}"
                "); if guihooks then guihooks.trigger('BeamNGRLTrainingHUDData', data) end"
            )
        )

    def send_status(self, status: str, *, active: bool = False) -> None:
        self.send({"has_data": True, "status": status, "active": active})

    def _queue_lua(self, chunk: str) -> None:
        try:
            self.beamng.queue_lua_command(chunk)
            self._failure_count = 0
        except Exception as exc:
            self._failure_count += 1
            if self._failure_count == 1:
                print(f"WARNING: BeamNG training HUD update failed: {exc!r}")
            if self._failure_count >= 5:
                self.enabled = False


def _lua_string(value: str) -> str:
    return json.dumps(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return 0.0
        return value
    try:
        result = float(value)
    except (TypeError, ValueError):
        return str(value)
    return result if math.isfinite(result) else 0.0
