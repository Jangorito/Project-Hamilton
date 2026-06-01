"""
Project Hamilton — Training Launcher
Flask web UI for launching, monitoring, and controlling training runs.

Run on the Jango machine (from repo root, with venv active):
    python scripts/launcher/app.py

Access from Student machine via SSH tunnel:
    ssh -L 5000:localhost:5000 jango@<JANGO_IP>
    Then open: http://localhost:5000
"""
from __future__ import annotations

import json
import os
import csv
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

LAUNCHER_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAUNCHER_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamng_rl.speed_modes import (
    LEGACY_SPEED_FACTOR_LABEL,
    existing_speed_tokens,
    is_supported_speed_factor,
    speed_mode_from_config,
    speed_mode_from_factor,
    speed_mode_payload,
)
from beamng_rl.training.run_manager import RunManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR = REPO_ROOT / "logs" / "remote"
PID_FILE = LOG_DIR / "training.pid"
LOG_LINK = LOG_DIR / "latest.log"
PROGRESS_FILE = LOG_DIR / "current_steps.txt"
STOP_SIGNAL_FILE = LOG_DIR / "stop_signal.txt"
WATCH_PID_FILE = LOG_DIR / "watch.pid"
WATCH_LOG_FILE = LOG_DIR / "watch_latest.log"
AI_RACELINE_PID = LOG_DIR / "ai_raceline.pid"
AI_RACELINE_LOG = LOG_DIR / "ai_raceline_latest.log"
WATCH_TASK_CONFIG = REPO_ROOT / "config" / "watch_task.json"
WATCH_TASK_NAME = "HamiltonWatch"
SESSION_CONFIG_PATH = REPO_ROOT / "config" / "launcher_session.json"

OBS_PORT = int(os.environ.get("OBS_PORT", 4455))
OBS_PASSWORD = os.environ.get("OBS_WEBSOCKET_PASSWORD", "")
BEAMNG_PROCESS_NAMES = ("BeamNG.tech.x64.exe", "BeamNG.x64.exe")
CAR_LABELS = {
    "sbr": "SBR4 Track",
    "etkc": "ETK K-Series Trackday A",
}
CAR_RUN_SUFFIXES = {
    "sbr": "subaru",
    "etkc": "etk",
}
TIMING_PROFILES = {"legacy_60hz", "det50ms"}
REWARD_CONFIG_KEYS = {"v1", "v2", "v21a", "v21b", "v22", "v22b"}
OBS_CONFIG_KEYS = {"v1", "v2"}
SPAWN_MODES = {"bootstrap", "random_checkpoint"}

PYTHON_EXE = REPO_ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path("python")

# ---------------------------------------------------------------------------
# Slot 2 — parallel training instance
# ---------------------------------------------------------------------------
SLOT2_PORT = 25253
SLOT2_BEAMNG_USER = Path(os.environ.get(
    "BEAMNG_USER_SLOT2",
    str(Path.home() / "AppData" / "Local" / "BeamNG_w2"),
))
SLOT2_SESSION_CONFIG = REPO_ROOT / "config" / "launcher_session_2.json"
SLOT2_PID_FILE       = LOG_DIR / "training_2.pid"
SLOT2_LOG_FILE       = LOG_DIR / "training_2.log"
SLOT2_PROGRESS_FILE  = LOG_DIR / "current_steps_2.txt"
SLOT2_STOP_SIGNAL    = LOG_DIR / "stop_signal_2.txt"

_slot2_proc: dict  = {"proc": None, "log_file": None, "run_name": None}
_pace2: dict       = {"mtime": 0.0, "steps": 0, "rate": None}

# OBS streaming status cache — avoids a 2 s TCP timeout on every /api/status poll.
_streaming_cache: dict = {"value": False, "checked_at": 0.0}
_STREAMING_CACHE_TTL = 15.0  # seconds between actual OBS connection attempts

# Fallback pace used before a real measurement is available.
_STEPS_PER_SECOND_FALLBACK = 27_000 / 7_200

# Pace tracker — updated by _read_live_steps() from consecutive progress file writes.
# Using file mtime as the timestamp means the rate reflects actual wall-clock training
# speed, so it automatically adjusts when steps_per_action or physics rate changes.
_pace: dict = {"mtime": 0.0, "steps": 0, "rate": None}
_watch_proc: dict = {"proc": None, "log_file": None}
_ai_raceline_proc: dict = {"proc": None, "log_file": None}

# Guard: refuse to set max_damage below this to protect against fat-finger triggers
_MIN_DAMAGE_THRESHOLD = 50.0
_STEP_SUFFIX_RE = re.compile(r"_\d+(?:k|m)?steps$", re.IGNORECASE)

app = Flask(__name__)
rm = RunManager()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _is_beamng_running() -> bool:
    for process_name in BEAMNG_PROCESS_NAMES:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/NH"],
            capture_output=True, text=True,
        )
        if process_name.lower() in result.stdout.lower():
            return True
    return False


def _kill_beamng() -> None:
    for process_name in BEAMNG_PROCESS_NAMES:
        subprocess.run(
            ["taskkill", "/F", "/IM", process_name, "/T"],
            capture_output=True,
        )


def _run_schtask(task_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["schtasks", "/run", "/tn", task_name],
        capture_output=True,
        text=True,
    )


def _get_training_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    )
    return pid if str(pid) in result.stdout else None


def _is_training() -> bool:
    if _get_training_pid() is not None:
        return True
    # Fallback: scan for a python process running the training script on the
    # slot 1 session file specifically.  Checking for train_live_ppo_run alone
    # would also match slot 2, causing slot 1 to appear busy while slot 2 runs.
    result = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "commandline", "/format:csv"],
        capture_output=True, text=True,
    )
    slot2_session = str(SLOT2_SESSION_CONFIG).replace("\\", "/")
    for line in result.stdout.splitlines():
        if "train_live_ppo_run" in line and "launcher_session_2" not in line:
            return True
    return False


def _get_training_2_pid() -> int | None:
    if not SLOT2_PID_FILE.exists():
        return None
    try:
        pid = int(SLOT2_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    )
    return pid if str(pid) in result.stdout else None


def _is_training_2() -> bool:
    return _get_training_2_pid() is not None


def _read_live_steps_2() -> int | None:
    if not SLOT2_PROGRESS_FILE.exists():
        _pace2["mtime"] = 0.0
        _pace2["steps"] = 0
        _pace2["rate"] = None
        return None
    try:
        steps = int(SLOT2_PROGRESS_FILE.read_text(encoding="utf-8").strip())
        mtime = SLOT2_PROGRESS_FILE.stat().st_mtime
    except (ValueError, OSError):
        return None
    if _pace2["mtime"] > 0 and mtime > _pace2["mtime"]:
        dt = mtime - _pace2["mtime"]
        ds = steps - _pace2["steps"]
        if dt > 0 and ds > 0:
            _pace2["rate"] = ds / dt
    if mtime != _pace2["mtime"]:
        _pace2["mtime"] = mtime
        _pace2["steps"] = steps
    return steps


def _get_active_run_info_2() -> dict:
    run_name = _slot2_proc.get("run_name")
    session = _load_slot2_session()
    total = _safe_timesteps(session.get("total_timesteps"), 100_000)
    if not run_name:
        return {"run_name": None, "current_steps": 0, "total_steps": total}
    current_steps = _read_live_steps_2()
    if current_steps is None:
        ckpt = rm.find_latest_checkpoint(run_name) if rm.exists(run_name) else None
        current_steps = 0
        if ckpt:
            match = re.search(r"_(\d+)_steps", ckpt.stem)
            if match:
                current_steps = int(match.group(1))
    if rm.exists(run_name):
        config = rm.load_config(run_name)
        total = _safe_timesteps(
            session.get("total_timesteps",
                        config.get("training", {}).get("total_timesteps", 100_000)),
            100_000,
        )
    return {"run_name": run_name, "current_steps": current_steps, "total_steps": total}


def _load_slot2_session() -> dict:
    if not SLOT2_SESSION_CONFIG.exists():
        return {}
    try:
        return json.loads(SLOT2_SESSION_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_slot2_session(config: dict) -> None:
    SLOT2_SESSION_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SLOT2_SESSION_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _is_streaming() -> bool:
    import time
    now = time.monotonic()
    if now - _streaming_cache["checked_at"] < _STREAMING_CACHE_TTL:
        return _streaming_cache["value"]
    try:
        import obsws_python as obs # type: ignore
        cl = obs.ReqClient(host="localhost", port=OBS_PORT, password=_get_obs_password(), timeout=2)
        result = bool(cl.get_stream_status().output_active)
    except Exception:
        result = False
    _streaming_cache["value"] = result
    _streaming_cache["checked_at"] = now
    return result


def _read_live_steps() -> int | None:
    """Read current timestep from StepProgressWriter and update the pace tracker."""
    if not PROGRESS_FILE.exists():
        _pace["mtime"] = 0.0
        _pace["steps"] = 0
        _pace["rate"] = None
        return None
    try:
        steps = int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
        mtime = PROGRESS_FILE.stat().st_mtime
    except (ValueError, OSError):
        return None
    if _pace["mtime"] > 0 and mtime > _pace["mtime"]:
        dt = mtime - _pace["mtime"]
        ds = steps - _pace["steps"]
        if dt > 0 and ds > 0:
            _pace["rate"] = ds / dt
    if mtime != _pace["mtime"]:
        _pace["mtime"] = mtime
        _pace["steps"] = steps
    return steps


def _get_active_run_info() -> dict:
    run_name = rm.find_latest_run()
    session = _load_session_config()
    if not run_name:
        return {
            "run_name": None,
            "current_steps": 0,
            "total_steps": _safe_timesteps(session.get("total_timesteps"), 100_000),
        }
    # Prefer the live progress file (updated every rollout by StepProgressWriter).
    # Fall back to the latest checkpoint name when no live file exists.
    current_steps = _read_live_steps()
    if current_steps is None:
        ckpt = rm.find_latest_checkpoint(run_name)
        current_steps = 0
        if ckpt:
            match = re.search(r"_(\d+)_steps", ckpt.stem)
            if match:
                current_steps = int(match.group(1))
    config = rm.load_config(run_name)
    configured_total = session.get(
        "total_timesteps",
        config.get("training", {}).get("total_timesteps", 100_000),
    )
    total_steps = _safe_timesteps(configured_total, 100_000)
    return {"run_name": run_name, "current_steps": current_steps, "total_steps": total_steps}


def _eta_seconds(current_steps: int, total_steps: int, speed_factor: int = 1) -> int:
    remaining = max(0, total_steps - current_steps)
    fallback_rate = _STEPS_PER_SECOND_FALLBACK * speed_mode_from_factor(speed_factor).eta_multiplier
    rate = _pace["rate"] if _pace["rate"] is not None else fallback_rate
    return int(remaining / rate)


def _load_session_config() -> dict:
    if not SESSION_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(SESSION_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_session_config(config: dict) -> None:
    SESSION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _parse_timesteps(value, default: int = 100_000) -> int:
    try:
        timesteps = int(value if value is not None else default)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timesteps must be a positive integer.") from exc
    if timesteps < 1:
        raise ValueError("Timesteps must be a positive integer.")
    return timesteps


def _safe_timesteps(value, default: int = 100_000) -> int:
    try:
        return _parse_timesteps(value, default)
    except ValueError:
        return default


def _format_step_suffix(total_timesteps: int) -> str:
    if total_timesteps % 1_000_000 == 0:
        return f"{total_timesteps // 1_000_000}msteps"
    if total_timesteps % 1_000 == 0:
        return f"{total_timesteps // 1_000}ksteps"
    return f"{total_timesteps}steps"


def _ensure_step_suffix(run_name: str, total_timesteps: int) -> str:
    base = _STEP_SUFFIX_RE.sub("", run_name.strip())
    suffix = _format_step_suffix(total_timesteps)
    return f"{base}_{suffix}" if base else suffix


def _run_vehicle_model(run_name: str) -> str | None:
    config = rm.load_config(run_name)
    env = config.get("env", {})
    car = str(env.get("vehicle_model", "")).strip().lower()
    return car if car in CAR_LABELS else None


def _run_speed_factor(run_name: str) -> int:
    config = rm.load_config(run_name)
    env = config.get("env", {})
    return speed_mode_from_config(env).requested_factor


def _run_timing_profile(run_name: str) -> str:
    config = rm.load_config(run_name)
    env = config.get("env", {})
    timing_profile = str(env.get("timing_profile", "")).strip().lower()
    if timing_profile in TIMING_PROFILES:
        return timing_profile
    return "det50ms" if _run_speed_factor(run_name) > 1 else "legacy_60hz"


def _fresh_timing_profile(speed_factor: int) -> str:
    return "det50ms" if speed_factor > 1 else "legacy_60hz"


def _requested_timing_profile(run_name: str, speed_factor: int) -> str:
    run_timing_profile = _run_timing_profile(run_name)
    if run_timing_profile == "det50ms":
        return "det50ms"
    return _fresh_timing_profile(speed_factor)


def _run_details(run_name: str) -> dict:
    car = _run_vehicle_model(run_name)
    config = rm.load_config(run_name)
    env = config.get("env", {}) if isinstance(config.get("env"), dict) else {}
    speed_mode = speed_mode_from_config(env)
    speed_factor = speed_mode.requested_factor
    timing_profile = _run_timing_profile(run_name)
    return {
        "car": car,
        "car_label": CAR_LABELS.get(car, "Unknown vehicle"),
        "has_vehicle_metadata": car is not None,
        "speed_factor": speed_factor,
        **speed_mode_payload(speed_mode),
        "timing_profile": timing_profile,
    }


def _ensure_car_suffix(
    run_name: str,
    car: str,
    reward_config: str,
    speed_factor: int = 1,
    total_timesteps: int | None = None,
    spawn_mode: str = "bootstrap",
    obs_config: str = "v1",
) -> str:
    suffix = CAR_RUN_SUFFIXES[car]
    base = _STEP_SUFFIX_RE.sub("", run_name.strip() or f"{reward_config}_{suffix}")
    tokens = base.lower().replace("-", "_").split("_")
    if suffix not in tokens:
        base = f"{base}_{suffix}"
        tokens = base.lower().replace("-", "_").split("_")
    speed_mode = speed_mode_from_factor(speed_factor)
    speed_suffix = speed_mode.run_suffix
    if speed_suffix and not any(token in existing_speed_tokens() for token in tokens):
        base = f"{base}_{speed_suffix}"
        tokens = base.lower().replace("-", "_").split("_")
    if obs_config == "v2" and "obs2" not in tokens:
        base = f"{base}_obs2"
        tokens = base.lower().replace("-", "_").split("_")
    if spawn_mode == "random_checkpoint" and "respawn" not in tokens:
        base = f"{base}_respawn"
    if total_timesteps is not None:
        base = _ensure_step_suffix(base, total_timesteps)
    return base


def _can_resume_with_car(run_name: str, car: str) -> tuple[bool, str]:
    run_car = _run_vehicle_model(run_name)
    if run_car == car:
        return True, ""
    if run_car is None and car == "sbr":
        return True, ""
    reason = (
        f"it is tagged as {CAR_LABELS[run_car]}"
        if run_car
        else "it has no saved vehicle metadata"
    )
    return False, reason


def _can_resume_with_timing(run_name: str, timing_profile: str) -> tuple[bool, str]:
    run_timing_profile = _run_timing_profile(run_name)
    if run_timing_profile == timing_profile:
        return True, ""
    return False, f"it was created for {run_timing_profile} timing"


def _get_obs_password() -> str:
    if OBS_PASSWORD:
        return OBS_PASSWORD
    if os.name != "nt":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "OBS_WEBSOCKET_PASSWORD")
            return str(value)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — read
# ---------------------------------------------------------------------------

@app.route("/api/status")
def api_status():
    training = _is_training()
    run_info = _get_active_run_info()
    session = _load_session_config()
    car = str(session.get("car", "sbr")).strip().lower()
    speed_mode = speed_mode_from_config(session)
    speed_factor = speed_mode.requested_factor
    timing_profile = str(session.get("timing_profile", "")).strip().lower()
    if timing_profile not in TIMING_PROFILES:
        timing_profile = _fresh_timing_profile(speed_factor)
    eta = _eta_seconds(
        run_info["current_steps"],
        run_info["total_steps"],
        speed_factor,
    ) if training else None
    return jsonify({
        "is_training": training,
        "is_streaming": _is_streaming(),
        "is_beamng": _is_beamng_running(),
        "run_name": run_info["run_name"],
        "current_steps": run_info["current_steps"],
        "total_steps": run_info["total_steps"],
        "eta_seconds": eta,
        "steps_per_second": _pace["rate"],
        "max_damage": session.get("max_damage"),
        "prev_max_damage": session.get("prev_max_damage"),
        "speed_factor": speed_factor,
        **speed_mode_payload(speed_mode),
        "timing_profile": timing_profile,
        "reward_config": session.get("reward_config", "v1"),
        "obs_config": session.get("obs_config", "v1"),
        "spawn_mode": session.get("spawn_mode", "bootstrap"),
        "car": car,
        "car_label": CAR_LABELS.get(car, car),
    })


@app.route("/api/runs")
def api_runs():
    runs = rm.list_runs()
    return jsonify({
        "runs": runs,
        "details": {run_name: _run_details(run_name) for run_name in runs},
    })


@app.route("/api/log")
def api_log():
    log_path = _resolve_log_path()
    if not log_path:
        return jsonify({"lines": ["No log file found. Start training to see output here."]})
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-50:]
        return jsonify({"lines": lines})
    except OSError:
        return jsonify({"lines": ["Could not read log file."]})


def _resolve_log_path() -> Path | None:
    if LOG_LINK.exists():
        if LOG_LINK.is_symlink():
            target = LOG_LINK.resolve()
            if target.exists():
                return target
        try:
            content = LOG_LINK.read_text(encoding="utf-8").strip()
            candidate = Path(content)
            if candidate.exists():
                return candidate
        except OSError:
            pass
    logs = sorted(LOG_DIR.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


# ---------------------------------------------------------------------------
# Routes — actions
# ---------------------------------------------------------------------------

@app.route("/api/launch", methods=["POST"])
def api_launch():
    data = request.get_json(force=True)
    mode = data.get("mode", "training")
    car = str(data.get("car", "sbr")).strip().lower()
    if car not in CAR_LABELS:
        return jsonify({
            "ok": False,
            "message": f"Unknown car {car!r}. Choose one of: {', '.join(CAR_LABELS)}",
        }), 400
    run_name = str(data.get("run_name", "")).strip()
    fresh = bool(data.get("fresh", False))
    try:
        timesteps = _parse_timesteps(data.get("timesteps", 100_000))
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    stream = bool(data.get("stream", mode == "training"))
    max_damage = float(data.get("max_damage", 500.0))
    raw_speed_factor = data.get("speed_factor", 1)
    if not is_supported_speed_factor(raw_speed_factor):
        return jsonify({
            "ok": False,
            "message": (
                f"Unknown speed factor {raw_speed_factor!r}. Choose one of: "
                f"{LEGACY_SPEED_FACTOR_LABEL} (8/16/32 are legacy Turbo aliases)."
            ),
        }), 400
    speed_mode = speed_mode_from_factor(raw_speed_factor)
    speed_factor = speed_mode.requested_factor
    # Reward config key ("v1" or "v2"). Written into launcher_session.json so
    # train_live_ppo_run.py picks it up without needing a CLI flag from schtasks.
    reward_config = str(data.get("reward_config", "v1"))
    if reward_config not in REWARD_CONFIG_KEYS:
        return jsonify({
            "ok": False,
            "message": f"Unknown reward config {reward_config!r}. Choose one of: {sorted(REWARD_CONFIG_KEYS)}",
        }), 400
    spawn_mode = str(data.get("spawn_mode", "bootstrap")).strip().lower()
    if spawn_mode not in SPAWN_MODES:
        return jsonify({
            "ok": False,
            "message": f"Unknown spawn mode {spawn_mode!r}. Choose one of: {sorted(SPAWN_MODES)}",
        }), 400
    obs_config = str(data.get("obs_config", "v1")).strip().lower()
    if obs_config not in OBS_CONFIG_KEYS:
        return jsonify({
            "ok": False,
            "message": f"Unknown obs config {obs_config!r}. Choose one of: {sorted(OBS_CONFIG_KEYS)}",
        }), 400
    timing_profile = _fresh_timing_profile(speed_factor)
    debug_mode = bool(data.get("debug_mode", False))

    if mode == "stream_only":
        session = _load_session_config()
        session.update({
            "car": car,
            "car_label": CAR_LABELS[car],
            "run_name": run_name,
            "fresh": fresh,
            "total_timesteps": timesteps,
            "stream": stream,
            "max_damage": max_damage,
            "speed_factor": speed_factor,
            **speed_mode_payload(speed_mode),
            "timing_profile": timing_profile,
            "reward_config": reward_config,
            "obs_config": obs_config,
            "spawn_mode": spawn_mode,
        })
        _save_session_config(session)
        obs_ok, obs_message = _obs_action("start")
        status = 200 if obs_ok else 500
        return jsonify({"ok": obs_ok, "message": obs_message or "Stream started."}), status

    if mode == "beamng_only":
        subprocess.run(["schtasks", "/run", "/tn", "BeamNGDirect"], capture_output=True)
        return jsonify({"ok": True, "message": "BeamNG launched standalone."})

    if fresh:
        run_name = _ensure_car_suffix(run_name, car, reward_config, speed_factor, timesteps, spawn_mode, obs_config)
    else:
        resume_name = run_name or rm.find_latest_run()
        if not resume_name:
            return jsonify({
                "ok": False,
                "message": "No run selected to resume. Start a fresh run.",
            }), 400
        can_resume, reason = _can_resume_with_car(resume_name, car)
        if not can_resume and not debug_mode:
            return jsonify({
                "ok": False,
                "message": (
                    f"Cannot resume '{resume_name}' as {CAR_LABELS[car]} because {reason}. "
                    "Start a fresh run for this car."
                ),
            }), 409
        timing_profile = _requested_timing_profile(resume_name, speed_factor)
        can_resume, reason = _can_resume_with_timing(resume_name, timing_profile)
        if not can_resume and not debug_mode:
            return jsonify({
                "ok": False,
                "message": (
                    f"Cannot resume '{resume_name}' with {timing_profile} because {reason}. "
                    "Start a fresh run for this timing mode."
                ),
            }), 409
        run_name = resume_name

    session = _load_session_config()
    session.update({
        "car": car,
        "car_label": CAR_LABELS[car],
        "run_name": run_name,
        "fresh": fresh,
        "total_timesteps": timesteps,
        "stream": stream,
        "max_damage": max_damage,
        "speed_factor": speed_factor,
        **speed_mode_payload(speed_mode),
        "timing_profile": timing_profile,
        "reward_config": reward_config,
        "obs_config": obs_config,
        "spawn_mode": spawn_mode,
        "debug_mode": debug_mode,
    })
    _save_session_config(session)

    _kill_beamng()

    obs_ok = True
    obs_message = ""
    if stream:
        obs_ok, obs_message = _obs_action("start")

    result = subprocess.run(
        ["schtasks", "/run", "/tn", "HamiltonTraining"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return jsonify({"ok": False, "message": f"schtasks failed: {result.stderr.strip()}"}), 500
    spawn_label = " · respawn" if spawn_mode == "random_checkpoint" else ""
    stream_label = " with OBS stream" if stream else ""
    obs_warning = "" if obs_ok else f" OBS warning: {obs_message}"
    return jsonify({
        "ok": True,
        "message": (
            f"Training started: {CAR_LABELS[car]}, {timesteps:,} steps, "
            f"{speed_mode.short_label} speed, reward {reward_config}{spawn_label}{stream_label}.{obs_warning}"
        ),
    })


@app.route("/api/stop/training", methods=["POST"])
def api_stop_training():
    if not _is_training():
        return jsonify({"ok": False, "message": "No training process found."})
    STOP_SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_SIGNAL_FILE.write_text("stop", encoding="utf-8")
    return jsonify({
        "ok": True,
        "message": "Stop signal sent — training will save model and run evaluation after the current rollout.",
    })



@app.route("/api/stop/stream", methods=["POST"])
def api_stop_stream():
    obs_ok, obs_message = _obs_action("stop")
    _invalidate_streaming_cache()
    return jsonify({"ok": obs_ok, "message": obs_message or "Stream stopped."})


@app.route("/api/start/stream", methods=["POST"])
def api_start_stream():
    obs_ok, obs_message = _obs_action("start")
    _invalidate_streaming_cache()
    return jsonify({"ok": obs_ok, "message": obs_message or "Stream started."})


@app.route("/api/wake", methods=["POST"])
def api_wake():
    subprocess.run(["schtasks", "/run", "/tn", "WakeScreen2"], capture_output=True)
    return jsonify({"ok": True, "message": "Wake signal sent."})


@app.route("/api/damage/set", methods=["POST"])
def api_damage_set():
    """Override max_damage in the session config. Guards against very low values."""
    data = request.get_json(force=True)
    new_damage = float(data.get("max_damage", 500.0))

    if new_damage < _MIN_DAMAGE_THRESHOLD:
        return jsonify({
            "ok": False,
            "message": f"max_damage {new_damage:.0f} is below the minimum allowed "
                       f"({_MIN_DAMAGE_THRESHOLD:.0f}). Refusing to set.",
        }), 400

    session = _load_session_config()
    prev = session.get("max_damage", 500.0)
    session["prev_max_damage"] = prev
    session["max_damage"] = new_damage
    _save_session_config(session)
    return jsonify({
        "ok": True,
        "message": f"max_damage set to {new_damage:.0f} (was {prev:.0f}).",
        "max_damage": new_damage,
        "prev_max_damage": prev,
    })


@app.route("/api/damage/restore", methods=["POST"])
def api_damage_restore():
    """Restore the previous max_damage value."""
    session = _load_session_config()
    prev = session.get("prev_max_damage")
    if prev is None:
        return jsonify({"ok": False, "message": "No previous value to restore."}), 400
    current = session.get("max_damage", 500.0)
    session["max_damage"] = prev
    session["prev_max_damage"] = current
    _save_session_config(session)
    return jsonify({
        "ok": True,
        "message": f"max_damage restored to {prev:.0f} (was {current:.0f}).",
        "max_damage": prev,
        "prev_max_damage": current,
    })


def _invalidate_streaming_cache() -> None:
    _streaming_cache["checked_at"] = 0.0


def _obs_action(action: str) -> tuple[bool, str]:
    obs_password = _get_obs_password()
    args = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "env" / "obs_control.py"),
        "--port", str(OBS_PORT),
        action,
    ]
    if obs_password:
        args += ["--password", obs_password]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        return False, str(exc)
    message = (result.stdout or result.stderr).strip()
    return result.returncode == 0, message


# ---------------------------------------------------------------------------
# Eval / watch helpers
# ---------------------------------------------------------------------------

def _best_checkpoint_from_eval_csv(run_name: str) -> Path | None:
    csv_path = rm.log_dir(run_name) / "checkpoints_eval.csv"
    if not csv_path.exists():
        return None
    best_ckpt: str | None = None
    best_progress = -1.0
    try:
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    p = float(row["progress_m"])
                except (KeyError, ValueError):
                    continue
                if p > best_progress:
                    best_progress = p
                    best_ckpt = row.get("checkpoint")
    except OSError:
        return None
    return Path(best_ckpt) if best_ckpt and Path(best_ckpt).is_file() else None


def _checkpoint_label(path: Path) -> str:
    m = re.search(r"_(\d+)_steps", path.stem)
    if m:
        return f"{int(m.group(1)):,} steps"
    if path.stem == "model_final":
        return "Final model"
    return path.stem


def _get_watch_pid() -> int | None:
    if not WATCH_PID_FILE.exists():
        return None
    try:
        pid = int(WATCH_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True,
    )
    return pid if str(pid) in result.stdout else None


def _is_watching() -> bool:
    return _get_watch_pid() is not None


def _process_running(pid: int) -> bool:
    """Check if a process is still running by PID."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=2,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Routes — eval & watch
# ---------------------------------------------------------------------------

@app.route("/api/eval/checkpoints/<run_name>")
def api_eval_checkpoints(run_name):
    if not rm.exists(run_name):
        return jsonify({"ok": False, "message": f"Run '{run_name}' not found."}), 404
    checkpoints = rm.find_all_checkpoints(run_name)
    best = _best_checkpoint_from_eval_csv(run_name) or rm.find_latest_checkpoint(run_name)
    ckpt_list = [
        {"path": str(c), "name": c.stem, "label": _checkpoint_label(c)}
        for c in checkpoints
    ]
    return jsonify({
        "checkpoints": ckpt_list,
        "best": str(best) if best else None,
    })


@app.route("/api/watch", methods=["POST"])
def api_watch():
    if _is_training():
        return jsonify({
            "ok": False,
            "message": "Cannot watch while training is running — stop training first.",
        }), 409
    if _is_watching():
        return jsonify({
            "ok": False,
            "message": "A watch session is already active — stop it first.",
        }), 409
    data = request.get_json(force=True)
    run_name = str(data.get("run_name", "")).strip()
    checkpoint = data.get("checkpoint")
    steps = max(1, int(data.get("steps", 500)))
    watch_speed = data.get("watch_speed")  # 0 = use training speed, 1+ = override

    if checkpoint:
        model_path = Path(checkpoint)
        if not model_path.is_file():
            return jsonify({"ok": False, "message": f"Checkpoint not found: {checkpoint}"}), 400
    else:
        if not run_name:
            return jsonify({"ok": False, "message": "Provide a run name or explicit checkpoint."}), 400
        if not rm.exists(run_name):
            return jsonify({"ok": False, "message": f"Run '{run_name}' not found."}), 404
        model_path = _best_checkpoint_from_eval_csv(run_name) or rm.find_latest_checkpoint(run_name)
        if model_path is None:
            return jsonify({"ok": False, "message": f"No checkpoints found for run '{run_name}'."}), 404

    _kill_beamng()

    WATCH_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_f = WATCH_LOG_FILE.open("w", encoding="utf-8", buffering=1)
    WATCH_TASK_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    WATCH_TASK_CONFIG.write_text(
        json.dumps({
            "model_path": str(model_path),
            "run_name": run_name,
            "steps": steps,
            "speed_factor": int(watch_speed) if watch_speed and int(watch_speed) > 0 else None,
            "no_wait": True,
        }, indent=2),
        encoding="utf-8",
    )

    task_result = _run_schtask(WATCH_TASK_NAME)
    if task_result.returncode == 0:
        return jsonify({
            "ok": True,
            "message": (
                f"Watch scheduled in interactive desktop session: {Path(model_path).stem}, {steps} steps."
            ),
        })

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    proc = subprocess.Popen(
        [
            str(PYTHON_EXE),
            str(REPO_ROOT / "scripts" / "eval" / "watch_model.py"),
            str(model_path),
            "--steps", str(steps),
            "--no-wait",
            *( ["--run", run_name] if run_name else []),
            *(["--speed-factor", str(int(watch_speed))] if watch_speed and int(watch_speed) > 0 else []),
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    _watch_proc["proc"] = proc
    _watch_proc["log_file"] = log_f

    WATCH_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCH_PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    label = Path(model_path).stem
    return jsonify({"ok": True, "message": f"Watch started: {label}, {steps} steps."})


@app.route("/api/watch/stop", methods=["POST"])
def api_watch_stop():
    pid = _get_watch_pid()
    if pid is None:
        return jsonify({"ok": False, "message": "No watch session found."})
    subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True)
    try:
        WATCH_PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    lf = _watch_proc.get("log_file")
    if lf:
        try:
            lf.close()
        except Exception:
            pass
    _watch_proc["proc"] = None
    _watch_proc["log_file"] = None
    return jsonify({"ok": True, "message": "Watch session stopped. BeamNG stays open — kill it when you're done."})


@app.route("/api/watch/status")
def api_watch_status():
    return jsonify({"is_watching": _is_watching()})


@app.route("/api/watch/log")
def api_watch_log():
    if not WATCH_LOG_FILE.exists():
        return jsonify({"lines": ["No watch session active."]})
    try:
        text = WATCH_LOG_FILE.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-50:]
        return jsonify({"lines": lines or ["(empty)"]})
    except OSError:
        return jsonify({"lines": ["Could not read watch log."]})


@app.route("/api/stop/beamng", methods=["POST"])
def api_stop_beamng():
    _kill_beamng()
    return jsonify({"ok": True, "message": "BeamNG killed."})


@app.route("/api/collect_ai_raceline", methods=["POST"])
def api_collect_ai_raceline():
    """Launch AI raceline collection in BeamNG."""
    data = request.get_json(force=True)
    raw_vehicles = data.get("vehicles", data.get("vehicle", "etkc"))
    if isinstance(raw_vehicles, (list, tuple)):
        vehicles = [str(value).strip().lower() for value in raw_vehicles]
    else:
        vehicles = [str(raw_vehicles).strip().lower()]
    vehicles = [value for value in vehicles if value]
    if not vehicles:
        vehicles = ["etkc"]
    unknown_vehicles = [value for value in vehicles if value not in ("both", "etkc", "sbr")]
    if unknown_vehicles:
        return jsonify({"ok": False, "message": f"Unknown vehicle {unknown_vehicles[0]!r}."}), 400

    raw_aggressions = data.get("aggressions", data.get("aggression", 1.0))
    if isinstance(raw_aggressions, (list, tuple)):
        aggressions = [float(value) for value in raw_aggressions]
    else:
        aggressions = [float(raw_aggressions)]
    racer_skill = float(data.get("racer_skill", 1.0))
    speed_profile = str(data.get("speed_profile", "curvature")).strip().lower()
    max_line_speed = float(data.get("max_line_speed", 42.0))
    max_laps = max(1, int(data.get("max_laps", 2)))
    session_id = _current_windows_session_id()
    use_nogfx = bool(data.get("nogfx", False)) or session_id == 0
    
    _kill_beamng()
    
    AI_RACELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_f = AI_RACELINE_LOG.open("w", encoding="utf-8", buffering=1)
    
    # Set up environment with PYTHONPATH for venv context
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    
    cmd = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "eval" / "collect_ai_raceline.py"),
        "--vehicles", *vehicles,
        "--aggressions", *(str(value) for value in aggressions),
        "--racer-skill", str(racer_skill),
        "--speed-profile", speed_profile,
        "--max-line-speed", str(max_line_speed),
        "--max-laps", str(max_laps),
    ]
    if use_nogfx:
        cmd.append("--nogfx")

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    _ai_raceline_proc["proc"] = proc
    _ai_raceline_proc["log_file"] = log_f
    
    AI_RACELINE_PID.parent.mkdir(parents=True, exist_ok=True)
    AI_RACELINE_PID.write_text(str(proc.pid), encoding="utf-8")
    
    return jsonify({
        "ok": True,
        "message": (
            "AI raceline collection started: "
            f"{', '.join(vehicles)} @ {', '.join(str(value) for value in aggressions)}."
            + (" Running no-GFX because the launcher is in Session 0." if session_id == 0 else "")
        ),
    })


@app.route("/api/collect_ai_raceline/status")
def api_collect_ai_raceline_status():
    pid = None
    try:
        pid = int(AI_RACELINE_PID.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    is_running = pid is not None and _process_running(pid)
    return jsonify({"is_collecting": is_running})


@app.route("/api/collect_ai_raceline/log")
def api_collect_ai_raceline_log():
    if not AI_RACELINE_LOG.exists():
        return jsonify({"lines": ["No AI raceline collection active."]})
    try:
        text = AI_RACELINE_LOG.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-50:]
        return jsonify({"lines": lines or ["(empty)"]})
    except OSError:
        return jsonify({"lines": ["Could not read AI raceline log."]})


@app.route("/api/collect_ai_raceline/stop", methods=["POST"])
def api_collect_ai_raceline_stop():
    pid = None
    try:
        pid = int(AI_RACELINE_PID.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    if pid is None:
        return jsonify({"ok": False, "message": "No AI raceline collection active."})
    subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], capture_output=True)
    try:
        AI_RACELINE_PID.unlink(missing_ok=True)
    except OSError:
        pass
    lf = _ai_raceline_proc.get("log_file")
    if lf:
        try:
            lf.close()
        except Exception:
            pass
    _ai_raceline_proc["proc"] = None
    _ai_raceline_proc["log_file"] = None
    return jsonify({"ok": True, "message": "AI raceline collection stopped."})


# ---------------------------------------------------------------------------
# Routes — slot 2 (parallel training instance)
# ---------------------------------------------------------------------------

@app.route("/api/2/status")
def api_2_status():
    training = _is_training_2()
    run_info = _get_active_run_info_2()
    session = _load_slot2_session()
    car = str(session.get("car", "etkc")).strip().lower()
    speed_mode = speed_mode_from_config(session)
    speed_factor = speed_mode.requested_factor
    eta = _eta_seconds(
        run_info["current_steps"],
        run_info["total_steps"],
        speed_factor,
    ) if training else None
    return jsonify({
        "is_training": training,
        "run_name": run_info["run_name"],
        "current_steps": run_info["current_steps"],
        "total_steps": run_info["total_steps"],
        "eta_seconds": eta,
        "steps_per_second": _pace2["rate"],
        "max_damage": session.get("max_damage"),
        "prev_max_damage": session.get("prev_max_damage"),
        "speed_factor": speed_factor,
        **speed_mode_payload(speed_mode),
        "reward_config": session.get("reward_config", "v1"),
        "car": car,
        "car_label": CAR_LABELS.get(car, car),
    })


@app.route("/api/2/log")
def api_2_log():
    if not SLOT2_LOG_FILE.exists():
        return jsonify({"lines": ["No Run 2 log yet. Start training to see output here."]})
    try:
        text = SLOT2_LOG_FILE.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-50:]
        return jsonify({"lines": lines})
    except OSError:
        return jsonify({"lines": ["Could not read Run 2 log."]})


@app.route("/api/2/launch", methods=["POST"])
def api_2_launch():
    data = request.get_json(force=True)
    car = str(data.get("car", "etkc")).strip().lower()
    if car not in CAR_LABELS:
        return jsonify({"ok": False, "message": f"Unknown car {car!r}."}), 400
    run_name = str(data.get("run_name", "")).strip()
    fresh = bool(data.get("fresh", False))
    try:
        timesteps = _parse_timesteps(data.get("timesteps", 100_000))
    except ValueError as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400
    max_damage = float(data.get("max_damage", 500.0))
    raw_speed_factor = data.get("speed_factor", 16)
    if not is_supported_speed_factor(raw_speed_factor):
        return jsonify({
            "ok": False,
            "message": (
                f"Unknown speed factor {raw_speed_factor!r}. Choose one of: "
                f"{LEGACY_SPEED_FACTOR_LABEL} (8/16/32 are legacy Turbo aliases)."
            ),
        }), 400
    speed_mode = speed_mode_from_factor(raw_speed_factor)
    speed_factor = speed_mode.requested_factor
    reward_config = str(data.get("reward_config", "v1"))
    if reward_config not in REWARD_CONFIG_KEYS:
        return jsonify({"ok": False, "message": f"Unknown reward config {reward_config!r}."}), 400
    obs_config_2 = str(data.get("obs_config", "v1")).strip().lower()
    if obs_config_2 not in OBS_CONFIG_KEYS:
        return jsonify({"ok": False, "message": f"Unknown obs config {obs_config_2!r}."}), 400
    spawn_mode_2 = str(data.get("spawn_mode", "bootstrap")).strip().lower()
    if spawn_mode_2 not in SPAWN_MODES:
        return jsonify({"ok": False, "message": f"Unknown spawn mode {spawn_mode_2!r}."}), 400
    timing_profile = _fresh_timing_profile(speed_factor)
    debug_mode = bool(data.get("debug_mode", False))

    if fresh:
        run_name = _ensure_car_suffix(run_name, car, reward_config, speed_factor, timesteps, spawn_mode_2, obs_config_2)
    else:
        resume_name = run_name or rm.find_latest_run()
        if not resume_name:
            return jsonify({"ok": False, "message": "No run to resume. Start fresh."}), 400
        can_resume, reason = _can_resume_with_car(resume_name, car)
        if not can_resume and not debug_mode:
            return jsonify({"ok": False, "message": f"Cannot resume '{resume_name}': {reason}."}), 409
        timing_profile = _requested_timing_profile(resume_name, speed_factor)
        can_resume, reason = _can_resume_with_timing(resume_name, timing_profile)
        if not can_resume and not debug_mode:
            return jsonify({"ok": False, "message": f"Cannot resume '{resume_name}': {reason}."}), 409
        run_name = resume_name

    # Kill any existing slot 2 training process.
    pid2 = _get_training_2_pid()
    if pid2:
        subprocess.run(["taskkill", "/F", "/PID", str(pid2), "/T"], capture_output=True)

    session = {
        "car": car,
        "car_label": CAR_LABELS[car],
        "run_name": run_name,
        "fresh": fresh,
        "total_timesteps": timesteps,
        "max_damage": max_damage,
        "speed_factor": speed_factor,
        **speed_mode_payload(speed_mode),
        "timing_profile": timing_profile,
        "reward_config": reward_config,
        "obs_config": obs_config_2,
        "spawn_mode": spawn_mode_2,
        "debug_mode": debug_mode,
        "beamng_hud": True,
        "port": SLOT2_PORT,
        "beamng_user": str(SLOT2_BEAMNG_USER),
        "prev_max_damage": max_damage,
    }
    _save_slot2_session(session)

    SLOT2_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    lf = _slot2_proc.get("log_file")
    if lf:
        try:
            lf.close()
        except Exception:
            pass
    log_f = SLOT2_LOG_FILE.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        [
            str(PYTHON_EXE),
            str(REPO_ROOT / "scripts" / "train" / "train_live_ppo_run.py"),
            "--session-file",    str(SLOT2_SESSION_CONFIG),
            "--progress-file",   str(SLOT2_PROGRESS_FILE),
            "--stop-signal-file", str(SLOT2_STOP_SIGNAL),
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    _slot2_proc["proc"] = proc
    _slot2_proc["log_file"] = log_f
    _slot2_proc["run_name"] = run_name

    SLOT2_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    SLOT2_PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    return jsonify({
        "ok": True,
        "message": (
            f"Run 2 started: {CAR_LABELS[car]}, {timesteps:,} steps, "
            f"{speed_mode.short_label} speed, reward {reward_config}."
        ),
    })


@app.route("/api/2/stop/training", methods=["POST"])
def api_2_stop_training():
    if not _is_training_2():
        return jsonify({"ok": False, "message": "No Run 2 training process found."})
    SLOT2_STOP_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    SLOT2_STOP_SIGNAL.write_text("stop", encoding="utf-8")
    return jsonify({
        "ok": True,
        "message": "Stop signal sent to Run 2 — will save after current rollout.",
    })


@app.route("/api/2/damage/set", methods=["POST"])
def api_2_damage_set():
    data = request.get_json(force=True)
    new_damage = float(data.get("max_damage", 500.0))
    if new_damage < _MIN_DAMAGE_THRESHOLD:
        return jsonify({
            "ok": False,
            "message": f"max_damage {new_damage:.0f} is below minimum ({_MIN_DAMAGE_THRESHOLD:.0f}).",
        }), 400
    session = _load_slot2_session()
    prev = session.get("max_damage", 500.0)
    session["prev_max_damage"] = prev
    session["max_damage"] = new_damage
    _save_slot2_session(session)
    return jsonify({
        "ok": True,
        "message": f"Run 2 max_damage set to {new_damage:.0f} (was {prev:.0f}).",
        "max_damage": new_damage,
        "prev_max_damage": prev,
    })


@app.route("/api/2/damage/restore", methods=["POST"])
def api_2_damage_restore():
    session = _load_slot2_session()
    prev = session.get("prev_max_damage")
    if prev is None:
        return jsonify({"ok": False, "message": "No previous value to restore."}), 400
    current = session.get("max_damage", 500.0)
    session["max_damage"] = prev
    session["prev_max_damage"] = current
    _save_slot2_session(session)
    return jsonify({
        "ok": True,
        "message": f"Run 2 max_damage restored to {prev:.0f} (was {current:.0f}).",
        "max_damage": prev,
        "prev_max_damage": current,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _evict_stale_launchers() -> None:
    """Kill any other python process already running app.py so a redeployed launcher
    always claims port 5000 rather than silently losing the race to stale code."""
    current_pid = os.getpid()
    result = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'",
         "get", "processid,commandline", "/format:csv"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "app.py" not in line:
            continue
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1].strip())
        except ValueError:
            continue
        if pid == current_pid:
            continue
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        print(f"Evicted stale launcher (PID {pid})")


if __name__ == "__main__":
    _evict_stale_launchers()
    print("=" * 50)
    print("  Project Hamilton Launcher")
    print("  http://localhost:5000")
    print()
    print("  From Student machine:")
    print("  ssh -L 5000:localhost:5000 jango@<IP>")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
