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

from beamng_rl.training.run_manager import RunManager

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR = REPO_ROOT / "logs" / "remote"
PID_FILE = LOG_DIR / "training.pid"
LOG_LINK = LOG_DIR / "latest.log"
PROGRESS_FILE = LOG_DIR / "current_steps.txt"
STOP_SIGNAL_FILE = LOG_DIR / "stop_signal.txt"
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
SPEED_FACTORS = {1, 2, 4, 8, 16, 32}
REWARD_CONFIG_KEYS = {"v1", "v2"}

PYTHON_EXE = REPO_ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path("python")

# Fallback pace used before a real measurement is available.
_STEPS_PER_SECOND_FALLBACK = 27_000 / 7_200

# Pace tracker — updated by _read_live_steps() from consecutive progress file writes.
# Using file mtime as the timestamp means the rate reflects actual wall-clock training
# speed, so it automatically adjusts when steps_per_action or physics rate changes.
_pace: dict = {"mtime": 0.0, "steps": 0, "rate": None}

# Guard: refuse to set max_damage below this to protect against fat-finger triggers
_MIN_DAMAGE_THRESHOLD = 50.0

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
    # Fallback: scan for a python process running the training script directly.
    # Catches runs started outside the GUI (e.g. schtasks without a PID file).
    result = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "commandline", "/format:csv"],
        capture_output=True, text=True,
    )
    return "train_live_ppo_run" in result.stdout


def _is_streaming() -> bool:
    try:
        import obsws_python as obs # type: ignore
        cl = obs.ReqClient(host="localhost", port=OBS_PORT, password=_get_obs_password(), timeout=2)
        return bool(cl.get_stream_status().output_active)
    except Exception:
        return False


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
            "total_steps": int(session.get("total_timesteps", 100_000)),
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
    total_steps = int(session.get(
        "total_timesteps",
        config.get("training", {}).get("total_timesteps", 100_000),
    ))
    return {"run_name": run_name, "current_steps": current_steps, "total_steps": total_steps}


def _eta_seconds(current_steps: int, total_steps: int, speed_factor: int = 1) -> int:
    remaining = max(0, total_steps - current_steps)
    fallback_rate = _STEPS_PER_SECOND_FALLBACK * max(1, int(speed_factor))
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


def _run_vehicle_model(run_name: str) -> str | None:
    config = rm.load_config(run_name)
    env = config.get("env", {})
    car = str(env.get("vehicle_model", "")).strip().lower()
    return car if car in CAR_LABELS else None


def _run_details(run_name: str) -> dict:
    car = _run_vehicle_model(run_name)
    return {
        "car": car,
        "car_label": CAR_LABELS.get(car, "Unknown vehicle"),
        "has_vehicle_metadata": car is not None,
    }


def _ensure_car_suffix(run_name: str, car: str, reward_config: str) -> str:
    suffix = CAR_RUN_SUFFIXES[car]
    base = run_name.strip() or f"{reward_config}_{suffix}"
    tokens = base.lower().replace("-", "_").split("_")
    if suffix not in tokens:
        base = f"{base}_{suffix}"
    return base


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
    speed_factor = int(session.get("speed_factor", 1) or 1)
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
        "reward_config": session.get("reward_config", "v1"),
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
    timesteps = int(data.get("timesteps", 100_000))
    stream = bool(data.get("stream", mode == "training"))
    max_damage = float(data.get("max_damage", 500.0))
    speed_factor = int(data.get("speed_factor", 1))
    if speed_factor not in SPEED_FACTORS:
        return jsonify({
            "ok": False,
            "message": f"Unknown speed factor {speed_factor!r}. Choose one of: {sorted(SPEED_FACTORS)}",
        }), 400
    # Reward config key ("v1" or "v2"). Written into launcher_session.json so
    # train_live_ppo_run.py picks it up without needing a CLI flag from schtasks.
    reward_config = str(data.get("reward_config", "v1"))
    if reward_config not in REWARD_CONFIG_KEYS:
        return jsonify({
            "ok": False,
            "message": f"Unknown reward config {reward_config!r}. Choose one of: {sorted(REWARD_CONFIG_KEYS)}",
        }), 400

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
            "reward_config": reward_config,
        })
        _save_session_config(session)
        obs_ok, obs_message = _obs_action("start")
        status = 200 if obs_ok else 500
        return jsonify({"ok": obs_ok, "message": obs_message or "Stream started."}), status

    if mode == "beamng_only":
        subprocess.run(["schtasks", "/run", "/tn", "BeamNGDirect"], capture_output=True)
        return jsonify({"ok": True, "message": "BeamNG launched standalone."})

    if fresh:
        run_name = _ensure_car_suffix(run_name, car, reward_config)
    else:
        resume_name = run_name or rm.find_latest_run()
        if not resume_name:
            return jsonify({
                "ok": False,
                "message": "No run selected to resume. Start a fresh run.",
            }), 400
        run_car = _run_vehicle_model(resume_name)
        if run_car != car:
            reason = (
                f"it is tagged as {CAR_LABELS[run_car]}"
                if run_car
                else "it has no saved vehicle metadata"
            )
            return jsonify({
                "ok": False,
                "message": (
                    f"Cannot resume '{resume_name}' as {CAR_LABELS[car]} because {reason}. "
                    "Start a fresh run for this car."
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
        "reward_config": reward_config,
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
    stream_label = " with OBS stream" if stream else ""
    obs_warning = "" if obs_ok else f" OBS warning: {obs_message}"
    return jsonify({
        "ok": True,
        "message": (
            f"Training started: {CAR_LABELS[car]}, {timesteps:,} steps, "
            f"{speed_factor}x speed, reward {reward_config}{stream_label}.{obs_warning}"
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
    return jsonify({"ok": obs_ok, "message": obs_message or "Stream stopped."})


@app.route("/api/start/stream", methods=["POST"])
def api_start_stream():
    obs_ok, obs_message = _obs_action("start")
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
