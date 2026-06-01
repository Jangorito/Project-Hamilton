from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(os.environ.get("HAMILTON_WATCH_TASK_CONFIG", REPO_ROOT / "config" / "watch_task.json"))
LOG_FILE = REPO_ROOT / "logs" / "remote" / "watch_latest.log"
PID_FILE = REPO_ROOT / "logs" / "remote" / "watch.pid"
PYTHON_EXE = REPO_ROOT / "venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = Path("python")


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Watch task config not found: {CONFIG_FILE}")
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def main() -> int:
    config = _load_config()
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("watch task config missing model_path")

    args = [
        str(PYTHON_EXE),
        str(REPO_ROOT / "scripts" / "eval" / "watch_model.py"),
        str(model_path),
        "--steps",
        str(config.get("steps", 500)),
        "--no-wait",
    ]

    run_name = config.get("run_name")
    if run_name:
        args.extend(["--run", str(run_name)])

    speed_factor = config.get("speed_factor")
    if speed_factor is not None:
        args.extend(["--speed-factor", str(speed_factor)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    try:
        with LOG_FILE.open("w", encoding="utf-8", buffering=1) as log_f:
            process = subprocess.Popen(
                args,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            return process.wait()
    finally:
        try:
            PID_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Watch task runner failed: {exc}", file=sys.stderr)
        raise
