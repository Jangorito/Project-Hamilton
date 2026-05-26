"""OBS WebSocket controller — start, stop, check stream status, or watch a process."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

try:
    import obsws_python as obs
except ImportError:
    sys.exit("obsws-python not installed. Run: pip install obsws-python")

BEAMNG_PROCESS = "BeamNG.x64"
WATCH_POLL_INTERVAL = 5   # seconds between process checks
WATCH_START_TIMEOUT = 120  # seconds to wait for BeamNG to appear before giving up


def get_client(host: str, port: int, password: str) -> obs.ReqClient:
    try:
        return obs.ReqClient(host=host, port=port, password=password, timeout=5)
    except Exception as exc:
        sys.exit(f"Could not connect to OBS WebSocket at {host}:{port} - {exc}\n"
                 "Make sure OBS is running and WebSocket is enabled (Tools -> WebSocket Server Settings).")


def is_process_running(name: str) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"],
        capture_output=True, text=True,
    )
    return name.lower() in result.stdout.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Control OBS streaming via WebSocket.")
    parser.add_argument("action", choices=["start", "stop", "status", "watch"])
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=4455)
    parser.add_argument("--password", default="",
                        help="OBS WebSocket password (leave blank if not set)")
    parser.add_argument("--process",  default=BEAMNG_PROCESS,
                        help="Process name to watch (default: BeamNG.x64)")
    args = parser.parse_args()

    cl = get_client(args.host, args.port, args.password)
    status = cl.get_stream_status()

    if args.action == "status":
        if status.output_active:
            secs = status.output_duration // 1000
            print(f"Streaming  ({secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d})")
        else:
            print("Not streaming.")

    elif args.action == "start":
        if status.output_active:
            print("Stream already running.")
        else:
            cl.start_stream()
            print("Stream started.")

    elif args.action == "stop":
        if not status.output_active:
            print("Stream not running.")
        else:
            cl.stop_stream()
            print("Stream stopped.")

    elif args.action == "watch":
        # Wait for BeamNG to appear
        print(f"Waiting for {args.process} to start (timeout {WATCH_START_TIMEOUT}s)...")
        deadline = time.time() + WATCH_START_TIMEOUT
        while not is_process_running(args.process):
            if time.time() > deadline:
                print(f"{args.process} never started - stopping stream.")
                cl.stop_stream()
                sys.exit(1)
            time.sleep(WATCH_POLL_INTERVAL)

        print(f"{args.process} detected - monitoring.")
        while is_process_running(args.process):
            time.sleep(WATCH_POLL_INTERVAL)

        print(f"{args.process} closed - stopping stream.")
        try:
            status = cl.get_stream_status()
            if status.output_active:
                cl.stop_stream()
                print("Stream stopped.")
            else:
                print("Stream was already stopped.")
        except Exception as exc:
            print(f"Could not stop stream: {exc}")


if __name__ == "__main__":
    main()
