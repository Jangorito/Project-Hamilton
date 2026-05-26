"""OBS WebSocket controller — start, stop, or check stream status."""
from __future__ import annotations

import argparse
import sys

try:
    import obsws_python as obs
except ImportError:
    sys.exit("obsws-python not installed. Run: pip install obsws-python")


def get_client(host: str, port: int, password: str) -> obs.ReqClient:
    try:
        return obs.ReqClient(host=host, port=port, password=password, timeout=5)
    except Exception as exc:
        sys.exit(f"Could not connect to OBS WebSocket at {host}:{port} — {exc}\n"
                 "Make sure OBS is running and WebSocket is enabled (Tools → WebSocket Server Settings).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Control OBS streaming via WebSocket.")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=4455)
    parser.add_argument("--password", default="",
                        help="OBS WebSocket password (leave blank if not set)")
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


if __name__ == "__main__":
    main()
