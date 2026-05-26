"""OBS WebSocket controller — start, stop, check stream status, or watch a process."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
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


def _hwnd_for_process_name(name: str) -> int | None:
    """Return the first visible HWND whose owning process matches name, or None."""
    user32 = ctypes.windll.user32
    exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True,
    )
    pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.strip('"').split('","')
        if len(parts) >= 2:
            try:
                pids.add(int(parts[1]))
            except ValueError:
                pass
    if not pids:
        return None

    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def enum_cb(hwnd, _):
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            found.append(hwnd)
        return True  # keep enumerating to collect all windows

    user32.EnumWindows(enum_cb, 0)
    if not found:
        return None
    # prefer a visible window; fall back to any window owned by the process
    visible = [h for h in found if user32.IsWindowVisible(h)]
    return visible[0] if visible else found[0]


def discover_exe_from_obs(cl: obs.ReqClient) -> list[str]:
    """Query OBS window-capture sources and return their exe names (sans .exe)."""
    exes: list[str] = []
    try:
        scene = cl.get_current_program_scene().current_program_scene_name
        items = cl.get_scene_item_list(scene).scene_items
        for item in items:
            source_name = item.get("sourceName") or item.get("inputName", "")
            if not source_name:
                continue
            try:
                resp = cl.get_input_settings(source_name)
            except Exception:
                continue
            if resp.input_kind != "window_capture":
                continue
            # OBS window capture stores window as "title:class:exe.exe"
            window = resp.input_settings.get("window", "")
            parts = window.split(":")
            if len(parts) >= 3 and parts[2]:
                exes.append(parts[2])  # keep .exe so tasklist filter matches exactly
    except Exception as exc:
        print(f"OBS discovery failed: {exc}")
    return exes


def spotlight_process(name: str) -> bool:
    """Restore and bring the main window of a named process to the foreground."""
    hwnd = _hwnd_for_process_name(name)
    if hwnd is None:
        return False
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    return True


def is_process_running(name: str) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"],
        capture_output=True, text=True,
    )
    return name.lower() in result.stdout.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Control OBS streaming via WebSocket.")
    parser.add_argument("action", choices=["start", "stop", "status", "watch", "spotlight", "game-capture", "display-capture"])
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=4455)
    parser.add_argument("--password", default="",
                        help="OBS WebSocket password (leave blank if not set)")
    parser.add_argument("--process",  default=BEAMNG_PROCESS,
                        help="Process name to watch (default: BeamNG.x64)")
    args = parser.parse_args()

    if args.action == "spotlight":
        if spotlight_process(args.process):
            print(f"{args.process} window brought to foreground.")
            return
        print(f"{args.process} not found — querying OBS for window capture sources...")
        cl = get_client(args.host, args.port, args.password)
        candidates = discover_exe_from_obs(cl)
        if not candidates:
            print("No window capture sources found in the current OBS scene.")
            sys.exit(1)
        for exe in candidates:
            print(f"  trying: {exe}")
            if spotlight_process(exe):
                print(f"{exe} window brought to foreground. (use --process {exe} next time)")
                return
        print("Found window capture sources in OBS but none matched a running process.")
        sys.exit(1)

    cl = get_client(args.host, args.port, args.password)
    status = cl.get_stream_status()

    if args.action == "game-capture":
        scene = cl.get_current_program_scene().current_program_scene_name
        exe = args.process if args.process.lower().endswith(".exe") else f"{args.process}.exe"
        input_name = f"{exe.replace('.exe', '')}_game_capture"
        try:
            cl.create_input(scene, input_name, "game_capture",
                            {"mode": "any_fullscreen"}, True)
            print(f"Added game capture source '{input_name}' (any fullscreen) to scene '{scene}'.")
        except Exception as exc:
            print(f"Failed: {exc}")

    elif args.action == "display-capture":
        scene = cl.get_current_program_scene().current_program_scene_name
        try:
            cl.create_input(scene, "Display_Capture", "monitor_capture", {"monitor": 0}, True)
            print(f"Added display capture source to scene '{scene}'.")
        except Exception as exc:
            print(f"Failed: {exc}")

    elif args.action == "status":
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
