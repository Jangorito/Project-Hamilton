"""Benchmark BeamNG deterministic physics speed_factor on Jango.

Sweeps the speed_factor argument of bng.settings.set_deterministic() to find
the highest multiplier Jango's CPU can sustain without falling behind.

  speed_factor=1  => real-time
  speed_factor=4  => sim runs 4x faster than wall-clock
  speed_factor=16 => sim runs 16x faster (if the CPU can keep up)

Key metric — utilization: actual_speedup / requested_speed_factor
  ~1.0 = machine is keeping up
  <0.9 = physics can't meet the requested pace  <-- cracking point

The "cracking point" is where:
  - utilization drops below ~0.9
  - mean_step_ms exceeds budget_ms (the per-step time budget at that speedup)
  - p95 latency spikes

Usage (on Jango):
    # Step 1 — launch BeamNG in the desktop session (Session 1):
    #   schtasks /run /tn "BeamNGDirect"
    #   (wait ~2 min for it to be ready)
    # Step 2 — connect and benchmark:
    venv\\Scripts\\python.exe scripts\\env\\bench_deterministic.py --no-launch
    venv\\Scripts\\python.exe scripts\\env\\bench_deterministic.py --no-launch --speed-sweep 1,2,4,8,16,32,64
    venv\\Scripts\\python.exe scripts\\env\\bench_deterministic.py --no-launch --sps 120

Results saved to data/diag/bench_deterministic_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beamngpy import BeamNGpy
from beamngpy.sensors import Damage

from beamng_rl.bootstrap.beamng_setup import (
    HOST,
    PORT,
    SPAWN_POS,
    SPAWN_ROT,
    build_hirochi_etkc_scenario,
    resolve_beamng_home_from_path_or_env,
)

DEFAULT_SPS = 60
DEFAULT_SPEED_SWEEP = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_STEPS_PER_ACTION = 1   # keep at 1 to isolate speed_factor; use --steps-per-action=15 to match production
WARMUP_STEPS = 50
BENCH_STEPS = 300
RESET_TRIALS = 5
OUTPUT_DIR = REPO_ROOT / "data" / "diag"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(int(p / 100 * len(s)), len(s) - 1)]


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------

def bench_speed_factor(
    bng: BeamNGpy,
    vehicle,
    speed_factor: int,
    sps: int,
    steps_per_action: int,
    warmup: int,
    n_steps: int,
) -> dict:
    """Time n_steps of (bng.step + poll) at a given speed_factor."""

    # How long each step *should* take if the machine keeps up
    budget_s = steps_per_action / (sps * speed_factor)

    vehicle.control(throttle=0.3, steering=0.0, brake=0.0)

    for _ in range(warmup):
        bng.step(steps_per_action, wait=True)
        vehicle.sensors.poll("state", "damage")

    step_times: list[float] = []
    poll_times: list[float] = []
    nan_count = 0

    for _ in range(n_steps):
        t0 = time.perf_counter()
        bng.step(steps_per_action, wait=True)
        t1 = time.perf_counter()
        vehicle.sensors.poll("state", "damage")
        t2 = time.perf_counter()

        step_times.append(t1 - t0)
        poll_times.append(t2 - t1)

        pos = (vehicle.state or {}).get("pos", [0.0, 0.0, 0.0])
        if any(v != v for v in pos):   # NaN != NaN
            nan_count += 1

    sim_time_s = n_steps * steps_per_action / sps
    wall_time_s = sum(step_times) + sum(poll_times)
    actual_speedup = sim_time_s / wall_time_s
    utilization = actual_speedup / speed_factor

    return {
        "speed_factor": speed_factor,
        "sps": sps,
        "steps_per_action": steps_per_action,
        "n_steps": n_steps,
        "budget_ms": round(budget_s * 1000, 2),
        "actual_speedup": round(actual_speedup, 3),
        "utilization": round(utilization, 3),
        "rl_steps_per_sec": round(n_steps / wall_time_s, 1),
        "mean_step_ms": round(_mean(step_times) * 1000, 2),
        "p95_step_ms": round(_pct(step_times, 95) * 1000, 2),
        "mean_poll_ms": round(_mean(poll_times) * 1000, 2),
        "p95_poll_ms": round(_pct(poll_times, 95) * 1000, 2),
        "nan_count": nan_count,
    }


def bench_resets(bng: BeamNGpy, vehicle, n_resets: int) -> dict:
    times: list[float] = []
    for _ in range(n_resets):
        vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        t0 = time.perf_counter()
        vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT, reset=True)
        bng.step(3, wait=True)
        vehicle.sensors.poll("state", "damage")
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "n_resets": n_resets,
        "mean_ms": round(_mean(times) * 1000, 1),
        "p95_ms": round(_pct(times, 95) * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    hdr = (
        f"{'sf':>5}  {'actual_x':>9}  {'util':>6}  {'budget_ms':>10}"
        f"  {'step_ms':>8}  {'p95_ms':>8}  {'poll_ms':>8}  {'NaN':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        flag = "  <-- BEHIND" if r["utilization"] < 0.9 else ""
        print(
            f"{r['speed_factor']:>5}  "
            f"{r['actual_speedup']:>9.3f}  "
            f"{r['utilization']:>6.3f}  "
            f"{r['budget_ms']:>10.2f}"
            f"  {r['mean_step_ms']:>8.2f}  "
            f"{r['p95_step_ms']:>8.2f}  "
            f"{r['mean_poll_ms']:>8.2f}  "
            f"{r['nan_count']:>5}"
            f"{flag}"
        )


def kill_stale_beamng() -> None:
    for exe in ("BeamNG.tech.x64.exe", "BeamNG.tech.exe"):
        result = subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True)
        if result.returncode == 0:
            print(f"  Killed stale {exe}.")
            time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep speed_factor in set_deterministic() to find Jango's physics ceiling"
    )
    parser.add_argument("--beamng-home", type=Path, default=None)
    parser.add_argument(
        "--no-launch", action="store_true",
        help="Connect to an already-running BeamNG instance (use after: schtasks /run /tn BeamNGDirect)",
    )
    parser.add_argument(
        "--speed-sweep",
        default=",".join(str(s) for s in DEFAULT_SPEED_SWEEP),
        help="Comma-separated speed_factor values (default: 1,2,4,8,16,32,64)",
    )
    parser.add_argument(
        "--sps", type=int, default=DEFAULT_SPS,
        help=f"steps_per_second passed to set_deterministic (default: {DEFAULT_SPS})",
    )
    parser.add_argument(
        "--steps-per-action", type=int, default=DEFAULT_STEPS_PER_ACTION,
        help=f"Physics steps per bng.step() call (default: {DEFAULT_STEPS_PER_ACTION}; production uses 15)",
    )
    parser.add_argument("--bench-steps", type=int, default=BENCH_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--reset-trials", type=int, default=RESET_TRIALS)
    args = parser.parse_args()

    speed_sweep = [int(s.strip()) for s in args.speed_sweep.split(",")]
    beamng_home = resolve_beamng_home_from_path_or_env(args.beamng_home)

    print("=== bench_deterministic (speed_factor sweep) ===")
    print(f"BeamNG home      : {beamng_home}")
    print(f"sps              : {args.sps}")
    print(f"steps_per_action : {args.steps_per_action}")
    print(f"speed_factor     : {speed_sweep}")
    print(f"bench_steps      : {args.bench_steps}  (+{args.warmup_steps} warmup each)")
    print()

    if args.no_launch:
        bng = BeamNGpy(HOST, PORT, quit_on_close=False)
    else:
        kill_stale_beamng()
        bng = BeamNGpy(HOST, PORT, home=str(beamng_home))

    try:
        if args.no_launch:
            print("Connecting to already-running BeamNG...")
            t0 = time.perf_counter()
            bng.open(launch=False)
            print(f"  connected: {time.perf_counter() - t0:.1f}s")
        else:
            print("Launching BeamNG... (Hirochi may take 1-3 min to load)")
            t0 = time.perf_counter()
            bng.open(launch=True)
            print(f"  open(): {time.perf_counter() - t0:.1f}s")

        bng.settings.set_deterministic(steps_per_second=args.sps, speed_factor=speed_sweep[0])
        bng.pause()

        scenario, vehicle = build_hirochi_etkc_scenario(
            bng, vehicle_id="ego_vehicle", scenario_instance_name="bench_det"
        )
        vehicle.attach_sensor("damage", Damage())

        t1 = time.perf_counter()
        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()
        bng.pause()
        print(f"  scenario load+start: {time.perf_counter() - t1:.1f}s")

        vehicle.sensors.poll("state")

        # --- Speed factor sweep ---
        print(
            f"\n=== speed_factor sweep  "
            f"(sps={args.sps}, steps_per_action={args.steps_per_action}) ===\n"
        )

        results = []
        for sf in speed_sweep:
            print(f"  speed_factor={sf:>3} ...", end="", flush=True)

            bng.settings.set_deterministic(steps_per_second=args.sps, speed_factor=sf)
            bng.pause()

            vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
            vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT, reset=True)
            bng.step(3, wait=True)
            vehicle.sensors.poll("state", "damage")

            r = bench_speed_factor(
                bng, vehicle, sf, args.sps, args.steps_per_action,
                args.warmup_steps, args.bench_steps,
            )
            results.append(r)

            warn = "  BEHIND" if r["utilization"] < 0.9 else ""
            print(
                f"  actual={r['actual_speedup']:.2f}x  util={r['utilization']:.2f}"
                f"  step={r['mean_step_ms']:.1f}ms  budget={r['budget_ms']:.1f}ms{warn}"
            )

        print()
        print_table(results)

        # --- Reset latency ---
        print(f"\n=== Reset latency ({args.reset_trials} trials) ===")
        reset_stats = bench_resets(bng, vehicle, args.reset_trials)
        print(
            f"  mean={reset_stats['mean_ms']:.1f}ms  "
            f"p95={reset_stats['p95_ms']:.1f}ms  "
            f"min={reset_stats['min_ms']:.1f}ms  "
            f"max={reset_stats['max_ms']:.1f}ms"
        )

        # --- Save ---
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUTPUT_DIR / f"bench_deterministic_{ts}.json"
        payload = {
            "timestamp_utc": ts,
            "beamng_home": str(beamng_home),
            "sps": args.sps,
            "steps_per_action": args.steps_per_action,
            "bench_steps": args.bench_steps,
            "warmup_steps": args.warmup_steps,
            "speed_factor_results": results,
            "reset_stats": reset_stats,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")

    finally:
        bng.disconnect()
        print("BeamNG disconnected.")


if __name__ == "__main__":
    main()
