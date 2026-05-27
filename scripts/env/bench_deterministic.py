"""Baseline BeamNG deterministic physics throughput on Jango.

Launches BeamNG.tech, loads the production Hirochi + ETK-C scenario, then
sweeps steps_per_action to measure raw BeamNGpy API throughput directly —
one layer below the Gym env so the numbers are not polluted by reward
computation, observation building, or termination checks.

Metrics per steps_per_action config:
  RL/s         — policy steps per wall-clock second
  phys/s       — physics steps per wall-clock second
  speedup      — sim-time / wall-time ratio (>1 = faster than real-time)
  step_ms      — mean bng.step() latency
  poll_ms      — mean vehicle.sensors.poll() latency
  NaN          — positions that came back NaN (stability indicator)

Also times episode resets (teleport + settle + first poll).

Complements scripts/env/bench_nogfx.py, which tests the full Gym loop at a
fixed steps_per_action. Run this script first to pick a good steps_per_action
value, then run bench_nogfx.py to see end-to-end throughput.

Usage (on Jango, in the project venv):
    python scripts/env/bench_deterministic.py
    python scripts/env/bench_deterministic.py --steps-sweep 1,2,3,5,10,20,50
    python scripts/env/bench_deterministic.py --det-hz 120
    python scripts/env/bench_deterministic.py --bench-steps 500

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

PHYSICS_HZ = 120           # BeamNG physics rate — 1 step = 1/120 s of sim-time
PHYSICS_DT = 1.0 / PHYSICS_HZ

DEFAULT_DET_HZ = 60        # matches _connect_beamng() in the live env
DEFAULT_STEPS_SWEEP = [1, 2, 3, 5, 10, 15, 20]
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

def bench_steps(
    bng: BeamNGpy,
    vehicle,
    steps_per_action: int,
    warmup: int,
    n_steps: int,
) -> dict:
    """Time n_steps of (bng.step + vehicle.sensors.poll) after a warmup pass."""

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

    sim_time_s = n_steps * steps_per_action * PHYSICS_DT
    wall_time_s = sum(step_times) + sum(poll_times)

    return {
        "steps_per_action": steps_per_action,
        "n_steps": n_steps,
        "sim_time_s": round(sim_time_s, 4),
        "wall_time_s": round(wall_time_s, 4),
        "speedup": round(sim_time_s / wall_time_s, 3),
        "rl_steps_per_sec": round(n_steps / wall_time_s, 1),
        "physics_steps_per_sec": round(n_steps * steps_per_action / wall_time_s, 1),
        "mean_step_ms": round(_mean(step_times) * 1000, 2),
        "p95_step_ms": round(_pct(step_times, 95) * 1000, 2),
        "mean_poll_ms": round(_mean(poll_times) * 1000, 2),
        "p95_poll_ms": round(_pct(poll_times, 95) * 1000, 2),
        "nan_count": nan_count,
    }


def bench_resets(bng: BeamNGpy, vehicle, n_resets: int) -> dict:
    """Time episode resets: teleport + settle (3 steps) + poll."""
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

def print_step_table(results: list[dict]) -> None:
    hdr = (
        f"{'spa':>4}  {'RL/s':>7}  {'phys/s':>8}  {'speedup':>9}"
        f"  {'step_ms':>8}  {'poll_ms':>8}  {'p95_step':>9}  {'NaN':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['steps_per_action']:>4}  "
            f"{r['rl_steps_per_sec']:>7.1f}  "
            f"{r['physics_steps_per_sec']:>8.1f}  "
            f"{r['speedup']:>8.3f}x"
            f"  {r['mean_step_ms']:>8.2f}  "
            f"{r['mean_poll_ms']:>8.2f}  "
            f"{r['p95_step_ms']:>9.2f}  "
            f"{r['nan_count']:>5}"
        )


def kill_stale_beamng() -> None:
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "BeamNG.tech.x64.exe"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("  Killed stale BeamNG process.")
        time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline BeamNG deterministic physics throughput"
    )
    parser.add_argument(
        "--beamng-home", type=Path, default=None,
        help="Override BeamNG.tech install dir (default: auto-detect)",
    )
    parser.add_argument(
        "--steps-sweep",
        default=",".join(str(s) for s in DEFAULT_STEPS_SWEEP),
        help="Comma-separated steps_per_action values to test "
             "(default: 1,2,3,5,10,15,20)",
    )
    parser.add_argument(
        "--det-hz", type=int, default=DEFAULT_DET_HZ,
        help=f"set_deterministic() Hz value (default: {DEFAULT_DET_HZ})",
    )
    parser.add_argument(
        "--bench-steps", type=int, default=BENCH_STEPS,
        help=f"Measured steps per configuration (default: {BENCH_STEPS})",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=WARMUP_STEPS,
        help=f"Warmup steps discarded before timing (default: {WARMUP_STEPS})",
    )
    parser.add_argument(
        "--reset-trials", type=int, default=RESET_TRIALS,
        help=f"Number of reset trials to time (default: {RESET_TRIALS})",
    )
    args = parser.parse_args()

    steps_sweep = [int(s.strip()) for s in args.steps_sweep.split(",")]
    beamng_home = resolve_beamng_home_from_path_or_env(args.beamng_home)

    print("=== bench_deterministic ===")
    print(f"BeamNG home  : {beamng_home}")
    print(f"det_hz       : {args.det_hz}")
    print(f"steps sweep  : {steps_sweep}")
    print(f"bench_steps  : {args.bench_steps}  (+{args.warmup_steps} warmup each)")
    print(f"reset_trials : {args.reset_trials}")
    print()

    kill_stale_beamng()

    bng = BeamNGpy(HOST, PORT, home=str(beamng_home))

    try:
        # --- Launch + setup (mirrors _connect_beamng in the live env) -------
        print("Launching BeamNG... (Hirochi may take 1-3 min to load)")
        t0 = time.perf_counter()
        bng.open(launch=True)
        print(f"  open(): {time.perf_counter() - t0:.1f}s")

        bng.settings.set_deterministic(args.det_hz)
        bng.pause()

        scenario, vehicle = build_hirochi_etkc_scenario(
            bng,
            vehicle_id="ego_vehicle",
            scenario_instance_name="bench_det",
        )
        vehicle.attach_sensor("damage", Damage())

        t1 = time.perf_counter()
        bng.scenario.load(scenario)
        bng.ui.hide_hud()
        bng.scenario.start()
        bng.pause()
        print(f"  scenario load+start: {time.perf_counter() - t1:.1f}s")

        vehicle.sensors.poll("state")  # prime state sensor

        # --- Step throughput sweep ------------------------------------------
        print(f"\n=== Step throughput (det_hz={args.det_hz}) ===\n")

        step_results = []
        for spa in steps_sweep:
            print(f"  steps_per_action={spa:>2} ...", end="", flush=True)
            vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
            vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT, reset=True)
            bng.step(3, wait=True)
            vehicle.sensors.poll("state", "damage")

            r = bench_steps(bng, vehicle, spa, args.warmup_steps, args.bench_steps)
            step_results.append(r)
            print(
                f"  {r['rl_steps_per_sec']:>6.1f} RL/s  "
                f"{r['speedup']:.3f}x  "
                f"step={r['mean_step_ms']:.1f}ms  poll={r['mean_poll_ms']:.1f}ms"
            )

        print()
        print_step_table(step_results)

        # --- Reset latency --------------------------------------------------
        print(f"\n=== Reset latency ({args.reset_trials} trials) ===")
        reset_stats = bench_resets(bng, vehicle, args.reset_trials)
        print(
            f"  mean={reset_stats['mean_ms']:.1f}ms  "
            f"p95={reset_stats['p95_ms']:.1f}ms  "
            f"min={reset_stats['min_ms']:.1f}ms  "
            f"max={reset_stats['max_ms']:.1f}ms"
        )

        # --- Save -----------------------------------------------------------
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = OUTPUT_DIR / f"bench_deterministic_{ts}.json"
        payload = {
            "timestamp_utc": ts,
            "beamng_home": str(beamng_home),
            "det_hz": args.det_hz,
            "bench_steps": args.bench_steps,
            "warmup_steps": args.warmup_steps,
            "physics_hz": PHYSICS_HZ,
            "step_results": step_results,
            "reset_stats": reset_stats,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")

    finally:
        bng.disconnect()
        print("BeamNG disconnected.")


if __name__ == "__main__":
    main()
