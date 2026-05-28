"""
Export key TensorBoard scalars for a training run to a human-readable CSV.

Usage:
    python scripts/monitor/export_metrics.py --run v1_etk_16x_278670steps
    python scripts/monitor/export_metrics.py --run cond_C --tail 20

Reads the latest .tfevents file under logs/runs/<run_name>/ and appends new
rows to logs/runs/<run_name>/metrics_export.csv.  Safe to run while training
is active — uses event_accumulator with SIZE_GUIDANCE set to reload all events.
"""

import sys
import os
import argparse
import csv
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# ── scalar tags we care about ────────────────────────────────────────────────
TAGS = [
    # rollout
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    # reward components (written by RewardComponentLogger)
    "reward/progress_reward",
    "reward/heading_error_penalty",
    "reward/lateral_error_penalty",
    "reward/action_smoothness_penalty",
    "reward/off_track_penalty",
    "reward/curvature_overspeed_penalty",
    # v2.1 physics terms
    "reward/traction_budget_penalty",
    "reward/braking_shortfall_penalty",
    "reward/lateral_risk_ratio",
    "reward/risk_excess",
    "reward/physics_target_speed_mps",
    "reward/lateral_accel_required_mps2",
    "reward/braking_shortfall_m",
]

CORE_TAGS = {
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "reward/progress_reward",
    "reward/traction_budget_penalty",
    "reward/braking_shortfall_penalty",
    "reward/lateral_risk_ratio",
    "reward/physics_target_speed_mps",
}


def find_logdir(run_name: str) -> Path:
    """Return the deepest directory under logs/runs/<run> that has .tfevents files."""
    root = Path(__file__).resolve().parents[2] / "logs" / "runs" / run_name
    if not root.exists():
        sys.exit(f"[export_metrics] log dir not found: {root}")
    # SB3 nests events under PPO_1/, PPO_2/, etc.
    event_files = sorted(root.rglob("events.out.tfevents.*"))
    if event_files:
        # Use parent of the most-recently-modified event file
        return max(event_files, key=lambda p: p.stat().st_mtime).parent
    return root


def load_scalars(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for all available tags."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
        STORE_EVERYTHING_SIZE_GUIDANCE,
    )

    ea = EventAccumulator(str(logdir), size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    result: dict[str, list[tuple[int, float]]] = {}
    for tag in TAGS:
        if tag in available:
            result[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return result


def merge_by_step(scalars: dict[str, list[tuple[int, float]]]) -> list[dict]:
    """Pivot {tag: [(step, val)]} → [{step, tag1, tag2, ...}] aligned by step."""
    step_data: dict[int, dict] = {}
    for tag, entries in scalars.items():
        short = tag.split("/", 1)[-1]  # strip "rollout/" / "reward/"
        for step, val in entries:
            step_data.setdefault(step, {"step": step})[short] = (
                None if math.isnan(val) else round(val, 6)
            )
    return sorted(step_data.values(), key=lambda r: r["step"])


def write_csv(rows: list[dict], out_path: Path) -> int:
    """Write/overwrite CSV; return number of rows written."""
    if not rows:
        return 0
    fieldnames = ["step"] + sorted(k for k in rows[-1] if k != "step")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def print_tail(rows: list[dict], n: int) -> None:
    tail = rows[-n:]
    if not tail:
        print("  (no data yet)")
        return

    tags_present = [t.split("/", 1)[-1] for t in TAGS if t.split("/", 1)[-1] in (tail[-1] or {})]
    core_present = [k for k in tags_present if any(t.endswith(k) for t in CORE_TAGS)]
    show = core_present or tags_present[:8]

    header = f"{'step':>10}" + "".join(f"  {k[:22]:>22}" for k in show)
    print(header)
    print("-" * len(header))
    for row in tail:
        line = f"{row['step']:>10}"
        for k in show:
            v = row.get(k)
            line += f"  {str(v) if v is not None else '-':>22}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run name (subdirectory under logs/runs/)")
    ap.add_argument("--tail", type=int, default=10, help="Print last N rows to stdout (default 10)")
    ap.add_argument("--no-csv", action="store_true", help="Skip writing CSV (print only)")
    args = ap.parse_args()

    logdir = find_logdir(args.run)
    print(f"[export_metrics] reading {logdir}")

    scalars = load_scalars(logdir)
    if not scalars:
        print("[export_metrics] no scalar data found yet — is training running?")
        return

    rows = merge_by_step(scalars)
    print(f"[export_metrics] {len(rows)} steps across {len(scalars)} tags")

    if not args.no_csv:
        run_root = Path(__file__).resolve().parents[2] / "logs" / "runs" / args.run
        out = run_root / "metrics_export.csv"
        n = write_csv(rows, out)
        print(f"[export_metrics] wrote {n} rows -> {out}")

    print(f"\n-- last {min(args.tail, len(rows))} updates --")
    print_tail(rows, args.tail)


if __name__ == "__main__":
    main()
