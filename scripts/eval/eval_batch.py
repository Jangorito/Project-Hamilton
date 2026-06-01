"""Run eval_checkpoints.py sequentially for a list of runs.

Designed to run on the secondary BeamNG instance (port 25253) while training
occupies the primary instance (port 25252).

Each run gets a fresh BeamNG launch. If a run fails, the error is logged and
the script continues to the next run.

Usage:
    python scripts/eval/eval_batch.py
    python scripts/eval/eval_batch.py --runs v21b_etk_16x_280ksteps v21b_subaru_16x_278ksteps
    python scripts/eval/eval_batch.py --port 25253 --beamng-user "%LOCALAPPDATA%\\BeamNG_w2"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default runs to evaluate — edit this list or pass --runs on the CLI.
# Ordered: run cheaper/faster ones first.
DEFAULT_RUNS = [
    "v21b_etk_16x_280ksteps",        # Condition C
    "v21b_subaru_16x_278ksteps",      # Condition D
    "v21b_subaru_16x_obs2_454steps",  # Phase 2 obs2
    "v21b_subaru_16x_respawn_150ksteps",  # Phase 2 respawn
    "v22b_subaru_16x_obs2_280ksteps", # Phase 3 (only if training finished)
]

EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval" / "eval_checkpoints.py"


def has_final_model(run_name: str) -> bool:
    final = REPO_ROOT / "models" / "runs" / run_name / "model_final.zip"
    return final.is_file()


def has_any_checkpoint(run_name: str) -> bool:
    ckpt_dir = REPO_ROOT / "models" / "runs" / run_name / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob("ppo_*.zip"))


def eval_already_done(run_name: str) -> bool:
    """Return True if a checkpoints_eval.csv already exists for this run."""
    csv_path = REPO_ROOT / "logs" / "runs" / run_name / "checkpoints_eval.csv"
    return csv_path.is_file()


def run_eval(run_name: str, port: int, beamng_user: str | None, steps: int, force: bool) -> bool:
    """Run eval_checkpoints.py for one run. Returns True on success."""
    if not has_final_model(run_name) and not has_any_checkpoint(run_name):
        print(f"[eval_batch] SKIP {run_name}: no model_final.zip and no checkpoints found")
        return False

    if eval_already_done(run_name) and not force:
        print(f"[eval_batch] SKIP {run_name}: checkpoints_eval.csv already exists (use --force to re-run)")
        return True

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        run_name,
        "--port", str(port),
        "--steps", str(steps),
    ]
    if beamng_user:
        cmd += ["--beamng-user", beamng_user]

    print(f"\n{'='*60}")
    print(f"[eval_batch] Evaluating: {run_name}")
    print(f"[eval_batch] Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n[eval_batch] SUCCESS {run_name} ({elapsed:.0f}s)")
        return True
    else:
        print(f"\n[eval_batch] FAILED  {run_name} (exit {result.returncode}, {elapsed:.0f}s)")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="Run names to evaluate (default: DEFAULT_RUNS list in script)")
    ap.add_argument("--port", type=int, default=25252,
                    help="BeamNG port (default: 25252)")
    ap.add_argument("--beamng-user", default=None, metavar="PATH",
                    help="BeamNG user data folder for secondary instance "
                         "(e.g. %%LOCALAPPDATA%%\\BeamNG_w2)")
    ap.add_argument("--steps", type=int, default=500,
                    help="Rollout steps per checkpoint (default: 500)")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if checkpoints_eval.csv already exists")
    args = ap.parse_args()

    runs = args.runs or DEFAULT_RUNS

    # Validate: skip runs where training is clearly still in progress
    # (check training PID file — if it exists for this run, warn but don't skip)
    pid_file = REPO_ROOT / "logs" / "remote" / "training.pid"
    if pid_file.exists():
        print(f"[eval_batch] WARNING: training.pid exists — slot 1 training appears active.")
        print(f"[eval_batch] Using port {args.port} to avoid conflict.\n")

    results: list[tuple[str, bool]] = []
    for run_name in runs:
        ok = run_eval(run_name, args.port, args.beamng_user, args.steps, args.force)
        results.append((run_name, ok))

    print(f"\n{'='*60}")
    print("[eval_batch] SUMMARY")
    print(f"{'='*60}")
    for run_name, ok in results:
        status = "OK   " if ok else "FAIL "
        csv_path = REPO_ROOT / "logs" / "runs" / run_name / "checkpoints_eval.csv"
        csv_note = f"  -> {csv_path}" if csv_path.exists() else "  (no CSV)"
        print(f"  {status} {run_name}{csv_note}")

    failed = [r for r, ok in results if not ok]
    if failed:
        print(f"\n[eval_batch] {len(failed)} run(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n[eval_batch] All {len(results)} run(s) completed successfully.")


if __name__ == "__main__":
    main()
