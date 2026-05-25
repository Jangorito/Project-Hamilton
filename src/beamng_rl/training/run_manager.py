"""Shared run management for training, evaluation, and visualisation scripts.

A "run" is a named training experiment with its own checkpoint directory,
TensorBoard log directory, config snapshot, and results summary. All scripts
consume this module so run discovery and path resolution stay consistent.

Directory layout
----------------
models/
  runs/
    <run_name>/
      run_config.json   # machine-readable config snapshot (written at launch)
      run_info.md       # human-readable summary (written at launch, results appended at end)
      checkpoints/
        ppo_<ts>_steps.zip
        ...
logs/
  runs/
    <run_name>/         # TensorBoard event files
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_ROOT = _REPO_ROOT / "models" / "runs"
DEFAULT_LOGS_ROOT = _REPO_ROOT / "logs" / "runs"


def _parse_timestep(path: Path) -> int:
    match = re.search(r"_(\d+)_steps", path.stem)
    return int(match.group(1)) if match else 0


class RunManager:
    def __init__(
        self,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        logs_root: Path = DEFAULT_LOGS_ROOT,
    ) -> None:
        self.runs_root = runs_root
        self.logs_root = logs_root

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def run_dir(self, run_name: str) -> Path:
        return self.runs_root / run_name

    def checkpoint_dir(self, run_name: str) -> Path:
        return self.run_dir(run_name) / "checkpoints"

    def log_dir(self, run_name: str) -> Path:
        return self.logs_root / run_name

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def exists(self, run_name: str) -> bool:
        return self.run_dir(run_name).exists()

    def list_runs(self) -> list[str]:
        if not self.runs_root.exists():
            return []
        return sorted(d.name for d in self.runs_root.iterdir() if d.is_dir())

    def find_all_checkpoints(self, run_name: str) -> list[Path]:
        ckpt_dir = self.checkpoint_dir(run_name)
        if not ckpt_dir.exists():
            return []
        return sorted(ckpt_dir.glob("*.zip"), key=_parse_timestep)

    def find_latest_checkpoint(self, run_name: str) -> Path | None:
        checkpoints = self.find_all_checkpoints(run_name)
        return checkpoints[-1] if checkpoints else None

    def find_latest_run(self) -> str | None:
        """Return the run name whose most recent checkpoint was saved last."""
        best_run: str | None = None
        best_mtime: float | None = None
        for run_name in self.list_runs():
            ckpt = self.find_latest_checkpoint(run_name)
            if ckpt is None:
                continue
            mtime = ckpt.stat().st_mtime
            if best_mtime is None or mtime > best_mtime:
                best_mtime = mtime
                best_run = run_name
        return best_run

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_name: str,
        config: dict[str, Any],
        *,
        resumed_from: str | None = None,
    ) -> Path:
        """Create the run directory, write config JSON and run_info.md.

        Raises FileExistsError if the run already exists (caller should
        decide whether to abort or continue into an existing run).
        """
        run_dir = self.run_dir(run_name)
        if run_dir.exists():
            raise FileExistsError(
                f"Run '{run_name}' already exists at {run_dir}.\n"
                "Choose a different name or omit --fresh to resume it."
            )
        self.checkpoint_dir(run_name).mkdir(parents=True, exist_ok=True)
        self.log_dir(run_name).mkdir(parents=True, exist_ok=True)

        (run_dir / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
        (run_dir / "run_info.md").write_text(
            _build_run_info_md(run_name, config, resumed_from),
            encoding="utf-8",
        )
        return run_dir

    def open_run(self, run_name: str) -> Path:
        """Ensure directories exist for a run that already exists (resume)."""
        self.checkpoint_dir(run_name).mkdir(parents=True, exist_ok=True)
        self.log_dir(run_name).mkdir(parents=True, exist_ok=True)
        return self.run_dir(run_name)

    def finalize_run(self, run_name: str, results: dict[str, Any]) -> None:
        """Append the results block to run_info.md once training is done."""
        md_path = self.run_dir(run_name) / "run_info.md"
        if not md_path.exists():
            return
        lines = [f"- **{k}**: {v}" for k, v in results.items()]
        block = "\n".join(lines)
        current = md_path.read_text(encoding="utf-8")
        updated = current.replace(
            "*(populated when training completes)*", block
        )
        md_path.write_text(updated, encoding="utf-8")

    def load_config(self, run_name: str) -> dict[str, Any]:
        config_path = self.run_dir(run_name) / "run_config.json"
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Interactive helpers used by scripts
# ------------------------------------------------------------------

def prompt_run_name() -> str:
    """Prompt the user for a run name interactively; exit on empty input."""
    try:
        name = input("Run name: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if not name:
        sys.exit("Error: run name cannot be empty.")
    return name


# ------------------------------------------------------------------
# Markdown generation
# ------------------------------------------------------------------

def _section(title: str, rows: dict[str, Any]) -> str:
    lines = [f"## {title}", ""]
    lines += [f"| {k} | `{v}` |" for k, v in rows.items()]
    return "\n".join(["## " + title, "", "| Parameter | Value |", "|---|---|"]
                     + [f"| {k} | `{v}` |" for k, v in rows.items()])


def _build_run_info_md(
    run_name: str,
    config: dict[str, Any],
    resumed_from: str | None,
) -> str:
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resume_line = f"`{resumed_from}`" if resumed_from else "*(fresh start)*"

    sections: list[str] = [
        f"# Training Run: {run_name}",
        "",
        f"**Started:** {started}  ",
        f"**Resumed from:** {resume_line}",
        "",
    ]

    for section_key, title in [
        ("reward", "Reward Config"),
        ("env", "Environment"),
        ("ppo", "PPO Hyperparameters"),
        ("training", "Training"),
    ]:
        sub = config.get(section_key, {})
        if sub:
            sections.append(_section(title, sub))
            sections.append("")

    sections += [
        "## Results",
        "",
        "*(populated when training completes)*",
        "",
        "## Notes",
        "",
        "*(add your observations here)*",
        "",
    ]

    return "\n".join(sections)
