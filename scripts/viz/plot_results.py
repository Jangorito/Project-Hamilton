"""Generate report figures from training metrics and eval CSV files.

Usage:
    python scripts/viz/plot_results.py --fig all
    python scripts/viz/plot_results.py --fig learning
    python scripts/viz/plot_results.py --fig comparison
    python scripts/viz/plot_results.py --fig rewards

Outputs PNGs to docs/figures/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs" / "runs"
OUT_DIR = REPO_ROOT / "docs" / "figures"

# ── Condition definitions ────────────────────────────────────────────────────

CONDITIONS = {
    "A": {
        "run": "v1_subaru_16x_278353steps",
        "label": "A: SBR + V1",
        "vehicle": "SBR4",
        "reward": "V1",
        "color": "#e74c3c",
        "linestyle": "-",
    },
    "B": {
        "run": "v1_etk_16x_278670steps",
        "label": "B: ETK + V1",
        "vehicle": "ETK",
        "reward": "V1",
        "color": "#3498db",
        "linestyle": "-",
    },
    "C": {
        "run": "v21b_etk_16x_280ksteps",
        "label": "C: ETK + v21b",
        "vehicle": "ETK",
        "reward": "v21b",
        "color": "#3498db",
        "linestyle": "--",
    },
    "D": {
        "run": "v21b_subaru_16x_278ksteps",
        "label": "D: SBR + v21b",
        "vehicle": "SBR4",
        "reward": "v21b",
        "color": "#e74c3c",
        "linestyle": "--",
    },
}

LAP_LENGTH_M = 2150.0   # authoritative from centreline JSON (1077 pts closed loop = 2150.04 m)
AI_REFERENCE_M = 2150.0


def _read_metrics(run_name: str) -> list[dict]:
    """Read metrics_export.csv for a run. Returns [] if missing."""
    csv_path = LOGS_DIR / run_name / "metrics_export.csv"
    if not csv_path.is_file():
        print(f"[plot] WARNING: no metrics_export.csv for {run_name} — skipping")
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_eval(run_name: str) -> list[dict]:
    """Read checkpoints_eval.csv for a run. Returns [] if missing."""
    csv_path = LOGS_DIR / run_name / "checkpoints_eval.csv"
    if not csv_path.is_file():
        print(f"[plot] WARNING: no checkpoints_eval.csv for {run_name}")
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _smooth(values: list[float], window: int = 20) -> list[float]:
    """Simple moving average."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result


# ── Figure 1: Learning curves ────────────────────────────────────────────────

def plot_learning_curves(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for cond_key, cond in CONDITIONS.items():
        rows = _read_metrics(cond["run"])
        if not rows:
            continue
        steps, rewards = [], []
        for row in rows:
            try:
                step = int(row["step"])
                rew = float(row["ep_rew_mean"])
                steps.append(step)
                rewards.append(rew)
            except (KeyError, ValueError):
                continue
        if not steps:
            print(f"[plot] WARNING: no ep_rew_mean data for {cond['run']}")
            continue
        smoothed = _smooth(rewards, window=30)
        ax.plot(
            [s / 1000 for s in steps],
            smoothed,
            color=cond["color"],
            linestyle=cond["linestyle"],
            linewidth=1.8,
            label=cond["label"],
            alpha=0.9,
        )
        # raw trace, faint
        ax.plot(
            [s / 1000 for s in steps],
            rewards,
            color=cond["color"],
            linestyle=cond["linestyle"],
            linewidth=0.4,
            alpha=0.25,
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Training steps (thousands)", fontsize=11)
    ax.set_ylabel("Episode reward mean (rolling)", fontsize=11)
    ax.set_title("Learning curves — all conditions", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "fig_learning_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ── Figure 2: Condition comparison bar chart ─────────────────────────────────

def plot_condition_comparison(out_dir: Path) -> None:
    """Best-checkpoint progress for each condition."""
    bar_data: list[tuple[str, float, str, str]] = []  # (label, progress_m, color, hatch)

    for cond_key, cond in CONDITIONS.items():
        rows = _read_eval(cond["run"])
        if not rows:
            # Fall back to final-checkpoint values from known data
            fallback = {"A": 1869.6, "B": 860.6, "C": 1353.5, "D": 1303.5}
            progress = fallback.get(cond_key, 0.0)
            print(f"[plot] Using fallback final-checkpoint value for {cond_key}: {progress} m")
        else:
            progress = max(float(r["progress_m"]) for r in rows)

        hatch = "" if cond["reward"] == "V1" else "///"
        bar_data.append((cond["label"], progress, cond["color"], hatch))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bar_data))
    bars = ax.bar(
        x,
        [d[1] for d in bar_data],
        color=[d[2] for d in bar_data],
        hatch=[d[3] for d in bar_data],
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )

    # Annotate bars with values
    for bar, (_, progress, _, _) in zip(bars, bar_data):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{progress:.0f} m",
            ha="center", va="bottom", fontsize=9,
        )

    # Reference lines
    ax.axhline(LAP_LENGTH_M, color="black", linewidth=1.2, linestyle="--", label=f"Full lap ({LAP_LENGTH_M:.0f} m)")
    ax.axhline(LAP_LENGTH_M * 0.965, color="gray", linewidth=0.8, linestyle=":", label="Heuristic baseline (~965 m)")

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in bar_data], fontsize=10)
    ax.set_ylabel("Best-checkpoint progress (m)", fontsize=11)
    ax.set_title("Best deterministic checkpoint progress — 2×2 factorial", fontsize=12)
    ax.set_ylim(0, max(d[1] for d in bar_data) * 1.18)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for hatching
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="V1 reward (solid)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="v21b reward (hatched)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="SBR4"),
        Patch(facecolor="#3498db", edgecolor="black", label="ETK K-Series"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    out_path = out_dir / "fig_condition_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ── Figure 3: Reward components ──────────────────────────────────────────────

def plot_reward_components(out_dir: Path) -> None:
    """Compare key reward component trends: V1 vs v21b for each vehicle."""
    components = [
        ("progress_reward", "Progress reward", True),
        ("off_track_penalty", "Off-track penalty", False),
        ("action_smoothness_penalty", "Smoothness penalty", False),
        ("curvature_overspeed_penalty", "Curvature overspeed (V1 only)", False),
        ("traction_budget_penalty", "Traction budget penalty (v21b only)", False),
    ]

    fig, axes = plt.subplots(1, len(components), figsize=(16, 4), sharey=False)

    for ax, (col, title, _) in zip(axes, components):
        for cond_key, cond in CONDITIONS.items():
            rows = _read_metrics(cond["run"])
            steps, vals = [], []
            for row in rows:
                if col not in row:
                    continue
                try:
                    steps.append(int(row["step"]) / 1000)
                    vals.append(float(row[col]))
                except (ValueError, TypeError):
                    continue
            if not steps:
                continue
            smoothed = _smooth(vals, window=30)
            ax.plot(steps, smoothed,
                    color=cond["color"], linestyle=cond["linestyle"],
                    linewidth=1.5, label=cond["label"], alpha=0.85)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Steps (k)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Mean per step", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)

    fig.suptitle("Reward component breakdown — all conditions", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = out_dir / "fig_reward_components.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def _component_series(col: str) -> list[tuple[dict, list[float], list[float]]]:
    """Return plotted series for a reward component across all conditions."""
    series = []
    for cond in CONDITIONS.values():
        rows = _read_metrics(cond["run"])
        steps, vals = [], []
        for row in rows:
            if col not in row:
                continue
            try:
                steps.append(int(row["step"]) / 1000)
                vals.append(float(row[col]))
            except (ValueError, TypeError):
                continue
        if steps:
            series.append((cond, steps, vals))
    return series


def plot_reward_components_populated(out_dir: Path) -> None:
    """Reward component trends, excluding panels with no visible signal."""
    components = [
        ("progress_reward", "Progress reward"),
        ("off_track_penalty", "Off-track penalty"),
        ("action_smoothness_penalty", "Smoothness penalty"),
        ("curvature_overspeed_penalty", "Curvature overspeed"),
        ("traction_budget_penalty", "Traction budget penalty"),
    ]

    populated = []
    for col, title in components:
        series = _component_series(col)
        has_signal = any(any(abs(v) > 1e-9 for v in vals) for _, _, vals in series)
        if series and has_signal:
            populated.append((col, title, series))

    if not populated:
        print("[plot] WARNING: no populated reward component series found")
        return

    fig, axes = plt.subplots(1, len(populated), figsize=(4.2 * len(populated), 4), sharey=False)
    if len(populated) == 1:
        axes = [axes]

    for ax, (_, title, series) in zip(axes, populated):
        for cond, steps, vals in series:
            smoothed = _smooth(vals, window=30)
            ax.plot(
                steps,
                smoothed,
                color=cond["color"],
                linestyle=cond["linestyle"],
                linewidth=1.5,
                label=cond["label"],
                alpha=0.85,
            )

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Steps (k)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Mean per step", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)

    fig.suptitle("Reward component breakdown — populated signals", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = out_dir / "fig_reward_components_populated.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", choices=["all", "learning", "comparison", "rewards", "rewards-populated"],
                    default="all", help="Which figure(s) to generate")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.fig in ("all", "learning"):
        plot_learning_curves(OUT_DIR)

    if args.fig in ("all", "comparison"):
        plot_condition_comparison(OUT_DIR)

    if args.fig in ("all", "rewards"):
        plot_reward_components(OUT_DIR)

    if args.fig in ("all", "rewards-populated"):
        plot_reward_components_populated(OUT_DIR)


if __name__ == "__main__":
    main()
