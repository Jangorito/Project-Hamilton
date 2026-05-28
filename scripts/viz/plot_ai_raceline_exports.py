"""Overlay exported AI racelines against the canonical Hirochi centreline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CENTRELINE_PATH = REPO_ROOT / "data" / "hirochi_track" / "centreline_resampled_2_0m.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AI raceline JSON exports.")
    parser.add_argument(
        "racelines",
        nargs="*",
        type=Path,
        help="Raceline JSON files to overlay. Defaults to data/hirochi_track/ai_*_raceline_*.json.",
    )
    parser.add_argument("--centreline", type=Path, default=CENTRELINE_PATH)
    parser.add_argument("--save", type=Path, default=None, help="Optional image path to save.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    racelines = args.racelines or sorted(
        (REPO_ROOT / "data" / "hirochi_track").glob("ai_*_raceline_*.json")
    )
    if not racelines:
        raise SystemExit("No raceline JSONs found.")

    fig, ax = plt.subplots(figsize=(12, 12))
    _plot_path(ax, _load_points(args.centreline), "centreline", "black", linewidth=1.5, alpha=0.7)

    for index, path in enumerate(racelines):
        label = path.stem
        color = f"C{index % 10}"
        linewidth = 2.0 if "driven" in path.stem else 1.5
        linestyle = "-" if "driven" in path.stem else "--"
        _plot_path(
            ax,
            _load_points(path),
            label,
            color,
            linewidth=linewidth,
            alpha=0.9,
            linestyle=linestyle,
        )

    ax.set_title("Hirochi AI Raceline Exports")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved plot: {args.save}")
    else:
        plt.show()


def _load_points(path: Path) -> list[Mapping[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["points"])


def _plot_path(
    ax,
    points: list[Mapping[str, float]],
    label: str,
    color: str,
    *,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
) -> None:
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    ax.plot(xs, ys, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


if __name__ == "__main__":
    main()
