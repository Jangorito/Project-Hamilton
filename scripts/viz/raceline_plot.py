# scripts/raceline_plot.py

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from beamng_rl.track.raceline_loader import (
    load_race_path,
    load_items_lines,
    build_sequential_chain,
    derive_race_path,
)


def main() -> None:
    race_file = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\race.race.json")
    items_file = Path(r"C:\Users\Jango\workspace\BeamNG\data\hirochi_track\items.level.json")

    data, pathnodes, segments, start_positions, node_by_id = load_race_path(race_file)
    decal_roads = load_items_lines(items_file)

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.subplots_adjust(right=0.78)

    cmap = plt.get_cmap("tab20")
    num_colors = getattr(cmap, "N", 20)
    decal_handles = []

    highlight_ids = [30, 12, 10, 17]
    highlight_set = set(highlight_ids)

    chain_slices = build_sequential_chain(decal_roads, highlight_ids)
    race_path = derive_race_path(decal_roads, highlight_ids, chain_slices)

    print(f"\nDerived race path: {len(race_path)} coordinates")
    for i, (x, y, z) in enumerate(race_path):
        print(f"  [{i:>4}]  x={x:.2f}  y={y:.2f}  z={z:.2f}")

    for road_idx, road in enumerate(decal_roads):
        nodes = road["nodes"]
        color = cmap(road_idx % num_colors)
        base_style = "--" if road.get("oneWay", False) else "-"

        if road_idx in highlight_set:
            start_i, end_i = chain_slices[road_idx]
            plot_nodes = nodes[start_i : end_i + 1]
            xs = [n[0] for n in plot_nodes]
            ys = [n[1] for n in plot_nodes]
            lw, alpha, zord = 3.0, 1.0, 5
        else:
            xs = [n[0] for n in nodes]
            ys = [n[1] for n in nodes]
            lw, alpha, zord = 3.0, 0.12, 1

        ax.plot(xs, ys, linewidth=lw, alpha=alpha, linestyle=base_style, color=color, zorder=zord)

        try:
            hexcol = mcolors.to_hex(color)
        except Exception:
            hexcol = str(color)

        if nodes:
            sx, sy = nodes[0][0], nodes[0][1]
            lbl_fs, lbl_z = (10, 7) if road_idx in highlight_set else (7, 4)
            ax.text(
                sx - 2,
                sy - 2,
                str(road_idx),
                fontsize=lbl_fs,
                weight="bold",
                color=color,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=lbl_z,
            )

        if road_idx in highlight_set:
            decal_handles.append(
                Line2D([0], [0], color=color, lw=2, linestyle=base_style, label=f"R{road_idx}")
            )

    for road_idx in highlight_ids:
        start_i, end_i = chain_slices[road_idx]
        nodes = decal_roads[road_idx]["nodes"]
        color = cmap(road_idx % num_colors)

        entry_x, entry_y = nodes[start_i][0], nodes[start_i][1]
        ax.scatter([entry_x], [entry_y], s=80, color=color, zorder=10, edgecolors="black", linewidths=0.8)

        exit_x, exit_y = nodes[end_i][0], nodes[end_i][1]
        ax.scatter([exit_x], [exit_y], s=80, color=color, zorder=10, edgecolors="black", linewidths=0.8)

    if decal_handles:
        legend_roads = ax.legend(
            handles=decal_handles,
            title="Decal Roads (R#)",
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
        )
        ax.add_artist(legend_roads)

    px = [node["pos"][0] for node in pathnodes]
    py = [node["pos"][1] for node in pathnodes]
    ax.scatter(px, py, s=60, marker="o", label="Race pathnodes", zorder=5)

    for node in pathnodes:
        x, y, _ = node["pos"]
        ax.text(x + 2, y + 2, node["name"], fontsize=8)

    ax.set_title("Race Path — Sequential Decal Road Chain")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
