import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D


def load_race_path(race_file: Path):
    with race_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    pathnodes = data["pathnodes"]
    segments = data["segments"]
    start_positions = data["startPositions"]
    node_by_id = {node["oldId"]: node for node in pathnodes}
    return data, pathnodes, segments, start_positions, node_by_id

def load_items_lines(items_file: Path):
    decal_roads = []
    with items_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("class") != "DecalRoad":
                continue
            if obj.get("__parent") != "AIDecalWaypointsGroup":
                continue
            if obj.get("material") != "road_invisible":
                continue
            if "nodes" not in obj:
                continue
            decal_roads.append(obj)
    return decal_roads

def find_closest_node_index(nodes, target_xy):
    """Return the index in `nodes` whose XY is closest to target_xy."""
    tx, ty = target_xy
    best_idx, best_dist = 0, float("inf")
    for i, n in enumerate(nodes):
        dx, dy = n[0] - tx, n[1] - ty
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, best_dist

def build_sequential_chain(decal_roads, highlight_ids: list):
    """
    For each road in highlight_ids:
      - Find the point on the CURRENT road closest to ANY node on the NEXT road
      - Current road is highlighted up to that point
      - Next road is highlighted starting from ITS node closest to that handoff point
    The last road closes back to the first road using the same logic.

    Returns a dict  road_idx -> (start_idx, end_idx)  — inclusive slice indices.
    """
    n = len(highlight_ids)
    slices = {}

    for i in range(n):
        curr_road_idx = highlight_ids[i]
        next_road_idx = highlight_ids[(i + 1) % n]

        curr_nodes = decal_roads[curr_road_idx]["nodes"]
        next_nodes = decal_roads[next_road_idx]["nodes"]

        # Find the node on the CURRENT road closest to any node on the NEXT road
        best_curr_idx, best_dist = 0, float("inf")
        for ci, cn in enumerate(curr_nodes):
            for nn in next_nodes:
                dx, dy = cn[0] - nn[0], cn[1] - nn[1]
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_curr_idx = ci

        slices[curr_road_idx] = slices.get(curr_road_idx, (0, None))
        start_idx = slices[curr_road_idx][0]
        slices[curr_road_idx] = (start_idx, best_curr_idx)

        # Next road starts from ITS node closest to the handoff point on curr road
        handoff_xy = (curr_nodes[best_curr_idx][0], curr_nodes[best_curr_idx][1])
        best_next_idx, best_dist = 0, float("inf")
        for ni, nn in enumerate(next_nodes):
            dx, dy = nn[0] - handoff_xy[0], nn[1] - handoff_xy[1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_next_idx = ni

        # Set next road's start (its end will be determined in the next iteration)
        existing_end = slices.get(next_road_idx, (None, len(next_nodes) - 1))[1]
        slices[next_road_idx] = (best_next_idx, existing_end)

    return slices

def derive_race_path(decal_roads, highlight_ids: list, chain_slices: dict):
    """
    Concatenates the trimmed node slices in highlight_ids order to produce
    a single ordered list of (x, y, z) coordinates representing the full race path.
    """
    race_path = []

    for road_idx in highlight_ids:
        start_i, end_i = chain_slices[road_idx]
        nodes = decal_roads[road_idx]["nodes"]
        for node in nodes[start_i: end_i + 1]:
            x, y = node[0], node[1]
            z = node[2] if len(node) > 2 else 0.0
            race_path.append((x, y, z))

    return race_path

# path between #68 & #69 needs interpolating

def main():
    race_file = Path(r"C:\Users\Jango\workspace\BeamNG\Test_Ground\race.race.json")
    items_file = Path(r"C:\Users\Jango\workspace\BeamNG\Test_Ground\items.level.json")

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

    # Print or export
    print(f"\nDerived race path: {len(race_path)} coordinates")
    for i, (x, y, z) in enumerate(race_path):
        print(f"  [{i:>4}]  x={x:.2f}  y={y:.2f}  z={z:.2f}")

    for road_idx, road in enumerate(decal_roads):
        nodes = road["nodes"]
        color = cmap(road_idx % num_colors)
        base_style = "--" if road.get("oneWay", False) else "-"

        if road_idx in highlight_set:
            start_i, end_i = chain_slices[road_idx]
            plot_nodes = nodes[start_i: end_i + 1]
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
        # print(f"R{road_idx}: color={hexcol}, nodes={len(nodes)}, oneWay={road.get('oneWay', False)}")

        if nodes:
            sx, sy = nodes[0][0], nodes[0][1]
            lbl_fs, lbl_z = (10, 7) if road_idx in highlight_set else (7, 4)
            ax.text(sx - 2, sy - 2, str(road_idx), fontsize=lbl_fs, weight="bold", color=color,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"), zorder=lbl_z)

        if road_idx in highlight_set:
            decal_handles.append(
                Line2D([0], [0], color=color, lw=2, linestyle=base_style, label=f"R{road_idx}")
            )

    # Draw handoff dots at each chain join point
    for road_idx in highlight_ids:
        start_i, end_i = chain_slices[road_idx]
        nodes = decal_roads[road_idx]["nodes"]
        color = cmap(road_idx % num_colors)
        # Entry dot
        ex, ey = nodes[start_i][0], nodes[start_i][1]
        ax.scatter([ex], [ey], s=80, color=color, zorder=10, edgecolors="black", linewidths=0.8)
        # Exit dot
        ex, ey = nodes[end_i][0], nodes[end_i][1]
        ax.scatter([ex], [ey], s=80, color=color, zorder=10, edgecolors="black", linewidths=0.8)

    if decal_handles:
        legend_roads = ax.legend(handles=decal_handles, title="Decal Roads (R#)",
                                 fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
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