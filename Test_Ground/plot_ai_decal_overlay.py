import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def load_race_path(race_file: Path):
    with race_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pathnodes = data["pathnodes"]
    segments = data["segments"]
    start_positions = data["startPositions"]

    node_by_id = {node["oldId"]: node for node in pathnodes}
    return data, pathnodes, segments, start_positions, node_by_id


def load_items_lines(items_file: Path):
    """
    items.level.json is newline-delimited JSON objects, not one big JSON array.
    We only keep DecalRoad entries inside AIDecalWaypointsGroup.
    """
    decal_roads = []

    with items_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed/non-JSON lines safely
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


def main():
    race_file = Path(r"C:\Users\Jango\workspace\BeamNG\Test_Ground\race.race.json")
    items_file = Path(r"C:\Users\Jango\workspace\BeamNG\Test_Ground\items.level.json")

    data, pathnodes, segments, start_positions, node_by_id = load_race_path(race_file)
    decal_roads = load_items_lines(items_file)

    fig, ax = plt.subplots(figsize=(12, 12))
    # make room on the right for the decal-road legend
    fig.subplots_adjust(right=0.78)

    # --- Plot all invisible AI decal roads using indexed colors and legend ---
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab20")
    num_colors = getattr(cmap, "N", 20)
    decal_handles = []

    # indices to show prominently (customize as needed)
    highlight_ids = {30, 12, 10, 17}
    # highlight_ids = {8, 9, 10, 11}


    for road_idx, road in enumerate(decal_roads):
        nodes = road["nodes"]
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]

        color = cmap(road_idx % num_colors)
        # oneWay roads are often connectors; draw them dashed
        if road.get("oneWay", False):
            base_style = "--"
        else:
            base_style = "-"

        # highlight selected roads, de-emphasize others
        if road_idx in highlight_ids:
            line_style = base_style
            lw = 3.0
            alpha = 1.0
            zord = 5
        else:
            line_style = base_style
            lw = 3
            alpha = 0.12
            zord = 1

        ax.plot(xs, ys, linewidth=lw, alpha=alpha, linestyle=line_style, color=color, zorder=zord)

        # print mapping to console so legend visibility isn't required
        try:
            hexcol = mcolors.to_hex(color)
        except Exception:
            hexcol = str(color)
        print(f"R{road_idx}: color={hexcol}, nodes={len(nodes)}, oneWay={road.get('oneWay', False)}")

        # label the start of the road with its numeric index for easy filtering
        if nodes:
            sx, sy = nodes[0][0], nodes[0][1]
            # make highlight labels larger and more visible
            if road_idx in highlight_ids:
                lbl_fs = 10
                lbl_z = 7
            else:
                lbl_fs = 7
                lbl_z = 4

            ax.text(sx - 2, sy - 2, str(road_idx), fontsize=lbl_fs, weight="bold", color=color,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"), zorder=lbl_z)

        # only add legend handles for highlighted roads to keep legend compact
        if road_idx in highlight_ids:
            decal_handles.append(Line2D([0], [0], color=color, lw=2, linestyle=line_style, label=f"R{road_idx}"))

    # place a separate legend for decal roads so other legend entries stay readable
    if decal_handles:
        # place legend in the reserved right margin so it doesn't get clipped
        legend_roads = ax.legend(handles=decal_handles, title="Decal Roads (R#)", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.add_artist(legend_roads)

    # --- Plot race pathnodes prominently ---
    px = [node["pos"][0] for node in pathnodes]
    py = [node["pos"][1] for node in pathnodes]
    ax.scatter(px, py, s=60, marker="o", label="Race pathnodes", zorder=5)

    for node in pathnodes:
        x, y, _ = node["pos"]
        ax.text(x + 2, y + 2, node["name"], fontsize=8)

    # --- Plot race segments prominently ---
    for segment in segments:
        from_node = node_by_id[segment["from"]]
        to_node = node_by_id[segment["to"]]

        x1, y1, _ = from_node["pos"]
        x2, y2, _ = to_node["pos"]

        # ax.plot([x1, x2], [y1, y2], linewidth=2.5, alpha=0.9, zorder=4)

    # --- Plot start positions ---
    racer_starts = [sp for sp in start_positions if sp["name"].startswith("Racer Start")]
    sx = [sp["pos"][0] for sp in racer_starts]
    sy = [sp["pos"][1] for sp in racer_starts]
    # ax.scatter(sx, sy, marker="s", s=55, label="Start positions", zorder=6)

    # --- Highlight start node ---
    start_node = node_by_id[data["startNode"]]
    snx, sny, _ = start_node["pos"]
    # ax.scatter([snx], [sny], marker="*", s=220, label="Start node", zorder=7)
    # 30, 12, 10, 17
    ax.set_title("Race Path vs AIDecalWaypointsGroup Invisible Splines")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()