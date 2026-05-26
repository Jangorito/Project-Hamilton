import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    # Update this path if needed
    race_file = Path(r"C:\Users\Jango\workspace\BeamNG\Test_Ground\race.race.json")
    # C:\Users\Jango\workspace\BeamNG\Test_Ground\race.race.json
    # C:/Users/Jango/workspace/BeamNG/roads_export

    with race_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pathnodes = data["pathnodes"]
    segments = data["segments"]
    start_positions = data["startPositions"]

    # Build lookup from oldId -> node
    node_by_id = {node["oldId"]: node for node in pathnodes}

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- Plot pathnodes and their radii ---
    for node in pathnodes:
        x, y, z = node["pos"]
        radius = node["radius"]
        name = node["name"]

        ax.scatter(x, y, s=50)
        ax.text(x + 2, y + 2, name, fontsize=8)

        radius_circle = Circle((x, y), radius, fill=False, alpha=0.4)
        ax.add_patch(radius_circle)

    # --- Plot segments between nodes ---
    for segment in segments:
        from_id = segment["from"]
        to_id = segment["to"]

        from_node = node_by_id[from_id]
        to_node = node_by_id[to_id]

        x1, y1, _ = from_node["pos"]
        x2, y2, _ = to_node["pos"]

        ax.plot([x1, x2], [y1, y2], linewidth=2)

    # --- Plot start positions ---
    start_x = []
    start_y = []

    for sp in start_positions:
        # Only plot actual racer starts, not recovery positions
        if sp["name"].startswith("Racer Start"):
            x, y, z = sp["pos"]
            start_x.append(x)
            start_y.append(y)
            ax.text(x + 1, y - 3, sp["name"], fontsize=7)

    ax.scatter(start_x, start_y, marker="s", s=60, label="Start Positions")

    # Highlight the start node
    start_node_id = data["startNode"]
    start_node = node_by_id[start_node_id]
    sx, sy, _ = start_node["pos"]
    ax.scatter([sx], [sy], marker="*", s=200, label="Start Node")

    # Formatting
    ax.set_title("BeamNG Race Path Visualisation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()