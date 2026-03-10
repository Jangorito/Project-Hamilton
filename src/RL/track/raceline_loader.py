# src/RL/track/raceline_loader.py

from __future__ import annotations

import json
from pathlib import Path

Float3 = tuple[float, float, float]


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


def build_sequential_chain(decal_roads, highlight_ids: list[int]):
    """
    For each road in highlight_ids:
      - Find the point on the CURRENT road closest to ANY node on the NEXT road
      - Current road is highlighted up to that point
      - Next road is highlighted starting from ITS node closest to that handoff point

    The last road closes back to the first road using the same logic.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping road_idx -> (start_idx, end_idx), inclusive.
    """
    n = len(highlight_ids)
    slices = {}

    for i in range(n):
        curr_road_idx = highlight_ids[i]
        next_road_idx = highlight_ids[(i + 1) % n]

        curr_nodes = decal_roads[curr_road_idx]["nodes"]
        next_nodes = decal_roads[next_road_idx]["nodes"]

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

        handoff_xy = (curr_nodes[best_curr_idx][0], curr_nodes[best_curr_idx][1])

        best_next_idx, best_dist = 0, float("inf")
        for ni, nn in enumerate(next_nodes):
            dx, dy = nn[0] - handoff_xy[0], nn[1] - handoff_xy[1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_next_idx = ni

        existing_end = slices.get(next_road_idx, (None, len(next_nodes) - 1))[1]
        slices[next_road_idx] = (best_next_idx, existing_end)

    return slices


def derive_race_path(decal_roads, highlight_ids: list[int], chain_slices: dict):
    """
    Concatenate the trimmed node slices in highlight_ids order to produce
    a single ordered list of (x, y, z) coordinates representing the full race path.
    """
    race_path: list[Float3] = []

    for road_idx in highlight_ids:
        start_i, end_i = chain_slices[road_idx]
        nodes = decal_roads[road_idx]["nodes"]

        for node in nodes[start_i : end_i + 1]:
            x, y = node[0], node[1]
            z = node[2] if len(node) > 2 else 0.0
            race_path.append((x, y, z))

    return race_path


def derive_race_path_from_files(
    items_file: Path,
    highlight_ids: list[int],
) -> list[Float3]:
    decal_roads = load_items_lines(items_file)
    chain_slices = build_sequential_chain(decal_roads, highlight_ids)
    return derive_race_path(decal_roads, highlight_ids, chain_slices)