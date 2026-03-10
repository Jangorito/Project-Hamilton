# src/RL/track/geometry.py
from __future__ import annotations

Float3 = tuple[float, float, float]

def resample_polyline(points: list[Float3], spacing: float) -> list[Float3]:
    """
    Resample a polyline to a uniform arc-length spacing.

    Walks the input polyline segment by segment, placing a new sample point
    every `spacing` metres along the cumulative arc length. The first input
    point is always preserved; the last point is appended if it would
    otherwise be missed (i.e. the tail segment is shorter than `spacing`).

    Parameters
    ----------
    points:
        Ordered (x, y, z) coordinates of the original polyline.
    spacing:
        Desired distance in metres between consecutive output points.

    Returns
    -------
    Resampled list of (x, y, z) tuples at uniform `spacing` intervals.

    **How the carry works** — it's the key to uniform spacing across segment boundaries:

    segment A (8m)   |----+----+----|   spacing=3m
                        ^    ^    ^--- carry=1m into segment B
    segment B (7m)   |-+----+----+--|
                    ^ starts at 3-1=2m in, not 3m

    """
    if len(points) < 2:
        return list(points)
    
    if spacing <= 0:
        raise ValueError(f"spacing must be > 0, got {spacing!r}")

    resampled: list[Float3] = [points[0]]
    carry = 0.0  # leftover distance from the previous segment

    for i in range(len(points) - 1):
        ax, ay, az = points[i]
        bx, by, bz = points[i + 1]

        dx, dy, dz = bx - ax, by - ay, bz - az
        seg_len = (dx * dx + dy * dy + dz * dz) ** 0.5

        if seg_len == 0.0:
            continue

        # Distance along this segment to the first new sample
        dist_to_next = spacing - carry

        while dist_to_next <= seg_len:
            t = dist_to_next / seg_len
            resampled.append((
                ax + t * dx,
                ay + t * dy,
                az + t * dz,
            ))
            dist_to_next += spacing

        # How far past the last sample we travelled on this segment
        carry = seg_len - (dist_to_next - spacing)

    # Always include the final point if it's meaningfully far from the last sample
    last = resampled[-1]
    end  = points[-1]
    tail = ((end[0] - last[0]) ** 2 + (end[1] - last[1]) ** 2 + (end[2] - last[2]) ** 2) ** 0.5
    if tail > spacing * 0.1:
        resampled.append(end)

    return resampled

def cumulative_arc_lengths(points: list[Float3]) -> list[float]:
    ...