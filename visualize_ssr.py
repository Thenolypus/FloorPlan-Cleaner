"""Visualize a room SSR JSON file: floor polygon + openings (doors/windows)."""

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_ssr(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def draw_room(ax, ssr: dict):
    # Extract floor polygon from bounds_bottom (x, z)
    verts = [(v[0], v[2]) for v in ssr["bounds_bottom"]]
    poly = plt.Polygon(verts, closed=True, facecolor="#e8e8e8", edgecolor="black", linewidth=2)
    ax.add_patch(poly)

    # Label vertices
    for i, (x, z) in enumerate(verts):
        ax.plot(x, z, "ko", markersize=3)
        ax.annotate(str(i), (x, z), fontsize=6, ha="center", va="bottom")


def draw_openings(ax, ssr: dict):
    openings = ssr.get("openings", [])
    for op in openings:
        ox, oy, oz = op["pos"]
        sx, sy, sz = op["size"]
        is_door = op["type"] == "door"

        color = "brown" if is_door else "deepskyblue"
        label_prefix = "D" if is_door else "W"

        # Draw as a rectangle centered at (ox, oz) with extents (sx, sz)
        rect = patches.FancyBboxPatch(
            (ox - sx / 2, oz - sz / 2), sx, sz,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", alpha=0.6, linewidth=1.5,
        )
        ax.add_patch(rect)

        # Label with type and y-height
        ax.text(ox, oz, f"{label_prefix}\ny={oy:.1f}m",
                ha="center", va="center", fontsize=7, fontweight="bold")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <room.json> [room2.json ...]")
        sys.exit(1)

    fig, axes = plt.subplots(1, len(sys.argv) - 1, squeeze=False,
                             figsize=(8 * len(sys.argv[1:]), 8))

    for idx, path in enumerate(sys.argv[1:]):
        ax = axes[0][idx]
        ssr = load_ssr(path)

        draw_room(ax, ssr)
        draw_openings(ax, ssr)

        # Formatting
        ax.set_aspect("equal")
        ax.autoscale()
        ax.margins(0.1)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{ssr.get('room_type', 'room')} — {path.split('/')[-1]}")
        ax.grid(True, alpha=0.3)

        # Legend
        door_patch = patches.Patch(color="brown", alpha=0.6, label="Door")
        window_patch = patches.Patch(color="deepskyblue", alpha=0.6, label="Window")
        ax.legend(handles=[door_patch, window_patch], loc="upper right")

        # Stats
        n_doors = sum(1 for o in ssr.get("openings", []) if o["type"] == "door")
        n_windows = sum(1 for o in ssr.get("openings", []) if o["type"] == "window")
        n_verts = len(ssr.get("bounds_bottom", []))
        ax.text(0.02, 0.02,
                f"Vertices: {n_verts} | Doors: {n_doors} | Windows: {n_windows}",
                transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()