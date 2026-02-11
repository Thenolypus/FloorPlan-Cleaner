import os
import re
import json
import shutil
import numpy as np
import cv2
from PIL import Image

from .models import Room, ApartmentUnit
from .flood_fill import FloodFiller
from .room_splitter import mask_to_contour

# SVG scale: 1:100 means 1 SVG unit = 0.1m
SVG_TO_METERS = 0.1

# Contour simplification epsilon in pixels
CONTOUR_EPSILON_PX = 3.0

# Margin around unit bbox for overview outputs (in SVG units)
OVERVIEW_MARGIN_SVG = 5.0


def _extract_boundary_meters(room: Room, filler: FloodFiller):
    """Extract room contour as polygon vertices in meters (SVG x -> X, SVG y -> Z).

    Returns:
        vertices_m: list of [x, z] in meters (centered at origin)
        center_offset: (cx, cz) the midpoint offset subtracted for centering
    """
    contour_px = mask_to_contour(room.flood_mask)

    # Simplify contour to reduce vertex count
    contour_px_f = contour_px.astype(np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(contour_px_f, CONTOUR_EPSILON_PX, closed=True)
    simplified = simplified.reshape(-1, 2)

    # Convert pixel -> SVG -> meters
    vertices_m = []
    for px, py in simplified:
        sx, sy = filler.pixel_to_svg(int(px), int(py))
        mx = sx * SVG_TO_METERS
        mz = -sy * SVG_TO_METERS  # negate: SVG y points down, Z should point up
        vertices_m.append([mx, mz])

    vertices_m = np.array(vertices_m)

    # Compute center (midpoint of bounding box)
    min_coords = vertices_m.min(axis=0)
    max_coords = vertices_m.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    cx, cz = float(center[0]), float(center[1])

    # Center the vertices
    vertices_m[:, 0] -= cx
    vertices_m[:, 1] -= cz

    # Round to 2 decimal places
    vertices_m = np.round(vertices_m, 2).tolist()

    return vertices_m, (cx, cz)


def _compute_unit_bbox_svg(unit_rooms: list[Room]) -> tuple[float, float, float, float]:
    """Compute combined SVG bounding box for all rooms in a unit.

    Returns (x, y, w, h) in SVG coordinates.
    """
    min_x = min(r.bbox_svg[0] for r in unit_rooms)
    min_y = min(r.bbox_svg[1] for r in unit_rooms)
    max_x = max(r.bbox_svg[0] + r.bbox_svg[2] for r in unit_rooms)
    max_y = max(r.bbox_svg[1] + r.bbox_svg[3] for r in unit_rooms)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _export_unit_svg(svg_path: str, unit_rooms: list[Room], output_path: str):
    """Export a cropped SVG showing only the unit's area."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    bx, by, bw, bh = _compute_unit_bbox_svg(unit_rooms)

    # Add margin
    new_vb_x = bx - OVERVIEW_MARGIN_SVG
    new_vb_y = by - OVERVIEW_MARGIN_SVG
    new_vb_w = bw + 2 * OVERVIEW_MARGIN_SVG
    new_vb_h = bh + 2 * OVERVIEW_MARGIN_SVG

    new_vb_str = f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}"
    svg_text = re.sub(r'viewBox="[^"]*"', f'viewBox="{new_vb_str}"', svg_text)

    # Update width/height to match new viewBox aspect ratio
    w_match = re.search(r'width="([\d.]+)mm"', svg_text)
    h_match = re.search(r'height="([\d.]+)mm"', svg_text)
    if w_match and h_match:
        vb_match = re.search(r'viewBox="[^"]*"', svg_text)
        old_vb = vb_match.group(0) if vb_match else ""
        # Recompute mm dimensions proportionally
        old_w_mm = float(w_match.group(1))
        old_h_mm = float(h_match.group(1))
        # Use the same mm-per-unit ratio (should be 1.0 for 1:100 scale SVGs)
        old_vb_match = re.search(r'viewBox="([^"]*)"', svg_text)
        if old_vb_match:
            svg_text = re.sub(r'width="[\d.]+mm"', f'width="{new_vb_w}mm"', svg_text)
            svg_text = re.sub(r'height="[\d.]+mm"', f'height="{new_vb_h}mm"', svg_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_text)


def _export_unit_png(
    filler: FloodFiller, unit_rooms: list[Room], output_path: str
):
    """Export a PNG showing the unit's boundary image with room masks overlaid."""
    # Start with the boundary image as grayscale base
    base = filler.boundary_image.copy()

    # Compute pixel bounding box of all rooms in this unit
    all_ys = []
    all_xs = []
    for room in unit_rooms:
        ys, xs = room.flood_mask.nonzero()
        all_ys.append(ys)
        all_xs.append(xs)
    all_ys = np.concatenate(all_ys)
    all_xs = np.concatenate(all_xs)

    margin_px = 20
    y_min = max(0, int(all_ys.min()) - margin_px)
    y_max = min(base.shape[0], int(all_ys.max()) + margin_px)
    x_min = max(0, int(all_xs.min()) - margin_px)
    x_max = min(base.shape[1], int(all_xs.max()) + margin_px)

    # Convert to RGB
    crop = base[y_min:y_max, x_min:x_max]
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    # Overlay each room mask with a semi-transparent color
    colors = [
        (100, 180, 255),  # blue
        (100, 255, 150),  # green
        (255, 180, 100),  # orange
        (200, 130, 255),  # purple
        (255, 255, 100),  # yellow
        (255, 130, 130),  # red
    ]
    for i, room in enumerate(unit_rooms):
        room_crop = room.flood_mask[y_min:y_max, x_min:x_max]
        color = colors[i % len(colors)]
        overlay = np.zeros_like(rgb)
        overlay[room_crop] = color
        rgb = cv2.addWeighted(rgb, 1.0, overlay, 0.35, 0)

    Image.fromarray(rgb).save(output_path)


def _map_room_type(label: str) -> str:
    """Map internal label to ReSpace room_type."""
    if label == "bedroom":
        return "bedroom"
    elif label == "livingroom/diningroom":
        return "livingroom/diningroom"
    elif label == "all":
        return "all"
    else:
        return "all"


class Exporter:
    def export_all(
        self,
        svg_path: str,
        input_name: str,
        rooms: list[Room],
        units: list[ApartmentUnit],
        filler: FloodFiller,
        height_m: float,
        output_dir: str,
    ):
        base_dir = os.path.join(output_dir, input_name)
        os.makedirs(base_dir, exist_ok=True)

        metadata = {
            "source_svg": input_name + ".svg",
            "svg_scale": "1:100",
            "default_height_m": height_m,
            "units": [],
        }

        for unit in units:
            unit_dir = os.path.join(base_dir, f"unit_{unit.id}")
            os.makedirs(unit_dir, exist_ok=True)

            unit_rooms = [r for r in rooms if r.unit_id == unit.id]

            unit_entry = {
                "unit_id": unit.id,
                "rooms": [],
            }

            for room_idx, room in enumerate(unit_rooms, start=1):
                room_type = _map_room_type(room.label)

                # Extract boundary polygon in meters, centered at origin
                vertices_m, (cx, cz) = _extract_boundary_meters(room, filler)

                # Build bounds_top and bounds_bottom
                # Coordinate system: X = SVG x, Y = up (height), Z = SVG y
                bounds_top = [[v[0], height_m, v[1]] for v in vertices_m]
                bounds_bottom = [[v[0], 0.0, v[1]] for v in vertices_m]

                # Build SSR JSON
                ssr = {
                    "room_type": room_type,
                    "bounds_top": bounds_top,
                    "bounds_bottom": bounds_bottom,
                    "objects": [],
                }

                # Sanitize room type for filename
                type_str = room_type.replace("/", "_")
                filename = f"unit_{unit.id}_room_{room_idx}_{type_str}.json"
                filepath = os.path.join(unit_dir, filename)
                relative_path = f"unit_{unit.id}/{filename}"

                with open(filepath, "w") as f:
                    json.dump(ssr, f, indent=4)

                print(f"Saved: {filepath} | vertices: {len(vertices_m)}")

                room_entry = {
                    "room_id": room_idx,
                    "room_type": room_type,
                    "output_file": relative_path,
                    "center_offset_m": {
                        "x": round(cx, 2),
                        "z": round(cz, 2),
                    },
                    "bbox_in_svg": {
                        "x": round(room.bbox_svg[0], 2),
                        "y": round(room.bbox_svg[1], 2),
                        "width": round(room.bbox_svg[2], 2),
                        "height": round(room.bbox_svg[3], 2),
                    },
                }
                unit_entry["rooms"].append(room_entry)

            # Export unit overview SVG (cropped to unit area)
            overview_svg_name = f"unit_{unit.id}_overview.svg"
            overview_svg_path = os.path.join(unit_dir, overview_svg_name)
            _export_unit_svg(svg_path, unit_rooms, overview_svg_path)
            print(f"Saved: {overview_svg_path}")

            # Export unit overview PNG (boundary image with room masks)
            overview_png_name = f"unit_{unit.id}_overview.png"
            overview_png_path = os.path.join(unit_dir, overview_png_name)
            _export_unit_png(filler, unit_rooms, overview_png_path)
            print(f"Saved: {overview_png_path}")

            unit_entry["overview_svg"] = f"unit_{unit.id}/{overview_svg_name}"
            unit_entry["overview_png"] = f"unit_{unit.id}/{overview_png_name}"

            metadata["units"].append(unit_entry)

        # Copy preprocessed SVG to output root
        svg_dest = os.path.join(base_dir, input_name + "_centered.svg")
        shutil.copy2(svg_path, svg_dest)
        metadata["centered_svg"] = input_name + "_centered.svg"
        print(f"Saved: {svg_dest}")

        # Write metadata
        meta_path = os.path.join(base_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved: {meta_path}")
