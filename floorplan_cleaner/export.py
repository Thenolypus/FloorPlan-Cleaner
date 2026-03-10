import os
import re
import json
import shutil
import math
import numpy as np
import cv2
from PIL import Image

from .models import Room, ApartmentUnit, IfcElement
from .flood_fill import FloodFiller
from .room_splitter import mask_to_contour

# SVG scale: 1:100 means 1 SVG unit = 0.1m
SVG_TO_METERS = 0.1

# Maximum number of vertices for a room polygon
MAX_VERTICES = 16

# Initial epsilon for approxPolyDP (in pixels)
INITIAL_EPSILON_PX = 3.0

# Maximum epsilon to prevent over-simplification (in pixels)
MAX_EPSILON_PX = 30.0

# Margin around unit bbox for overview SVG (in SVG units)
OVERVIEW_MARGIN_SVG = 5.0

# Max perpendicular distance (meters) for a vertex to be considered collinear
COLLINEAR_THRESHOLD_M = 0.20


def _remove_collinear(vertices_m: list[list[float]], threshold: float = COLLINEAR_THRESHOLD_M) -> list[list[float]]:
    """Remove vertices that are nearly collinear with their neighbors.

    For each vertex, compute its perpendicular distance to the line segment
    formed by its predecessor and successor. If below threshold, remove it.
    Iterates until no more removals occur.
    """
    pts = [v[:] for v in vertices_m]
    changed = True
    while changed:
        changed = False
        if len(pts) <= 4:
            break
        keep = []
        n = len(pts)
        for i in range(n):
            prev = np.array(pts[(i - 1) % n])
            curr = np.array(pts[i])
            nxt = np.array(pts[(i + 1) % n])

            edge = nxt - prev
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-9:
                keep.append(pts[i])
                continue

            # Perpendicular distance from curr to line(prev, nxt)
            dist = abs(np.cross(edge, curr - prev)) / edge_len
            if dist >= threshold:
                keep.append(pts[i])
            else:
                changed = True
        pts = keep
    return pts


def _simplify_contour(contour_px: np.ndarray) -> np.ndarray:
    """Simplify a contour to at most MAX_VERTICES vertices.

    Uses cv2.approxPolyDP with increasing epsilon until vertex count <= MAX_VERTICES.
    Falls back to convex hull if epsilon gets too large.
    """
    contour_f = contour_px.astype(np.float32).reshape(-1, 1, 2)

    epsilon = INITIAL_EPSILON_PX
    while epsilon <= MAX_EPSILON_PX:
        simplified = cv2.approxPolyDP(contour_f, epsilon, closed=True)
        if len(simplified) <= MAX_VERTICES:
            return simplified.reshape(-1, 2)
        epsilon *= 1.5

    # If we still have too many vertices after max epsilon,
    # use the last result from max epsilon (best we can do without convex hull)
    simplified = cv2.approxPolyDP(contour_f, MAX_EPSILON_PX, closed=True)
    return simplified.reshape(-1, 2)


def _find_openings_for_room(
    room: Room,
    door_elements: list[IfcElement],
    window_elements: list[IfcElement],
    filler: FloodFiller,
) -> list[dict]:
    """Find doors and windows that overlap with a room.

    Dilates the room mask to catch openings sitting in the wall zone.
    Returns list of dicts with 'type' and 'bbox' (x, y, w, h) in SVG coords.
    """
    h, w = room.flood_mask.shape
    kernel = np.ones((15, 15), dtype=np.uint8)
    dilated = cv2.dilate(room.flood_mask.astype(np.uint8), kernel) > 0

    openings = []
    seen_guids = set()
    for opening_type, elements in [("door", door_elements), ("window", window_elements)]:
        for elem in elements:
            # Deduplicate by IFC GUID (same object has projection + cut layers)
            if elem.ifc_guid and elem.ifc_guid in seen_guids:
                continue

            elem_mask = np.zeros((h, w), dtype=np.uint8)
            for path_coords in elem.paths:
                pts = np.array(
                    [(int((x - filler.viewbox[0]) * filler.scale),
                      int((y - filler.viewbox[1]) * filler.scale))
                     for x, y in path_coords],
                    dtype=np.int32,
                )
                if len(pts) < 2:
                    continue
                cv2.polylines(elem_mask, [pts], isClosed=False, color=255, thickness=3)

            if int(np.sum((elem_mask > 0) & dilated)) > 0:
                # Filter out tiny SVG fragments that aren't real openings
                _, _, ew, eh = elem.bbox
                if max(ew * SVG_TO_METERS, eh * SVG_TO_METERS) < MIN_OPENING_SIZE_M:
                    continue
                if min(ew * SVG_TO_METERS, eh * SVG_TO_METERS) < MIN_OPENING_DEPTH_M:
                    continue
                if elem.ifc_guid:
                    seen_guids.add(elem.ifc_guid)
                openings.append({
                    "type": opening_type,
                    "bbox_svg": elem.bbox,  # (x, y, w, h) in SVG coords
                    "ifc_guid": elem.ifc_guid,
                })

    return openings


def _bbox_overlap(a: tuple, b: tuple) -> float:
    """Return intersection-over-minimum-area ratio for two (x, y, w, h) bboxes."""
    ax, ay, aw, ah = a
    bx, by, bww, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bww)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    min_area = min(aw * ah, bww * bh)
    if min_area < 1e-9:
        return 0.0
    return inter / min_area


# Minimum IoMin overlap ratio to consider two openings as the same physical object
BBOX_OVERLAP_THRESHOLD = 0.5


def _dedup_openings_across_rooms(
    all_room_openings: list[list[dict]],
    rooms: list[Room],
    filler,
) -> list[list[dict]]:
    """Deduplicate openings across rooms in a unit.

    1. GUID dedup: if the same ifc_guid appears in multiple rooms, assign it
       to the room whose mask boundary is closest to the opening center.
    2. Spatial dedup: if two openings (different GUIDs) have heavily overlapping
       bboxes, keep only the one with the larger bbox.
    """
    # --- Step 1: cross-room GUID dedup ---
    # Map guid -> list of (room_idx, opening_idx)
    guid_locations: dict[str, list[tuple[int, int]]] = {}
    for ri, openings in enumerate(all_room_openings):
        for oi, op in enumerate(openings):
            guid = op.get("ifc_guid")
            if guid:
                guid_locations.setdefault(guid, []).append((ri, oi))

    remove_set: set[tuple[int, int]] = set()
    for guid, locs in guid_locations.items():
        if len(locs) <= 1:
            continue
        # Find the opening's SVG center
        ri0, oi0 = locs[0]
        bx, by, bw, bh = all_room_openings[ri0][oi0]["bbox_svg"]
        ocx = int((bx - filler.viewbox[0]) * filler.scale + bw * filler.scale / 2)
        ocy = int((by - filler.viewbox[1]) * filler.scale + bh * filler.scale / 2)

        # Find the room whose mask boundary is closest
        best_ri = locs[0][0]
        best_dist = float("inf")
        for ri, _ in locs:
            mask = rooms[ri].flood_mask
            contour = mask_to_contour(mask)
            if len(contour) == 0:
                continue
            dists = np.sqrt((contour[:, 0] - ocx) ** 2 + (contour[:, 1] - ocy) ** 2)
            d = float(dists.min())
            if d < best_dist:
                best_dist = d
                best_ri = ri

        for ri, oi in locs:
            if ri != best_ri:
                remove_set.add((ri, oi))

    # --- Step 2: spatial dedup within each room ---
    result = []
    for ri, openings in enumerate(all_room_openings):
        kept = []
        for oi, op in enumerate(openings):
            if (ri, oi) in remove_set:
                continue
            # Check against already-kept openings for spatial overlap
            dominated = False
            for ki, kop in enumerate(kept):
                if op["type"] != kop["type"]:
                    continue
                overlap = _bbox_overlap(op["bbox_svg"], kop["bbox_svg"])
                if overlap >= BBOX_OVERLAP_THRESHOLD:
                    # Keep the one with the larger bbox area
                    op_area = op["bbox_svg"][2] * op["bbox_svg"][3]
                    kop_area = kop["bbox_svg"][2] * kop["bbox_svg"][3]
                    if op_area > kop_area:
                        kept[ki] = op  # replace with larger
                    dominated = True
                    break
            if not dominated:
                kept.append(op)
        result.append(kept)

    return result


def _compute_door_rotation(
    door_centroids_m: list[tuple[float, float]], center_x: float, center_z: float
) -> float:
    """Compute rotation angle (radians) to place the door edge at the bottom (-Z).

    Args:
        door_centroids_m: Door positions as (x_m, z_m) in meters (before centering).
        center_x, center_z: The polygon center offset in meters.

    Returns:
        Rotation angle in radians. 0.0 if no doors found.
    """
    if not door_centroids_m:
        return 0.0

    # Average door position relative to room center
    avg_dx = sum(d[0] for d in door_centroids_m) / len(door_centroids_m) - center_x
    avg_dz = sum(d[1] for d in door_centroids_m) / len(door_centroids_m) - center_z

    # Current angle of door direction from center
    current_angle = math.atan2(avg_dz, avg_dx)

    # We want this direction to point to -Z (angle = -pi/2)
    rotation = -math.pi / 2 - current_angle

    # Snap to nearest 90-degree multiple to keep axis-aligned edges
    rotation = round(rotation / (math.pi / 2)) * (math.pi / 2)
    return rotation


def _rotate_vertices(vertices: list[list[float]], angle_rad: float) -> list[list[float]]:
    """Rotate (x, z) vertices around origin by angle_rad."""
    if angle_rad == 0.0:
        return vertices
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated = []
    for x, z in vertices:
        rx = x * cos_a - z * sin_a
        rz = x * sin_a + z * cos_a
        rotated.append([round(rx, 2), round(rz, 2)])
    return rotated


# Minimum opening dimension in meters (filter out SVG path fragments)
MIN_OPENING_SIZE_M = 0.30
# Minimum depth/thickness in meters (filter out thin line fragments)
MIN_OPENING_DEPTH_M = 0.05

# Max perpendicular distance (SVG units) for an opening to be considered on an edge
OPENING_EDGE_DIST_SVG = 5.0


def _apply_boundary_extensions_to_openings(
    openings: list[dict],
    boundary_extensions: list[dict],
) -> list[dict]:
    """Shift openings that lie on extended edges by the extension offset.

    For each boundary extension, find openings whose bbox center is close to
    the extended edge segment, and translate their bbox by normal * offset.
    """
    if not boundary_extensions:
        return openings

    result = []
    for op in openings:
        bx, by, bw, bh = op["bbox_svg"]
        cx = bx + bw / 2
        cy = by + bh / 2
        center = np.array([cx, cy])

        dx_total = 0.0
        dy_total = 0.0

        for ext in boundary_extensions:
            p1 = np.array(ext["edge_p1_svg"])
            p2 = np.array(ext["edge_p2_svg"])
            normal = np.array(ext["normal_svg"])
            offset = ext["offset_svg"]

            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-9:
                continue

            # Project opening center onto the edge line
            t = np.dot(center - p1, edge) / (edge_len * edge_len)

            # Allow some overshoot at endpoints (half of bbox extent)
            half_extent = max(bw, bh) / 2
            t_margin = half_extent / edge_len
            if t < -t_margin or t > 1.0 + t_margin:
                continue

            # Perpendicular distance from center to the edge line
            closest = p1 + t * edge
            perp_dist = np.linalg.norm(center - closest)

            if perp_dist <= OPENING_EDGE_DIST_SVG:
                dx_total += normal[0] * offset
                dy_total += normal[1] * offset

        result.append({
            "type": op["type"],
            "bbox_svg": (bx + dx_total, by + dy_total, bw, bh),
        })

    return result


# Default opening dimensions (meters) for values we can't extract from 2D SVG
WALL_THICKNESS_M = 0.24
DEFAULT_WINDOW_SILL_M = 0.90   # bottom of window above floor
DEFAULT_WINDOW_HEIGHT_M = 1.40
DEFAULT_DOOR_HEIGHT_M = 2.10


def _openings_to_3d(
    openings: list[dict],
    center_offset: tuple[float, float],
    rotation_rad: float,
) -> list[dict]:
    """Convert opening SVG bboxes to 3D pos/size in the room's local coordinate system.

    Uses the same centering and rotation applied to room vertices.
    Returns list of {"type", "pos": [x, y, z], "size": [sx, sy, sz]}.
    """
    cx, cz = center_offset
    result = []
    for op in openings:
        bx, by, bw, bh = op["bbox_svg"]

        # SVG bbox center -> meters (SVG x -> X, SVG y -> -Z)
        mx = (bx + bw / 2) * SVG_TO_METERS - cx
        mz = -(by + bh / 2) * SVG_TO_METERS - cz

        # Size in meters (SVG width -> X extent, SVG height -> Z extent)
        sx = bw * SVG_TO_METERS
        sz = bh * SVG_TO_METERS

        # Apply room rotation to the position
        if rotation_rad != 0.0:
            cos_a = math.cos(rotation_rad)
            sin_a = math.sin(rotation_rad)
            rx = mx * cos_a - mz * sin_a
            rz = mx * sin_a + mz * cos_a
            mx, mz = rx, rz
            # For 90-degree snapped rotations, swap sx/sz when rotated by odd multiples
            steps = round(rotation_rad / (math.pi / 2)) % 4
            if steps in (1, 3):
                sx, sz = sz, sx

        # Vertical placement depends on opening type
        if op["type"] == "window":
            y_center = DEFAULT_WINDOW_SILL_M + DEFAULT_WINDOW_HEIGHT_M / 2
            sy = DEFAULT_WINDOW_HEIGHT_M
        else:  # door
            y_center = DEFAULT_DOOR_HEIGHT_M / 2
            sy = DEFAULT_DOOR_HEIGHT_M

        result.append({
            "type": op["type"],
            "pos": [round(mx, 2), round(y_center, 2), round(mz, 2)],
            "size": [round(sx, 2), round(sy, 2), round(sz, 2)],
        })

    return result


def _extract_boundary_meters(room: Room, filler: FloodFiller):
    """Extract room contour as polygon vertices in meters, centered at origin.

    Coordinate mapping: SVG x -> X, SVG y -> -Z (SVG y points down, Z points up).

    Returns:
        vertices_m: list of [x, z] in meters (centered)
        center_offset: (cx, cz) the midpoint offset subtracted for centering
    """
    contour_px = mask_to_contour(room.flood_mask)
    simplified = _simplify_contour(contour_px)

    # Convert pixel -> SVG -> meters
    vertices_m = []
    for px, py in simplified:
        sx, sy = filler.pixel_to_svg(int(px), int(py))
        mx = sx * SVG_TO_METERS
        mz = -sy * SVG_TO_METERS
        vertices_m.append([mx, mz])

    vertices_m = np.array(vertices_m)

    # Center at bounding box midpoint
    min_coords = vertices_m.min(axis=0)
    max_coords = vertices_m.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    cx, cz = float(center[0]), float(center[1])

    vertices_m[:, 0] -= cx
    vertices_m[:, 1] -= cz

    vertices_m = np.round(vertices_m, 2).tolist()

    # Remove near-collinear vertices (door/window recess artifacts)
    vertices_m = _remove_collinear(vertices_m)

    return vertices_m, (cx, cz)


def _map_room_type_for_output(label: str) -> str:
    """Map internal label to ReSpace room_type for the per-room JSON."""
    if label in ("bedroom", "livingroom", "diningroom", "all", "bathroom", "balcony"):
        return label
    else:
        return "all"


def _compute_unit_bbox_svg(unit_rooms: list[Room]) -> tuple[float, float, float, float]:
    """Compute combined SVG bounding box for all rooms in a unit."""
    min_x = min(r.bbox_svg[0] for r in unit_rooms)
    min_y = min(r.bbox_svg[1] for r in unit_rooms)
    max_x = max(r.bbox_svg[0] + r.bbox_svg[2] for r in unit_rooms)
    max_y = max(r.bbox_svg[1] + r.bbox_svg[3] for r in unit_rooms)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _export_unit_svg(
    svg_path: str,
    unit_rooms: list[Room],
    output_path: str,
    filler: FloodFiller | None = None,
):
    """Export a cropped SVG showing only the unit's area, with room boundary outlines."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    bx, by, bw, bh = _compute_unit_bbox_svg(unit_rooms)

    new_vb_x = bx - OVERVIEW_MARGIN_SVG
    new_vb_y = by - OVERVIEW_MARGIN_SVG
    new_vb_w = bw + 2 * OVERVIEW_MARGIN_SVG
    new_vb_h = bh + 2 * OVERVIEW_MARGIN_SVG

    new_vb_str = f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}"
    svg_text = re.sub(r'viewBox="[^"]*"', f'viewBox="{new_vb_str}"', svg_text)

    # Update width/height to match new viewBox
    if re.search(r'width="[\d.]+mm"', svg_text):
        svg_text = re.sub(r'width="[\d.]+mm"', f'width="{new_vb_w}mm"', svg_text)
        svg_text = re.sub(r'height="[\d.]+mm"', f'height="{new_vb_h}mm"', svg_text)

    # Inject room boundary outlines
    if filler is not None:
        colors = ["#64b4ff", "#64ff96", "#ffb464", "#c882ff", "#ffff64", "#ff8282"]
        outlines = ""
        for i, room in enumerate(unit_rooms):
            contour = mask_to_contour(room.flood_mask)
            contour_f = contour.astype(np.float32).reshape(-1, 1, 2)
            simplified = cv2.approxPolyDP(contour_f, 3.0, closed=True).reshape(-1, 2)

            points_svg = []
            for px, py in simplified:
                sx, sy = filler.pixel_to_svg(int(px), int(py))
                points_svg.append(f"{sx:.2f},{sy:.2f}")

            pts_str = " ".join(points_svg)
            color = colors[i % len(colors)]
            outlines += (
                f'  <polygon points="{pts_str}" '
                f'fill="{color}" fill-opacity="0.15" '
                f'stroke="{color}" stroke-width="0.5" />\n'
            )

        svg_text = svg_text.replace(
            "</svg>",
            f'<g id="room-outlines">\n{outlines}</g>\n</svg>',
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_text)


def _export_unit_png(filler: FloodFiller, unit_rooms: list[Room], output_path: str):
    """Export a PNG showing the unit's boundary image with room masks overlaid."""
    base = filler.boundary_image.copy()

    # Compute pixel bounding box of all rooms
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

    crop = base[y_min:y_max, x_min:x_max]
    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    colors = [
        (100, 180, 255),
        (100, 255, 150),
        (255, 180, 100),
        (200, 130, 255),
        (255, 255, 100),
        (255, 130, 130),
    ]
    for i, room in enumerate(unit_rooms):
        room_crop = room.flood_mask[y_min:y_max, x_min:x_max]
        color = colors[i % len(colors)]
        overlay = np.zeros_like(rgb)
        overlay[room_crop] = color
        rgb = cv2.addWeighted(rgb, 1.0, overlay, 0.35, 0)

    Image.fromarray(rgb).save(output_path)


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
        door_elements: list[IfcElement] | None = None,
        window_elements: list[IfcElement] | None = None,
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

            # Collect openings for all rooms, then deduplicate across rooms
            all_room_openings = []
            for room in unit_rooms:
                room_openings = _find_openings_for_room(
                    room,
                    door_elements or [],
                    window_elements or [],
                    filler,
                )
                room_openings = _apply_boundary_extensions_to_openings(
                    room_openings, room.boundary_extensions,
                )
                all_room_openings.append(room_openings)
            all_room_openings = _dedup_openings_across_rooms(
                all_room_openings, unit_rooms, filler,
            )

            for room_idx, room in enumerate(unit_rooms, start=1):
                output_room_type = _map_room_type_for_output(room.label)

                # Extract boundary polygon in meters, centered at origin
                vertices_m, (cx, cz) = _extract_boundary_meters(room, filler)

                room_openings = all_room_openings[room_idx - 1]

                # Extract door centroids for rotation computation
                door_centroids_m = []
                for op in room_openings:
                    if op["type"] == "door":
                        bx, by, bw, bh = op["bbox_svg"]
                        door_centroids_m.append((
                            (bx + bw / 2) * SVG_TO_METERS,
                            -(by + bh / 2) * SVG_TO_METERS,
                        ))

                rotation_rad = _compute_door_rotation(door_centroids_m, cx, cz)
                if rotation_rad != 0.0:
                    vertices_m = _rotate_vertices(vertices_m, rotation_rad)

                # Convert openings to 3D in room-local coords
                openings_3d = _openings_to_3d(
                    room_openings, (cx, cz), rotation_rad,
                )

                # Build bounds_top and bounds_bottom
                # X = SVG x, Y = up (height), Z = -SVG y
                bounds_top = [[v[0], height_m, v[1]] for v in vertices_m]
                bounds_bottom = [[v[0], 0.0, v[1]] for v in vertices_m]

                ssr = {
                    "room_type": output_room_type,
                    "bounds_top": bounds_top,
                    "bounds_bottom": bounds_bottom,
                    "objects": [],
                    "openings": openings_3d,
                }

                # Use the output room type for the filename
                type_str = output_room_type.replace("/", "_")
                filename = f"unit_{unit.id}_room_{room_idx}_{type_str}.json"
                filepath = os.path.join(unit_dir, filename)
                relative_path = f"unit_{unit.id}/{filename}"

                with open(filepath, "w") as f:
                    json.dump(ssr, f, indent=4)

                n_doors = sum(1 for o in openings_3d if o["type"] == "door")
                n_windows = sum(1 for o in openings_3d if o["type"] == "window")
                print(f"Saved: {filepath} | vertices: {len(vertices_m)} | rotation: {math.degrees(rotation_rad):.1f} deg | doors: {n_doors} | windows: {n_windows}")

                # Metadata keeps the original label for reconstruction
                room_entry = {
                    "room_id": room_idx,
                    "room_type": room.label or "unlabelled",
                    "output_file": relative_path,
                    "center_offset_m": {
                        "x": round(cx, 2),
                        "z": round(cz, 2),
                    },
                    "rotation_rad": round(rotation_rad, 6),
                    "bbox_in_svg": {
                        "x": round(room.bbox_svg[0], 2),
                        "y": round(room.bbox_svg[1], 2),
                        "width": round(room.bbox_svg[2], 2),
                        "height": round(room.bbox_svg[3], 2),
                    },
                }
                unit_entry["rooms"].append(room_entry)

            # Export unit overview SVG
            overview_svg_name = f"unit_{unit.id}_overview.svg"
            overview_svg_path = os.path.join(unit_dir, overview_svg_name)
            _export_unit_svg(svg_path, unit_rooms, overview_svg_path, filler=filler)
            print(f"Saved: {overview_svg_path}")

            # Export unit overview PNG
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
