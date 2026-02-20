"""
Reproject top-down rendered room scenes back onto the original floorplan.

Reads metadata.json for room positions, retrieval result JSONs for room bounds,
and top-down render images. Composites them onto the unit overview PNG at the
correct scale and position.

The render viewport is computed to match viz.py's setup_camera():
  - PerspectiveCamera with yfov=pi/4
  - use_dynamic_zoom=True: camera_height = max(2.0, (limiting_span/2)/tan(pi/8) + 2.5)
  - limiting_span = max(x_span, z_span) from bounds_bottom
  - Top view is flipped vertically

Usage:
    python visualize_reprojection.py \
        --metadata "output/03OG PLAN/metadata.json" \
        --scenes input/generated_scenes/13feb_settings_retrieval \
        --unit 1 \
        --variant greedy \
        --output reprojection_result.png
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

SVG_TO_METERS = 0.1
HALF_FOV = math.pi / 8  # yfov=pi/4, half = pi/8


def load_metadata(metadata_path: str) -> dict:
    with open(metadata_path) as f:
        return json.load(f)


def get_unit_svg_bbox(unit_entry: dict) -> tuple[float, float, float, float]:
    """Compute the combined SVG bounding box of all rooms in a unit.
    Returns (min_x, min_y, max_x, max_y) in SVG coords.
    """
    rooms = unit_entry["rooms"]
    min_x = min(r["bbox_in_svg"]["x"] for r in rooms)
    min_y = min(r["bbox_in_svg"]["y"] for r in rooms)
    max_x = max(r["bbox_in_svg"]["x"] + r["bbox_in_svg"]["width"] for r in rooms)
    max_y = max(r["bbox_in_svg"]["y"] + r["bbox_in_svg"]["height"] for r in rooms)
    return min_x, min_y, max_x, max_y


def compute_render_viewport_half(bounds_bottom: list[list[float]]) -> float:
    """Compute the half-extent of the render viewport in meters.

    Replicates the camera setup from viz.py setup_camera() with
    use_dynamic_zoom=True, camera_height=None, yfov=pi/4.

    The top-down render covers a square viewport of
    [-viewport_half, +viewport_half] in both X and Z, centered at origin.
    """
    xs = [v[0] for v in bounds_bottom]
    zs = [v[2] for v in bounds_bottom]
    x_span = max(xs) - min(xs)
    z_span = max(zs) - min(zs)

    scene_aspect = x_span / max(z_span, 1e-5)
    limiting_span = x_span if scene_aspect > 1.0 else z_span

    required_distance = (limiting_span / 2) / math.tan(HALF_FOV)
    camera_height = max(2.0, required_distance + 2.5)
    viewport_half = camera_height * math.tan(HALF_FOV)
    return viewport_half


def room_local_to_svg(x: float, z: float, cx: float, cz: float,
                      rotation_rad: float) -> tuple[float, float]:
    """Convert room-local meter coords (x, z) back to SVG coords.

    Reverses: center, rotate, meters-to-SVG.
    Export did: SVG -> meters -> center -> rotate
    So inverse is: un-rotate -> un-center -> meters-to-SVG
    """
    if rotation_rad != 0.0:
        cos_a = math.cos(-rotation_rad)
        sin_a = math.sin(-rotation_rad)
        x_ur = x * cos_a - z * sin_a
        z_ur = x * sin_a + z * cos_a
    else:
        x_ur, z_ur = x, z

    x_m = x_ur + cx
    z_m = z_ur + cz

    svg_x = x_m / SVG_TO_METERS
    svg_y = -z_m / SVG_TO_METERS
    return svg_x, svg_y


def svg_to_overview_px(svg_x: float, svg_y: float,
                       svg_min_x: float, svg_min_y: float,
                       scale: float, margin: float) -> tuple[float, float]:
    """Convert SVG coords to overview PNG pixel coords."""
    px = (svg_x - svg_min_x) * scale + margin
    py = (svg_y - svg_min_y) * scale + margin
    return px, py


def compute_render_to_overview_affine(
    bounds_bottom: list[list[float]],
    render_w: int, render_h: int,
    cx: float, cz: float, rotation_rad: float,
    svg_min_x: float, svg_min_y: float,
    svg_scale: float, margin: float,
) -> np.ndarray:
    """Compute the 2x3 affine matrix from render pixels to overview pixels.

    The render viewport is a square [-vh, +vh] x [-vh, +vh] in room-local
    X-Z space, where vh = camera_height * tan(pi/8). The image is flipped
    vertically, so pixel (0,0) maps to (-vh, +vh) in (X, Z).
    """
    vh = compute_render_viewport_half(bounds_bottom)

    # Three source points in render pixel space
    src_pts = np.float32([
        [0, 0],                # top-left of render
        [render_w, 0],         # top-right of render
        [0, render_h],         # bottom-left of render
    ])

    # After vertical flip, the pixel-to-room mapping is:
    #   pixel (0,0) top-left     -> room (-vh, +vh)  [min X, max Z]
    #   pixel (W,0) top-right    -> room (+vh, +vh)  [max X, max Z]
    #   pixel (0,H) bottom-left  -> room (-vh, -vh)  [min X, min Z]
    room_pts = [
        (-vh, +vh),
        (+vh, +vh),
        (-vh, -vh),
    ]

    # Map each through the full transform chain to overview pixels
    dst_pts = []
    for rx, rz in room_pts:
        sx, sy = room_local_to_svg(rx, rz, cx, cz, rotation_rad)
        px, py = svg_to_overview_px(sx, sy, svg_min_x, svg_min_y, svg_scale, margin)
        dst_pts.append([px, py])
    dst_pts = np.float32(dst_pts)

    return cv2.getAffineTransform(src_pts, dst_pts)


def create_render_mask(render_img: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Create a binary mask of non-white (non-background) pixels."""
    gray = cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return mask


def find_retrieval_json(scenes_dir: str, unit_id: int, room_id: int,
                        room_type: str) -> str | None:
    """Find the retrieval results JSON for a given room."""
    type_str = room_type.replace("/", "_").split("/")[0]
    if type_str == "livingroom_diningroom":
        type_str = "livingroom"
    candidates = [
        f"retrieval_results_unit_{unit_id}_room_{room_id}_{type_str}.json",
    ]
    for name in candidates:
        path = os.path.join(scenes_dir, name)
        if os.path.exists(path):
            return path
    return None


def find_render_image(scenes_dir: str, unit_id: int, room_id: int,
                      room_type: str, variant: str) -> str | None:
    """Find the top-down render image for a given room."""
    type_str = room_type.replace("/", "_").split("/")[0]
    if type_str == "livingroom_diningroom":
        type_str = "livingroom"
    dir_name = f"render_unit_{unit_id}_room_{room_id}_{type_str}"
    filename = f"{variant}_unit_{unit_id}_room_{room_id}_{type_str}.jpg"
    path = os.path.join(scenes_dir, dir_name, "top", filename)
    if os.path.exists(path):
        return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Reproject rendered room scenes onto the floorplan overview."
    )
    parser.add_argument("--metadata", required=True,
                        help="Path to metadata.json")
    parser.add_argument("--scenes", required=True,
                        help="Path to generated scenes directory with retrieval results and renders")
    parser.add_argument("--unit", type=int, required=True,
                        help="Unit ID to visualize")
    parser.add_argument("--variant", default="greedy", choices=["greedy", "stochastic"],
                        help="Which retrieval variant to use (default: greedy)")
    parser.add_argument("--output", default="reprojection_result.png",
                        help="Output image path")
    parser.add_argument("--opacity", type=float, default=0.85,
                        help="Opacity of the rendered overlay (0-1, default: 0.85)")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    base_dir = os.path.dirname(args.metadata)

    unit_entry = None
    for u in metadata["units"]:
        if u["unit_id"] == args.unit:
            unit_entry = u
            break
    if unit_entry is None:
        print(f"Unit {args.unit} not found in metadata.")
        sys.exit(1)

    overview_path = os.path.join(base_dir, unit_entry["overview_png"])
    overview = cv2.imread(overview_path)
    if overview is None:
        print(f"Could not load overview: {overview_path}")
        sys.exit(1)

    oh, ow = overview.shape[:2]

    svg_min_x, svg_min_y, svg_max_x, svg_max_y = get_unit_svg_bbox(unit_entry)
    svg_w = svg_max_x - svg_min_x
    svg_h = svg_max_y - svg_min_y

    margin_px = 20.0
    scale_x = (ow - 2 * margin_px) / svg_w
    scale_y = (oh - 2 * margin_px) / svg_h
    svg_scale = (scale_x + scale_y) / 2.0

    print(f"Overview: {ow}x{oh} px")
    print(f"Unit SVG bbox: ({svg_min_x:.1f}, {svg_min_y:.1f}) - ({svg_max_x:.1f}, {svg_max_y:.1f})")
    print(f"SVG-to-pixel scale: {svg_scale:.3f} px/SVG-unit")

    result = overview.copy()

    rooms_processed = 0
    for room_entry in unit_entry["rooms"]:
        room_id = room_entry["room_id"]
        room_type = room_entry["room_type"]
        cx = room_entry["center_offset_m"]["x"]
        cz = room_entry["center_offset_m"]["z"]
        rotation_rad = room_entry.get("rotation_rad", 0.0)

        retrieval_path = find_retrieval_json(
            args.scenes, args.unit, room_id, room_type)
        if retrieval_path is None:
            print(f"  Room {room_id} ({room_type}): no retrieval JSON found, skipping")
            continue

        render_path = find_render_image(
            args.scenes, args.unit, room_id, room_type, args.variant)
        if render_path is None:
            print(f"  Room {room_id} ({room_type}): no render image found, skipping")
            continue

        with open(retrieval_path) as f:
            retrieval = json.load(f)
        scene = retrieval[args.variant]
        bounds_bottom = scene["bounds_bottom"]

        render_img = cv2.imread(render_path)
        if render_img is None:
            print(f"  Room {room_id} ({room_type}): could not load render, skipping")
            continue

        rh, rw = render_img.shape[:2]
        vh = compute_render_viewport_half(bounds_bottom)

        # Compute affine using exact camera viewport
        affine = compute_render_to_overview_affine(
            bounds_bottom, rw, rh,
            cx, cz, rotation_rad,
            svg_min_x, svg_min_y, svg_scale, margin_px,
        )

        warped = cv2.warpAffine(render_img, affine, (ow, oh),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        mask = create_render_mask(warped)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(bool)

        blended = result.copy()
        blended[mask_3ch] = (
            args.opacity * warped[mask_3ch].astype(float) +
            (1 - args.opacity) * result[mask_3ch].astype(float)
        ).astype(np.uint8)
        result = blended

        n_objects = len(scene.get("objects", []))
        print(f"  Room {room_id} ({room_type}): viewport_half={vh:.2f}m, cam_h={vh/math.tan(HALF_FOV):.1f}m, {n_objects} objects")
        rooms_processed += 1

    if rooms_processed == 0:
        print("No rooms were projected. Check your --scenes path and file naming.")
        sys.exit(1)

    cv2.imwrite(args.output, result)
    print(f"\nSaved: {args.output} ({rooms_processed} rooms projected)")


if __name__ == "__main__":
    main()
