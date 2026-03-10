"""
Reproject top-down rendered room scenes back onto the original floorplan.

Reads metadata.json for room positions, generated_scene.json for room bounds,
and top-down render images. Composites them onto the unit overview PNG at the
correct scale and position.

The render viewport is computed to match viz.py's setup_camera():
  - PerspectiveCamera with yfov=pi/4
  - use_dynamic_zoom=True: camera_height = max(2.0, (limiting_span/2)/tan(pi/8) + 2.5)
  - limiting_span = max(x_span, z_span) from bounds_bottom
  - Top view is flipped vertically

Usage:
    python visualize_reprojection.py --floorplan input/generated_scenes/03OG_PLAN
    python visualize_reprojection.py --floorplan input/generated_scenes/03OG_PLAN --unit 1
"""

import argparse
import json
import math
import os
import re
import sys

import cv2
import numpy as np

SVG_TO_METERS = 0.1
HALF_FOV = math.pi / 8  # yfov=pi/4, half = pi/8


def load_metadata(metadata_path: str) -> dict:
    with open(metadata_path) as f:
        return json.load(f)


def parse_svg_viewbox(svg_path: str) -> tuple[float, float, float, float]:
    """Parse viewBox from an SVG file. Returns (x, y, width, height)."""
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read(2000)  # viewBox is always near the top
    match = re.search(r'viewBox="([^"]*)"', svg_text)
    if not match:
        raise ValueError(f"No viewBox found in {svg_path}")
    parts = match.group(1).split()
    return tuple(float(p) for p in parts)


def compute_render_viewport_half(bounds_bottom: list[list[float]]) -> float:
    """Compute the half-extent of the render viewport in meters.

    Replicates the camera setup from viz.py setup_camera() with
    use_dynamic_zoom=True, camera_height=None, yfov=pi/4.
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

    Export pipeline: SVG -> meters -> center -> rotate
    Inverse: un-rotate -> un-center -> meters-to-SVG
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
                       vb_x: float, vb_y: float,
                       vb_w: float, vb_h: float,
                       png_w: int, png_h: int) -> tuple[float, float]:
    """Convert SVG coords to overview PNG pixel coords using SVG viewBox."""
    px = (svg_x - vb_x) / vb_w * png_w
    py = (svg_y - vb_y) / vb_h * png_h
    return px, py


def compute_render_to_overview_affine(
    bounds_bottom: list[list[float]],
    render_w: int, render_h: int,
    cx: float, cz: float, rotation_rad: float,
    vb_x: float, vb_y: float, vb_w: float, vb_h: float,
    png_w: int, png_h: int,
) -> np.ndarray:
    """Compute the 2x3 affine matrix from render pixels to overview pixels.

    The render viewport is a square [-vh, +vh] x [-vh, +vh] in room-local
    X-Z space. After the vertical flip in viz.py:
      pixel (0,0) top-left     -> room (-vh, +vh)  [min X, max Z]
      pixel (W,0) top-right    -> room (+vh, +vh)  [max X, max Z]
      pixel (0,H) bottom-left  -> room (-vh, -vh)  [min X, min Z]
    """
    vh = compute_render_viewport_half(bounds_bottom)

    src_pts = np.float32([
        [0, 0],
        [render_w, 0],
        [0, render_h],
    ])

    room_pts = [
        (-vh, +vh),
        (+vh, +vh),
        (-vh, -vh),
    ]

    dst_pts = []
    for rx, rz in room_pts:
        sx, sy = room_local_to_svg(rx, rz, cx, cz, rotation_rad)
        px, py = svg_to_overview_px(sx, sy, vb_x, vb_y, vb_w, vb_h, png_w, png_h)
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


def room_type_to_filename(room_type: str) -> str:
    """Convert room_type from metadata to the filename convention."""
    t = room_type.replace("/", "_").split("/")[0]
    if t == "livingroom_diningroom":
        t = "livingroom"
    return t


def find_generated_scene(floorplan_dir: str, unit_id: int, room_id: int,
                         room_type: str) -> str | None:
    """Find the generated_scene.json for a room."""
    t = room_type_to_filename(room_type)
    path = os.path.join(
        floorplan_dir, f"unit_{unit_id}", "output",
        f"unit_{unit_id}_room_{room_id}_{t}",
        "generated_scene.json"
    )
    return path if os.path.exists(path) else None


def find_render_image(floorplan_dir: str, unit_id: int, room_id: int,
                      room_type: str) -> str | None:
    """Find the top-down render image for a room."""
    t = room_type_to_filename(room_type)
    name = f"unit_{unit_id}_room_{room_id}_{t}"
    path = os.path.join(
        floorplan_dir, f"unit_{unit_id}", "output", name,
        "render", "top", f"{name}.jpg"
    )
    return path if os.path.exists(path) else None


def process_unit(floorplan_dir: str, metadata: dict, unit_entry: dict,
                 opacity: float) -> np.ndarray | None:
    """Process a single unit: reproject all room renders onto the overview."""
    unit_id = unit_entry["unit_id"]
    base_dir = os.path.dirname(os.path.join(floorplan_dir, "metadata.json"))

    # Load overview PNG
    overview_path = os.path.join(base_dir, unit_entry["overview_png"])
    overview = cv2.imread(overview_path)
    if overview is None:
        print(f"  Could not load overview: {overview_path}")
        return None
    oh, ow = overview.shape[:2]

    # Parse overview SVG viewBox for coordinate mapping
    svg_path = os.path.join(base_dir, unit_entry["overview_svg"])
    if not os.path.exists(svg_path):
        print(f"  Could not find overview SVG: {svg_path}")
        return None
    vb_x, vb_y, vb_w, vb_h = parse_svg_viewbox(svg_path)

    print(f"  Overview: {ow}x{oh} px, viewBox: ({vb_x:.1f}, {vb_y:.1f}, {vb_w:.1f}, {vb_h:.1f})")

    result = overview.copy()
    rooms_processed = 0

    for room_entry in unit_entry["rooms"]:
        room_id = room_entry["room_id"]
        room_type = room_entry["room_type"]
        cx = room_entry["center_offset_m"]["x"]
        cz = room_entry["center_offset_m"]["z"]
        rotation_rad = room_entry.get("rotation_rad", 0.0)

        scene_path = find_generated_scene(floorplan_dir, unit_id, room_id, room_type)
        if scene_path is None:
            print(f"    Room {room_id} ({room_type}): no generated_scene.json, skipping")
            continue

        render_path = find_render_image(floorplan_dir, unit_id, room_id, room_type)
        if render_path is None:
            print(f"    Room {room_id} ({room_type}): no render image, skipping")
            continue

        with open(scene_path) as f:
            scene = json.load(f)
        bounds_bottom = scene["bounds_bottom"]

        render_img = cv2.imread(render_path)
        if render_img is None:
            print(f"    Room {room_id} ({room_type}): could not load render, skipping")
            continue

        rh, rw = render_img.shape[:2]
        vh = compute_render_viewport_half(bounds_bottom)

        affine = compute_render_to_overview_affine(
            bounds_bottom, rw, rh,
            cx, cz, rotation_rad,
            vb_x, vb_y, vb_w, vb_h, ow, oh,
        )

        warped = cv2.warpAffine(render_img, affine, (ow, oh),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        mask = create_render_mask(warped)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(bool)

        blended = result.copy()
        blended[mask_3ch] = (
            opacity * warped[mask_3ch].astype(float) +
            (1 - opacity) * result[mask_3ch].astype(float)
        ).astype(np.uint8)
        result = blended

        n_objects = len(scene.get("objects", []))
        print(f"    Room {room_id} ({room_type}): vh={vh:.2f}m, rot={math.degrees(rotation_rad):.0f}deg, {n_objects} objects")
        rooms_processed += 1

    if rooms_processed == 0:
        print(f"  No rooms projected for unit {unit_id}.")
        return None

    print(f"  {rooms_processed} rooms projected.")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Reproject rendered room scenes onto the floorplan overview."
    )
    parser.add_argument("--floorplan", required=True,
                        help="Path to floorplan directory (e.g. input/generated_scenes/03OG_PLAN)")
    parser.add_argument("--unit", type=int, default=None,
                        help="Unit ID to visualize (default: all units)")
    parser.add_argument("--opacity", type=float, default=0.85,
                        help="Opacity of the rendered overlay (0-1, default: 0.85)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <floorplan>/reprojection)")
    args = parser.parse_args()

    metadata_path = os.path.join(args.floorplan, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"metadata.json not found in {args.floorplan}")
        sys.exit(1)

    metadata = load_metadata(metadata_path)

    output_dir = args.output_dir or os.path.join(args.floorplan, "reprojection")
    os.makedirs(output_dir, exist_ok=True)

    units_to_process = metadata["units"]
    if args.unit is not None:
        units_to_process = [u for u in units_to_process if u["unit_id"] == args.unit]
        if not units_to_process:
            print(f"Unit {args.unit} not found in metadata.")
            sys.exit(1)

    for unit_entry in units_to_process:
        unit_id = unit_entry["unit_id"]
        print(f"Unit {unit_id}:")

        result = process_unit(args.floorplan, metadata, unit_entry, args.opacity)
        if result is None:
            continue

        output_path = os.path.join(output_dir, f"unit_{unit_id}_reprojection.png")
        cv2.imwrite(output_path, result)
        print(f"  Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
