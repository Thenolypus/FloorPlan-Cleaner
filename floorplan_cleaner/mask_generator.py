import numpy as np
import cv2
from .models import IfcElement, Room
from .flood_fill import FloodFiller

# Margin in pixels to expand the crop region beyond the room mask bbox.
# This ensures doors/windows sitting in the wall just outside the room
# are captured by the dilated boundary zone.
CROP_MARGIN = 15


class MaskGenerator:
    def __init__(self, filler: FloodFiller, doors: list[IfcElement], windows: list[IfcElement]):
        self.filler = filler
        self.doors = doors
        self.windows = windows

    def generate_mask(self, room: Room) -> np.ndarray:
        """Generate a 120x120 grayscale mask for a room.

        Pixel values: 0=void, 85=floor, 170=door, 255=window.
        """
        full_mask = room.flood_mask
        img_h, img_w = full_mask.shape
        ys, xs = np.where(full_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        # Expand crop region by margin to capture doors/windows in adjacent walls
        y_min_ext = max(0, y_min - CROP_MARGIN)
        y_max_ext = min(img_h - 1, y_max + CROP_MARGIN)
        x_min_ext = max(0, x_min - CROP_MARGIN)
        x_max_ext = min(img_w - 1, x_max + CROP_MARGIN)

        crop_mask = full_mask[y_min_ext:y_max_ext + 1, x_min_ext:x_max_ext + 1]
        h, w = crop_mask.shape

        # Start with void
        result = np.zeros((h, w), dtype=np.uint8)

        # Fill floor area
        result[crop_mask] = 85

        # Build the room boundary zone for door/window detection:
        # Dilate the mask outward and subtract eroded mask to get a boundary strip.
        # The dilation extends into the wall region where doors/windows sit.
        kernel_dilate = np.ones((11, 11), dtype=np.uint8)
        kernel_erode = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(crop_mask.astype(np.uint8), kernel_dilate, iterations=1)
        eroded = cv2.erode(crop_mask.astype(np.uint8), kernel_erode, iterations=1)
        boundary_zone = (dilated > 0) & (eroded == 0)

        # Overlay doors
        self._overlay_elements(result, boundary_zone, self.doors, x_min_ext, y_min_ext, w, h, 170)

        # Overlay windows
        self._overlay_elements(result, boundary_zone, self.windows, x_min_ext, y_min_ext, w, h, 255)

        # Overlay synthetic door at split boundary if this room was split
        if room.split_line_px is not None:
            self._overlay_split_boundary(result, crop_mask, room.split_line_px,
                                         x_min_ext, y_min_ext, w, h)

        # Pad to square
        result = self._pad_to_square(result)

        # Resize to 120x120 with nearest-neighbor
        result = cv2.resize(result, (120, 120), interpolation=cv2.INTER_NEAREST)

        return result

    def _overlay_elements(self, result: np.ndarray, boundary_zone: np.ndarray,
                          elements: list[IfcElement], x_off: int, y_off: int,
                          w: int, h: int, value: int):
        """Render IFC elements onto the result mask where they intersect the room boundary."""
        for elem in elements:
            ebbox = self._element_pixel_bbox(elem)
            ex, ey, ew, eh = ebbox
            # Check bbox overlap with the extended crop region
            if ex > x_off + w or ex + ew < x_off or ey > y_off + h or ey + eh < y_off:
                continue

            # Render element paths into a local mask
            elem_mask = np.zeros((h, w), dtype=np.uint8)
            for path_coords in elem.paths:
                pts = np.array(
                    [(int((x - self.filler.viewbox[0]) * self.filler.scale) - x_off,
                      int((y - self.filler.viewbox[1]) * self.filler.scale) - y_off)
                     for x, y in path_coords],
                    dtype=np.int32
                )
                if len(pts) < 2:
                    continue
                cv2.polylines(elem_mask, [pts], isClosed=False, color=255, thickness=3)

            # Only apply where the element intersects the boundary zone
            overlap = (elem_mask > 0) & boundary_zone
            result[overlap] = value

    def _element_pixel_bbox(self, elem: IfcElement) -> tuple[int, int, int, int]:
        bx, by, bw, bh = elem.bbox
        px = int((bx - self.filler.viewbox[0]) * self.filler.scale)
        py = int((by - self.filler.viewbox[1]) * self.filler.scale)
        pw = int(bw * self.filler.scale)
        ph = int(bh * self.filler.scale)
        return (px, py, pw, ph)

    def _overlay_split_boundary(self, result: np.ndarray, crop_mask: np.ndarray,
                                split_line_px: tuple, x_off: int, y_off: int,
                                w: int, h: int):
        """Encode the split boundary as a synthetic archway door.

        Central ~65% of the split edge is marked as door (170).
        Corner ~17.5% on each end stays void (0) to simulate wall segments.
        """
        p1_full, p2_full = split_line_px
        p1 = (p1_full[0] - x_off, p1_full[1] - y_off)
        p2 = (p2_full[0] - x_off, p2_full[1] - y_off)

        # Build boundary zone for this room half
        kernel_dilate = np.ones((11, 11), dtype=np.uint8)
        kernel_erode = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(crop_mask.astype(np.uint8), kernel_dilate, iterations=1)
        eroded = cv2.erode(crop_mask.astype(np.uint8), kernel_erode, iterations=1)
        boundary_zone = (dilated > 0) & (eroded == 0)

        # Draw the full split line
        line_img = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_img, p1, p2, 255, thickness=3)

        # Find pixels on the split line AND in the boundary zone
        split_boundary = (line_img > 0) & boundary_zone
        ys, xs = np.where(split_boundary)

        if len(xs) == 0:
            return

        # Compute parametric position along the line for each pixel (0..1)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        line_len_sq = dx * dx + dy * dy
        if line_len_sq == 0:
            return

        t_values = ((xs - p1[0]) * dx + (ys - p1[1]) * dy) / line_len_sq
        t_values = np.clip(t_values, 0.0, 1.0)

        # Central 65% is door (170), corners (17.5% each end) stay void (0)
        corner_frac = 0.175
        door_pixels = (t_values >= corner_frac) & (t_values <= 1.0 - corner_frac)

        result[ys[door_pixels], xs[door_pixels]] = 170

    def generate_combined_mask(self, rooms: list[Room]) -> np.ndarray:
        """Generate a single 120x120 grayscale mask with all rooms combined.

        Each room's floor area is filled, plus doors/windows on boundaries.
        Pixel values: 0=void, 85=floor, 170=door, 255=window.
        """
        img_h, img_w = self.filler.raster_h, self.filler.raster_w

        # Find the bounding box covering all rooms
        all_ys = []
        all_xs = []
        for room in rooms:
            ys, xs = np.where(room.flood_mask)
            all_ys.append(ys)
            all_xs.append(xs)
        all_ys = np.concatenate(all_ys)
        all_xs = np.concatenate(all_xs)
        y_min, y_max = int(all_ys.min()), int(all_ys.max())
        x_min, x_max = int(all_xs.min()), int(all_xs.max())

        # Expand by margin
        y_min_ext = max(0, y_min - CROP_MARGIN)
        y_max_ext = min(img_h - 1, y_max + CROP_MARGIN)
        x_min_ext = max(0, x_min - CROP_MARGIN)
        x_max_ext = min(img_w - 1, x_max + CROP_MARGIN)

        h = y_max_ext - y_min_ext + 1
        w = x_max_ext - x_min_ext + 1
        result = np.zeros((h, w), dtype=np.uint8)

        # Fill all rooms' floor areas
        for room in rooms:
            crop_mask = room.flood_mask[y_min_ext:y_max_ext + 1, x_min_ext:x_max_ext + 1]
            result[crop_mask] = 85

        # Build combined boundary zone for door/window detection
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for room in rooms:
            crop_mask = room.flood_mask[y_min_ext:y_max_ext + 1, x_min_ext:x_max_ext + 1]
            combined_mask[crop_mask] = 1
        kernel_dilate = np.ones((11, 11), dtype=np.uint8)
        kernel_erode = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(combined_mask, kernel_dilate, iterations=1)
        eroded = cv2.erode(combined_mask, kernel_erode, iterations=1)
        boundary_zone = (dilated > 0) & (eroded == 0)

        # Overlay doors and windows
        self._overlay_elements(result, boundary_zone, self.doors, x_min_ext, y_min_ext, w, h, 170)
        self._overlay_elements(result, boundary_zone, self.windows, x_min_ext, y_min_ext, w, h, 255)

        # Overlay synthetic doors at split boundaries for split rooms
        for room in rooms:
            if room.split_line_px is not None:
                crop_mask = room.flood_mask[y_min_ext:y_max_ext + 1, x_min_ext:x_max_ext + 1]
                self._overlay_split_boundary(result, crop_mask, room.split_line_px,
                                             x_min_ext, y_min_ext, w, h)

        result = self._pad_to_square(result)
        result = cv2.resize(result, (120, 120), interpolation=cv2.INTER_NEAREST)
        return result

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
        return padded
