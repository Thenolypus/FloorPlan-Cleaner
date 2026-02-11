import numpy as np
import cv2
from PySide6.QtGui import QImage, QPainter
from PySide6.QtCore import QRectF, Qt
from PySide6.QtSvg import QSvgRenderer
from .models import IfcElement


class FloodFiller:
    def __init__(self, viewbox: tuple[float, float, float, float], scale: float):
        self.viewbox = viewbox
        self.scale = scale
        self.raster_w = int(viewbox[2] * scale)
        self.raster_h = int(viewbox[3] * scale)
        self.boundary_image: np.ndarray | None = None

    def build_boundary_raster(
        self,
        svg_path: str,
        wall_elements: list[IfcElement],
        door_elements: list[IfcElement],
        window_elements: list[IfcElement],
    ) -> np.ndarray:
        """Build a binary boundary image for flood-fill room detection.

        Combines three sources:
        1. Thresholded rendered SVG (captures all visual wall outlines)
        2. Parsed wall element paths drawn as thick lines (reinforces wall boundaries)
        3. Door/window bboxes filled as solid blocks (seals openings in walls)
        """
        # Step 1: Render SVG and threshold
        renderer = QSvgRenderer(svg_path)
        image = QImage(self.raster_w, self.raster_h, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, self.raster_w, self.raster_h))
        painter.end()

        image = image.convertToFormat(QImage.Format.Format_RGB32)
        ptr = image.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(self.raster_h, self.raster_w, 4)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Step 2: Draw wall element paths as thick lines to reinforce boundaries
        for elem in wall_elements:
            for path_coords in elem.paths:
                pts = self._coords_to_pixels(path_coords)
                if len(pts) < 2:
                    continue
                cv2.polylines(binary, [pts], isClosed=False, color=0, thickness=4)
                if len(pts) >= 3 and self._is_closed_polygon(path_coords):
                    cv2.fillPoly(binary, [pts], 0)

        # Step 3: Seal door and window openings
        for elem in door_elements + window_elements:
            bx, by, bw, bh = elem.bbox
            px1 = int((bx - self.viewbox[0]) * self.scale)
            py1 = int((by - self.viewbox[1]) * self.scale)
            px2 = int(((bx + bw) - self.viewbox[0]) * self.scale)
            py2 = int(((by + bh) - self.viewbox[1]) * self.scale)
            cv2.rectangle(binary, (px1, py1), (px2, py2), 0, -1)

        # Step 4: Morphological closing for remaining small gaps
        kernel = np.ones((5, 5), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.boundary_image = binary
        return binary

    def _coords_to_pixels(self, coords: list[tuple[float, float]]) -> np.ndarray:
        return np.array(
            [(int((x - self.viewbox[0]) * self.scale),
              int((y - self.viewbox[1]) * self.scale))
             for x, y in coords],
            dtype=np.int32
        )

    def _is_closed_polygon(self, coords: list[tuple[float, float]]) -> bool:
        if len(coords) < 3:
            return False
        dx = abs(coords[0][0] - coords[-1][0])
        dy = abs(coords[0][1] - coords[-1][1])
        return dx < 0.5 and dy < 0.5

    def fill_at(self, px: int, py: int) -> np.ndarray | None:
        """Flood-fill from pixel (px, py) on the boundary raster.

        Returns a boolean mask of the filled region, or None if the click
        is on a boundary or the fill is trivially small / too large.
        """
        if self.boundary_image is None:
            raise RuntimeError("build_boundary_raster() must be called first")

        if px < 0 or px >= self.raster_w or py < 0 or py >= self.raster_h:
            return None

        if self.boundary_image[py, px] == 0:
            return None

        fill_img = self.boundary_image.copy()
        h, w = fill_img.shape
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        cv2.floodFill(fill_img, mask, (px, py), 128)
        room_mask = (fill_img == 128)

        area = np.sum(room_mask)
        total = self.raster_w * self.raster_h
        if area < 100 or area > total * 0.5:
            return None

        return room_mask

    def svg_to_pixel(self, sx: float, sy: float) -> tuple[int, int]:
        px = int((sx - self.viewbox[0]) * self.scale)
        py = int((sy - self.viewbox[1]) * self.scale)
        return (px, py)

    def pixel_to_svg(self, px: int, py: int) -> tuple[float, float]:
        sx = px / self.scale + self.viewbox[0]
        sy = py / self.scale + self.viewbox[1]
        return (sx, sy)
