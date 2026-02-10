import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QLineF
from PySide6.QtGui import (
    QImage, QPixmap, QColor, QPainter, QWheelEvent, QMouseEvent, QPen,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsLineItem, QGraphicsEllipseItem,
)
from PySide6.QtSvg import QSvgRenderer

from .room_splitter import snap_to_contour


# Colors for room overlays, cycled per unit
UNIT_COLORS = [
    QColor(255, 100, 100, 80),   # red
    QColor(100, 100, 255, 80),   # blue
    QColor(100, 255, 100, 80),   # green
    QColor(255, 200, 50, 80),    # orange
    QColor(200, 100, 255, 80),   # purple
    QColor(50, 200, 200, 80),    # cyan
]

# Colors for unsaved rooms (current unit being built)
SELECTED_ROOM_COLOR = QColor(255, 255, 0, 120)   # yellow - currently selected in list
LABELED_ROOM_COLOR = QColor(100, 255, 100, 100)   # green - labeled but not yet saved

# Colors for split half previews
SPLIT_HALF_A_COLOR = QColor(0, 200, 200, 100)    # cyan
SPLIT_HALF_B_COLOR = QColor(200, 0, 200, 100)     # magenta


class FloorPlanCanvas(QGraphicsView):
    room_clicked = Signal(int, int)  # pixel x, y on the raster image
    split_line_complete = Signal(int, int, int, int)  # p1x, p1y, p2x, p2y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        # No drag mode by default; right-click drag pans
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._base_pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: dict[int, QGraphicsPixmapItem] = {}  # room_id -> overlay
        self._raster_w = 0
        self._raster_h = 0
        self._panning = False
        self._pan_start = None

        # Split-line state
        self._split_mode = False
        self._split_state = "IDLE"  # "IDLE" | "FIRST_CLICK" | "COMPLETE"
        self._split_p1: tuple[int, int] | None = None
        self._split_p2: tuple[int, int] | None = None
        self._split_contour: np.ndarray | None = None
        self._split_room_id: int | None = None

        # Split graphics items
        self._split_preview_line: QGraphicsLineItem | None = None
        self._split_final_line: QGraphicsLineItem | None = None
        self._split_point_items: list[QGraphicsEllipseItem] = []
        self._split_half_overlays: dict[str, QGraphicsPixmapItem] = {}

        # Enable mouse tracking for live preview line
        self.setMouseTracking(True)

    def load_svg(self, svg_path: str, target_longest_side: int = 2000) -> tuple[int, int]:
        """Load an SVG file and render it to a raster image for display.

        Returns (raster_width, raster_height).
        """
        renderer = QSvgRenderer(svg_path)
        vb = renderer.viewBox()
        vb_w, vb_h = vb.width(), vb.height()

        # Compute scale to make longest side = target
        scale = target_longest_side / max(vb_w, vb_h)
        self._raster_w = int(vb_w * scale)
        self._raster_h = int(vb_h * scale)

        # Render SVG to QImage
        image = QImage(self._raster_w, self._raster_h, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, self._raster_w, self._raster_h))
        painter.end()

        pixmap = QPixmap.fromImage(image)
        self._scene.clear()
        self._overlay_items.clear()
        self._base_pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(0, 0, self._raster_w, self._raster_h))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        return (self._raster_w, self._raster_h)

    def add_room_overlay(self, room_id: int, mask: np.ndarray, unit_id: int | None = None, color: QColor | None = None):
        """Add a semi-transparent colored overlay for a room."""
        if color is not None:
            pass  # use explicit color
        elif unit_id is not None:
            color = UNIT_COLORS[unit_id % len(UNIT_COLORS)]
        else:
            color = SELECTED_ROOM_COLOR

        h, w = mask.shape

        # Build BGRA buffer directly via numpy (QImage Format_ARGB32 is BGRA in memory)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask, 0] = color.blue()
        rgba[mask, 1] = color.green()
        rgba[mask, 2] = color.red()
        rgba[mask, 3] = color.alpha()

        overlay = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        # QImage doesn't own the buffer, so we must copy before rgba goes out of scope
        overlay = overlay.copy()

        pixmap = QPixmap.fromImage(overlay)

        # Remove old overlay for this room if exists
        if room_id in self._overlay_items:
            self._scene.removeItem(self._overlay_items[room_id])

        item = self._scene.addPixmap(pixmap)
        item.setZValue(1)  # Above the base image
        self._overlay_items[room_id] = item

    def remove_room_overlay(self, room_id: int):
        if room_id in self._overlay_items:
            self._scene.removeItem(self._overlay_items[room_id])
            del self._overlay_items[room_id]

    def update_room_overlay_color(self, room_id: int, mask: np.ndarray, unit_id: int):
        """Update a room overlay's color (e.g., when assigned to a unit)."""
        self.add_room_overlay(room_id, mask, unit_id)

    # --- Split mode methods ---

    def enter_split_mode(self, room_id: int, contour: np.ndarray):
        """Enter split-line drawing mode for a room."""
        self._split_mode = True
        self._split_state = "IDLE"
        self._split_p1 = None
        self._split_p2 = None
        self._split_contour = contour
        self._split_room_id = room_id
        self.setCursor(Qt.CursorShape.CrossCursor)

    def exit_split_mode(self):
        """Exit split mode and clean up all split graphics."""
        self._split_mode = False
        self._split_state = "IDLE"
        self._split_p1 = None
        self._split_p2 = None
        self._split_contour = None
        self._split_room_id = None
        self._clear_split_graphics()
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def clear_split_preview(self):
        """Clear split preview graphics but stay in split mode for redo."""
        self._split_state = "IDLE"
        self._split_p1 = None
        self._split_p2 = None
        self._clear_split_graphics()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _clear_split_graphics(self):
        """Remove all split-related graphics items from the scene."""
        if self._split_preview_line is not None:
            self._scene.removeItem(self._split_preview_line)
            self._split_preview_line = None
        if self._split_final_line is not None:
            self._scene.removeItem(self._split_final_line)
            self._split_final_line = None
        for item in self._split_point_items:
            self._scene.removeItem(item)
        self._split_point_items.clear()
        for key in list(self._split_half_overlays.keys()):
            self._scene.removeItem(self._split_half_overlays[key])
        self._split_half_overlays.clear()

    def show_split_preview(self, half_a: np.ndarray, half_b: np.ndarray):
        """Show cyan/magenta overlays for the two split halves."""
        # Remove original room overlay
        if self._split_room_id is not None:
            self.remove_room_overlay(self._split_room_id)

        for key, (mask, color) in [
            ("a", (half_a, SPLIT_HALF_A_COLOR)),
            ("b", (half_b, SPLIT_HALF_B_COLOR)),
        ]:
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask, 0] = color.blue()
            rgba[mask, 1] = color.green()
            rgba[mask, 2] = color.red()
            rgba[mask, 3] = color.alpha()

            overlay = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_ARGB32)
            overlay = overlay.copy()
            pixmap = QPixmap.fromImage(overlay)

            if key in self._split_half_overlays:
                self._scene.removeItem(self._split_half_overlays[key])

            item = self._scene.addPixmap(pixmap)
            item.setZValue(2)
            self._split_half_overlays[key] = item

    def _add_split_point_dot(self, px: int, py: int):
        """Draw a small red dot at a split point."""
        r = 4
        dot = self._scene.addEllipse(
            px - r, py - r, r * 2, r * 2,
            QPen(QColor(255, 0, 0)),
            QColor(255, 0, 0, 200),
        )
        dot.setZValue(3)
        self._split_point_items.append(dot)

    # --- Event handlers ---

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            # Start panning
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap_item:
            scene_pos = self.mapToScene(event.position().toPoint())
            px = int(scene_pos.x())
            py = int(scene_pos.y())

            if not (0 <= px < self._raster_w and 0 <= py < self._raster_h):
                super().mousePressEvent(event)
                return

            # Split mode takes precedence
            if self._split_mode and self._split_contour is not None:
                snapped = snap_to_contour((px, py), self._split_contour)

                if self._split_state == "IDLE":
                    self._split_p1 = snapped
                    self._add_split_point_dot(snapped[0], snapped[1])
                    self._split_state = "FIRST_CLICK"
                    event.accept()
                    return

                elif self._split_state == "FIRST_CLICK":
                    self._split_p2 = snapped
                    self._add_split_point_dot(snapped[0], snapped[1])

                    # Remove preview line
                    if self._split_preview_line is not None:
                        self._scene.removeItem(self._split_preview_line)
                        self._split_preview_line = None

                    # Draw final line
                    pen = QPen(QColor(255, 0, 0), 2)
                    self._split_final_line = self._scene.addLine(
                        QLineF(
                            QPointF(self._split_p1[0], self._split_p1[1]),
                            QPointF(self._split_p2[0], self._split_p2[1]),
                        ),
                        pen,
                    )
                    self._split_final_line.setZValue(3)

                    self._split_state = "COMPLETE"
                    self.split_line_complete.emit(
                        self._split_p1[0], self._split_p1[1],
                        self._split_p2[0], self._split_p2[1],
                    )
                    event.accept()
                    return

                # In COMPLETE state, ignore clicks
                event.accept()
                return

            # Normal room selection
            self.room_clicked.emit(px, py)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Live preview line during split mode
        if (self._split_mode and self._split_state == "FIRST_CLICK"
                and self._split_p1 is not None and self._split_contour is not None):
            scene_pos = self.mapToScene(event.position().toPoint())
            mx = int(scene_pos.x())
            my = int(scene_pos.y())
            if 0 <= mx < self._raster_w and 0 <= my < self._raster_h:
                snapped = snap_to_contour((mx, my), self._split_contour)
                line = QLineF(
                    QPointF(self._split_p1[0], self._split_p1[1]),
                    QPointF(snapped[0], snapped[1]),
                )
                if self._split_preview_line is not None:
                    self._split_preview_line.setLine(line)
                else:
                    pen = QPen(QColor(255, 0, 0, 150), 1, Qt.PenStyle.DashLine)
                    self._split_preview_line = self._scene.addLine(line, pen)
                    self._split_preview_line.setZValue(3)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            if self._split_mode:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
