import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QWheelEvent, QMouseEvent
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtSvg import QSvgRenderer


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


class FloorPlanCanvas(QGraphicsView):
    room_clicked = Signal(int, int)  # pixel x, y on the raster image

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
            # Room selection click
            scene_pos = self.mapToScene(event.position().toPoint())
            px = int(scene_pos.x())
            py = int(scene_pos.y())
            if 0 <= px < self._raster_w and 0 <= py < self._raster_h:
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
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
