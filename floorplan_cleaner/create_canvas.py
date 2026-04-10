import math
import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QLineF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPolygonF, QWheelEvent, QMouseEvent,
    QFont, QKeyEvent,
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsLineItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsItemGroup,
)

from .models import ManualRoom, ManualOpening

PIXELS_PER_METER = 100
SNAP_GRID_M = 0.1  # snap to 10cm
CANVAS_SIZE_M = (20, 15)

WALL_PEN = QPen(QColor(40, 40, 40), 3)
WALL_PEN_PREVIEW = QPen(QColor(100, 100, 255, 150), 2, Qt.PenStyle.DashLine)
ROOM_FILL = QColor(200, 220, 255, 60)
ROOM_FILL_SELECTED = QColor(255, 255, 150, 80)
VERTEX_RADIUS = 4
VERTEX_COLOR = QColor(255, 50, 50)
CLOSE_SNAP_M = 0.3  # snap distance to close polygon

DOOR_COLOR = QColor(180, 100, 50)
DOOR_COLOR_SELECTED = QColor(255, 200, 0)
WINDOW_COLOR = QColor(100, 180, 255)
WINDOW_COLOR_SELECTED = QColor(255, 200, 0)

DEFAULT_DOOR_WIDTH_M = 0.9
DEFAULT_DOOR_HEIGHT_M = 2.1
DEFAULT_WINDOW_WIDTH_M = 1.2
DEFAULT_WINDOW_HEIGHT_M = 1.4

OPENING_VISUAL_DEPTH = 0.15  # meters, visual thickness perpendicular to wall

RESIZE_HANDLE_SIZE = 6  # pixels, half-size of resize handles


class CreateModeCanvas(QGraphicsView):
    status_message = Signal(str)
    room_created = Signal()
    rooms_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)

        # Tool: "draw_walls", "add_door", "add_window", "select"
        self._tool = "draw_walls"

        # Data
        self._rooms: list[ManualRoom] = []
        self._room_counter = 0
        self._opening_counter = 0

        # Drawing state
        self._current_vertices: list[tuple[float, float]] = []
        self._vertex_dot_items: list[QGraphicsEllipseItem] = []
        self._wall_line_items: list[QGraphicsLineItem] = []
        self._preview_line: QGraphicsLineItem | None = None

        # Room graphics: room_id -> list of scene items
        self._room_graphics: dict[int, list] = {}
        # Opening graphics: opening_id -> list of scene items
        self._opening_graphics: dict[int, list] = {}

        # Selection
        self._selected_room_id: int | None = None
        self._selected_opening_id: int | None = None
        self._resize_edge: str | None = None  # "left" or "right"
        self._resize_start_t: float = 0.0
        self._resize_start_width: float = 0.0
        self._resize_drag_origin: QPointF | None = None

        # Pan
        self._panning = False
        self._pan_start = None

        self._draw_grid()
        w_px = CANVAS_SIZE_M[0] * PIXELS_PER_METER
        h_px = CANVAS_SIZE_M[1] * PIXELS_PER_METER
        self._scene.setSceneRect(QRectF(
            -1 * PIXELS_PER_METER, -1 * PIXELS_PER_METER,
            (CANVAS_SIZE_M[0] + 2) * PIXELS_PER_METER,
            (CANVAS_SIZE_M[1] + 2) * PIXELS_PER_METER,
        ))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @property
    def rooms(self) -> list[ManualRoom]:
        return self._rooms

    def _draw_grid(self):
        self._scene.setBackgroundBrush(QBrush(QColor(250, 250, 250)))
        w_px = CANVAS_SIZE_M[0] * PIXELS_PER_METER
        h_px = CANVAS_SIZE_M[1] * PIXELS_PER_METER
        pen_minor = QPen(QColor(225, 225, 225), 0.5)
        pen_major = QPen(QColor(190, 190, 190), 1.0)

        for i in range(int(CANVAS_SIZE_M[0] / 0.5) + 1):
            x = i * 0.5 * PIXELS_PER_METER
            pen = pen_major if i % 2 == 0 else pen_minor
            item = self._scene.addLine(x, 0, x, h_px, pen)
            item.setZValue(-2)

        for i in range(int(CANVAS_SIZE_M[1] / 0.5) + 1):
            y = i * 0.5 * PIXELS_PER_METER
            pen = pen_major if i % 2 == 0 else pen_minor
            item = self._scene.addLine(0, y, w_px, y, pen)
            item.setZValue(-2)

        font = QFont("monospace", 8)
        for i in range(int(CANVAS_SIZE_M[0]) + 1):
            text = self._scene.addText(f"{i}m", font)
            text.setPos(i * PIXELS_PER_METER - 8, -20)
            text.setDefaultTextColor(QColor(130, 130, 130))
            text.setZValue(-2)
        for i in range(int(CANVAS_SIZE_M[1]) + 1):
            text = self._scene.addText(f"{i}m", font)
            text.setPos(-35, i * PIXELS_PER_METER - 8)
            text.setDefaultTextColor(QColor(130, 130, 130))
            text.setZValue(-2)

    # --- Coordinate helpers ---

    def _snap(self, x_m: float, y_m: float) -> tuple[float, float]:
        return round(x_m / SNAP_GRID_M) * SNAP_GRID_M, round(y_m / SNAP_GRID_M) * SNAP_GRID_M

    def _to_meters(self, scene_pos: QPointF) -> tuple[float, float]:
        return scene_pos.x() / PIXELS_PER_METER, scene_pos.y() / PIXELS_PER_METER

    def _to_scene(self, x_m: float, y_m: float) -> tuple[float, float]:
        return x_m * PIXELS_PER_METER, y_m * PIXELS_PER_METER

    # --- Tool switching ---

    def set_tool(self, tool: str):
        if self._tool == "draw_walls" and self._current_vertices:
            self._cancel_drawing()
        self._tool = tool
        self._selected_opening_id = None
        self._update_selection_visuals()
        if tool in ("draw_walls", "add_door", "add_window"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # --- Drawing walls ---

    def _add_vertex(self, x_m: float, y_m: float):
        x_m, y_m = self._snap(x_m, y_m)

        # Check for close to first vertex -> close polygon
        if len(self._current_vertices) >= 3:
            fx, fy = self._current_vertices[0]
            dist = math.sqrt((x_m - fx) ** 2 + (y_m - fy) ** 2)
            if dist < CLOSE_SNAP_M:
                self._close_polygon()
                return

        self._current_vertices.append((x_m, y_m))
        sx, sy = self._to_scene(x_m, y_m)

        # Draw vertex dot
        r = VERTEX_RADIUS
        dot = self._scene.addEllipse(sx - r, sy - r, r * 2, r * 2,
                                      QPen(VERTEX_COLOR), QBrush(VERTEX_COLOR))
        dot.setZValue(3)
        self._vertex_dot_items.append(dot)

        # Draw wall line from previous vertex
        if len(self._current_vertices) >= 2:
            px_m, py_m = self._current_vertices[-2]
            psx, psy = self._to_scene(px_m, py_m)
            line = self._scene.addLine(psx, psy, sx, sy, WALL_PEN)
            line.setZValue(2)
            self._wall_line_items.append(line)

        self.status_message.emit(
            f"Vertex {len(self._current_vertices)} placed at ({x_m:.1f}, {y_m:.1f}). "
            f"Click near first vertex to close, Esc to cancel."
        )

    def _close_polygon(self):
        if len(self._current_vertices) < 3:
            return

        room_id = self._room_counter
        self._room_counter += 1
        room = ManualRoom(id=room_id, label="", vertices=list(self._current_vertices))
        self._rooms.append(room)

        # Clean up drawing items
        self._clear_drawing_items()

        # Render the finished room
        self._render_room(room)

        self._selected_room_id = room_id
        self.room_created.emit()
        self.rooms_changed.emit()
        self.status_message.emit(f"Room created with {len(room.vertices)} vertices. Label it in the sidebar.")

    def _cancel_drawing(self):
        self._current_vertices.clear()
        self._clear_drawing_items()
        self.status_message.emit("Drawing cancelled.")

    def _clear_drawing_items(self):
        for item in self._vertex_dot_items:
            self._scene.removeItem(item)
        self._vertex_dot_items.clear()
        for item in self._wall_line_items:
            self._scene.removeItem(item)
        self._wall_line_items.clear()
        if self._preview_line is not None:
            self._scene.removeItem(self._preview_line)
            self._preview_line = None
        self._current_vertices.clear()

    # --- Room rendering ---

    def _render_room(self, room: ManualRoom):
        self._clear_room_graphics(room.id)
        items = []

        # Filled polygon
        poly_points = [QPointF(*self._to_scene(x, y)) for x, y in room.vertices]
        polygon = QPolygonF(poly_points)
        is_selected = (room.id == self._selected_room_id)
        fill = ROOM_FILL_SELECTED if is_selected else ROOM_FILL
        poly_item = self._scene.addPolygon(polygon, QPen(Qt.PenStyle.NoPen), QBrush(fill))
        poly_item.setZValue(0)
        items.append(poly_item)

        # Wall lines
        n = len(room.vertices)
        for i in range(n):
            x1, y1 = room.vertices[i]
            x2, y2 = room.vertices[(i + 1) % n]
            sx1, sy1 = self._to_scene(x1, y1)
            sx2, sy2 = self._to_scene(x2, y2)
            line = self._scene.addLine(sx1, sy1, sx2, sy2, WALL_PEN)
            line.setZValue(2)
            items.append(line)

        # Vertex dots
        for x, y in room.vertices:
            sx, sy = self._to_scene(x, y)
            r = VERTEX_RADIUS
            dot = self._scene.addEllipse(sx - r, sy - r, r * 2, r * 2,
                                          QPen(VERTEX_COLOR), QBrush(VERTEX_COLOR))
            dot.setZValue(3)
            items.append(dot)

        self._room_graphics[room.id] = items

        # Render openings
        for opening in room.openings:
            self._render_opening(room, opening)

    def _clear_room_graphics(self, room_id: int):
        if room_id in self._room_graphics:
            for item in self._room_graphics[room_id]:
                self._scene.removeItem(item)
            del self._room_graphics[room_id]
        # Also clear opening graphics for this room
        room = self._find_room(room_id)
        if room:
            for opening in room.openings:
                self._clear_opening_graphics(opening.id)

    def _clear_opening_graphics(self, opening_id: int):
        if opening_id in self._opening_graphics:
            for item in self._opening_graphics[opening_id]:
                self._scene.removeItem(item)
            del self._opening_graphics[opening_id]

    def _render_opening(self, room: ManualRoom, opening: ManualOpening):
        self._clear_opening_graphics(opening.id)
        items = []
        n = len(room.vertices)
        i = opening.wall_index
        p1 = np.array(room.vertices[i], dtype=np.float64)
        p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)

        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-9:
            return
        wall_dir = wall_vec / wall_len
        # Normal pointing inward (left of wall direction)
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Opening center along wall
        center = p1 + wall_dir * (opening.t * wall_len)
        half_w = opening.width / 2.0

        # Opening rectangle corners (along wall +/- half_w, perpendicular +/- depth)
        depth = OPENING_VISUAL_DEPTH
        c1 = center - wall_dir * half_w - normal * depth
        c2 = center + wall_dir * half_w - normal * depth
        c3 = center + wall_dir * half_w + normal * depth
        c4 = center - wall_dir * half_w + normal * depth

        is_selected = (opening.id == self._selected_opening_id)
        if opening.type == "door":
            color = DOOR_COLOR_SELECTED if is_selected else DOOR_COLOR
        else:
            color = WINDOW_COLOR_SELECTED if is_selected else WINDOW_COLOR

        poly = QPolygonF([QPointF(*self._to_scene(*c)) for c in [c1, c2, c3, c4]])
        rect_item = self._scene.addPolygon(poly, QPen(color, 2), QBrush(QColor(color.red(), color.green(), color.blue(), 120)))
        rect_item.setZValue(4)
        items.append(rect_item)

        # Label
        font = QFont("monospace", 7)
        label_text = "D" if opening.type == "door" else "W"
        label = self._scene.addText(f"{label_text} {opening.width:.1f}m", font)
        sc = self._to_scene(center[0], center[1])
        label.setPos(sc[0] - 20, sc[1] - 20)
        label.setDefaultTextColor(color)
        label.setZValue(5)
        items.append(label)

        # Resize handles when selected
        if is_selected:
            for edge_name, pos in [("left", center - wall_dir * half_w), ("right", center + wall_dir * half_w)]:
                hx, hy = self._to_scene(pos[0], pos[1])
                hs = RESIZE_HANDLE_SIZE
                handle = self._scene.addRect(hx - hs, hy - hs, hs * 2, hs * 2,
                                              QPen(QColor(255, 100, 0)), QBrush(QColor(255, 100, 0, 180)))
                handle.setZValue(6)
                items.append(handle)

        self._opening_graphics[opening.id] = items

    def _render_all(self):
        for room in self._rooms:
            self._render_room(room)

    def _update_selection_visuals(self):
        for room in self._rooms:
            self._render_room(room)

    # --- Shared wall detection ---

    def _find_shared_wall(self, room: ManualRoom, wall_idx: int, t: float, width: float):
        """Find another room that shares the same wall segment.

        Returns list of (other_room, other_wall_idx, other_t) where the opening
        center maps to the other room's wall parameterization.
        """
        COLLINEAR_TOL = 0.05  # meters tolerance for shared wall detection

        n = len(room.vertices)
        a1 = np.array(room.vertices[wall_idx], dtype=np.float64)
        a2 = np.array(room.vertices[(wall_idx + 1) % n], dtype=np.float64)
        a_vec = a2 - a1
        a_len = np.linalg.norm(a_vec)
        if a_len < 1e-9:
            return []
        a_dir = a_vec / a_len

        # World position of the opening center on this wall
        center_world = a1 + a_dir * (t * a_len)

        results = []

        for other_room in self._rooms:
            if other_room.id == room.id:
                continue
            m = len(other_room.vertices)
            for j in range(m):
                b1 = np.array(other_room.vertices[j], dtype=np.float64)
                b2 = np.array(other_room.vertices[(j + 1) % m], dtype=np.float64)
                b_vec = b2 - b1
                b_len = np.linalg.norm(b_vec)
                if b_len < 1e-9:
                    continue
                b_dir = b_vec / b_len

                # Check if segments are collinear: parallel and close
                dot = abs(np.dot(a_dir, b_dir))
                if dot < 0.99:  # not parallel
                    continue

                # Check distance from b1 to line through a1-a2
                perp = np.array([-a_dir[1], a_dir[0]])
                dist_to_line = abs(np.dot(b1 - a1, perp))
                if dist_to_line > COLLINEAR_TOL:
                    continue

                # Segments are collinear. Check if the opening center projects onto segment B.
                t_center_on_b = np.dot(center_world - b1, b_dir) / b_len
                if t_center_on_b < -0.01 or t_center_on_b > 1.01:
                    continue  # opening center not on this wall segment

                # Also check the opening fits
                half_ratio = (width / 2.0) / b_len
                if width > b_len * 0.95:
                    continue

                t_clamped = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, t_center_on_b))
                results.append((other_room, j, t_clamped))

        return results

    # --- Adding openings ---

    def _add_opening_at(self, x_m: float, y_m: float, opening_type: str):
        """Find nearest wall segment and add a door/window. Auto-mirrors to shared walls."""
        best_room = None
        best_wall_idx = -1
        best_t = 0.0
        best_dist = float("inf")

        click = np.array([x_m, y_m])

        for room in self._rooms:
            n = len(room.vertices)
            for i in range(n):
                p1 = np.array(room.vertices[i], dtype=np.float64)
                p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-9:
                    continue
                t = np.dot(click - p1, edge) / (edge_len ** 2)
                t = max(0.0, min(1.0, t))
                closest = p1 + t * edge
                dist = np.linalg.norm(click - closest)
                if dist < best_dist:
                    best_dist = dist
                    best_room = room
                    best_wall_idx = i
                    best_t = t

        if best_room is None or best_dist > 0.5:  # must click within 0.5m of a wall
            self.status_message.emit("Click closer to a wall to place an opening.")
            return

        # Check opening fits on wall
        n = len(best_room.vertices)
        p1 = np.array(best_room.vertices[best_wall_idx], dtype=np.float64)
        p2 = np.array(best_room.vertices[(best_wall_idx + 1) % n], dtype=np.float64)
        wall_len = np.linalg.norm(p2 - p1)

        default_w = DEFAULT_DOOR_WIDTH_M if opening_type == "door" else DEFAULT_WINDOW_WIDTH_M
        default_h = DEFAULT_DOOR_HEIGHT_M if opening_type == "door" else DEFAULT_WINDOW_HEIGHT_M

        if default_w > wall_len * 0.9:
            self.status_message.emit(f"Wall too short ({wall_len:.1f}m) for this opening.")
            return

        # Clamp t so opening doesn't overshoot wall ends
        half_ratio = (default_w / 2.0) / wall_len
        best_t = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, best_t))

        # Create the primary opening
        primary_id = self._opening_counter
        self._opening_counter += 1
        primary = ManualOpening(
            id=primary_id,
            type=opening_type,
            wall_index=best_wall_idx,
            t=best_t,
            width=default_w,
            height=default_h,
        )

        # Find shared walls and create mirrored openings
        shared = self._find_shared_wall(best_room, best_wall_idx, best_t, default_w)
        mirror_openings = []
        for other_room, other_wall_idx, other_t in shared:
            mirror_id = self._opening_counter
            self._opening_counter += 1
            mirror = ManualOpening(
                id=mirror_id,
                type=opening_type,
                wall_index=other_wall_idx,
                t=other_t,
                width=default_w,
                height=default_h,
                linked_opening=(best_room.id, primary_id),
            )
            other_room.openings.append(mirror)
            mirror_openings.append((other_room, mirror))

        # Link primary to mirrors
        if mirror_openings:
            # Link primary to the first mirror (1:1 linking for simplicity)
            first_mirror_room, first_mirror = mirror_openings[0]
            primary.linked_opening = (first_mirror_room.id, first_mirror.id)

        best_room.openings.append(primary)
        self._selected_opening_id = primary_id
        self._selected_room_id = best_room.id

        # Re-render affected rooms
        self._render_room(best_room)
        for other_room, _ in mirror_openings:
            self._render_room(other_room)

        self.rooms_changed.emit()

        n_rooms = 1 + len(mirror_openings)
        shared_msg = f" (shared across {n_rooms} rooms)" if mirror_openings else ""
        self.status_message.emit(
            f"{opening_type.capitalize()} added ({default_w:.1f}m){shared_msg}. "
            f"Select tool to resize. +/- keys to adjust width."
        )

    # --- Selection and resize ---

    def _select_at(self, x_m: float, y_m: float):
        click = np.array([x_m, y_m])

        # Check openings first
        best_opening = None
        best_opening_room = None
        best_dist = float("inf")

        for room in self._rooms:
            n = len(room.vertices)
            for opening in room.openings:
                i = opening.wall_index
                p1 = np.array(room.vertices[i], dtype=np.float64)
                p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)
                wall_vec = p2 - p1
                wall_len = np.linalg.norm(wall_vec)
                if wall_len < 1e-9:
                    continue
                wall_dir = wall_vec / wall_len
                center = p1 + wall_dir * (opening.t * wall_len)
                dist = np.linalg.norm(click - center)
                if dist < best_dist and dist < opening.width:
                    best_dist = dist
                    best_opening = opening
                    best_opening_room = room

        if best_opening is not None:
            self._selected_opening_id = best_opening.id
            self._selected_room_id = best_opening_room.id
            self._update_selection_visuals()
            self.status_message.emit(
                f"Selected {best_opening.type} ({best_opening.width:.2f}m). "
                f"+/- to resize, Delete to remove."
            )
            return

        # Check rooms
        for room in self._rooms:
            poly = np.array(room.vertices)
            if self._point_in_polygon(x_m, y_m, poly):
                self._selected_room_id = room.id
                self._selected_opening_id = None
                self._update_selection_visuals()
                self.rooms_changed.emit()
                self.status_message.emit(f"Room selected. Label it in the sidebar.")
                return

        # Deselect
        self._selected_room_id = None
        self._selected_opening_id = None
        self._update_selection_visuals()

    def _point_in_polygon(self, x: float, y: float, poly: np.ndarray) -> bool:
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _try_start_resize(self, x_m: float, y_m: float) -> bool:
        """Check if click is on a resize handle. Returns True if started resize."""
        if self._selected_opening_id is None:
            return False

        room = self._find_room(self._selected_room_id)
        if room is None:
            return False
        opening = self._find_opening(room, self._selected_opening_id)
        if opening is None:
            return False

        n = len(room.vertices)
        i = opening.wall_index
        p1 = np.array(room.vertices[i], dtype=np.float64)
        p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-9:
            return False
        wall_dir = wall_vec / wall_len
        center = p1 + wall_dir * (opening.t * wall_len)
        half_w = opening.width / 2.0

        left_handle = center - wall_dir * half_w
        right_handle = center + wall_dir * half_w

        click = np.array([x_m, y_m])
        handle_radius_m = RESIZE_HANDLE_SIZE / PIXELS_PER_METER * 2  # generous hit area

        if np.linalg.norm(click - left_handle) < handle_radius_m:
            self._resize_edge = "left"
        elif np.linalg.norm(click - right_handle) < handle_radius_m:
            self._resize_edge = "right"
        else:
            return False

        self._resize_start_t = opening.t
        self._resize_start_width = opening.width
        self._resize_drag_origin = QPointF(x_m, y_m)
        return True

    def _do_resize(self, x_m: float, y_m: float):
        room = self._find_room(self._selected_room_id)
        if room is None:
            return
        opening = self._find_opening(room, self._selected_opening_id)
        if opening is None:
            return

        n = len(room.vertices)
        i = opening.wall_index
        p1 = np.array(room.vertices[i], dtype=np.float64)
        p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1e-9:
            return
        wall_dir = wall_vec / wall_len

        # Project current mouse onto wall
        click = np.array([x_m, y_m])
        t_click = np.dot(click - p1, wall_dir) / wall_len
        t_click = max(0.01, min(0.99, t_click))

        # Compute new width and center based on which edge is dragged
        center_pos = self._resize_start_t * wall_len
        half_w = self._resize_start_width / 2.0
        drag_pos = t_click * wall_len

        if self._resize_edge == "left":
            new_left = drag_pos
            new_right = center_pos + half_w
            if new_right - new_left < 0.2:
                return
            new_width = new_right - new_left
            new_center = (new_left + new_right) / 2.0
        else:
            new_left = center_pos - half_w
            new_right = drag_pos
            if new_right - new_left < 0.2:
                return
            new_width = new_right - new_left
            new_center = (new_left + new_right) / 2.0

        # Snap width to 5cm
        new_width = round(new_width / 0.05) * 0.05
        new_width = max(0.2, min(wall_len * 0.95, new_width))

        opening.width = new_width
        opening.t = new_center / wall_len
        # Clamp
        half_ratio = (new_width / 2.0) / wall_len
        opening.t = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, opening.t))

        self._render_room(room)

    def _finish_resize(self):
        self._resize_edge = None
        self._resize_drag_origin = None

        # Sync linked opening after drag resize
        room = self._find_room(self._selected_room_id)
        if room:
            opening = self._find_opening(room, self._selected_opening_id)
            if opening:
                self._sync_linked_opening(room, opening)

        self.rooms_changed.emit()

    # --- Save / Load ---

    def save_project(self, path: str):
        """Save all rooms and openings to a JSON file."""
        import json
        data = {"rooms": []}
        for room in self._rooms:
            openings = []
            for o in room.openings:
                openings.append({
                    "id": o.id,
                    "type": o.type,
                    "wall_index": o.wall_index,
                    "t": o.t,
                    "width": o.width,
                    "height": o.height,
                    "linked_opening": list(o.linked_opening) if o.linked_opening else None,
                })
            data["rooms"].append({
                "id": room.id,
                "label": room.label,
                "vertices": room.vertices,
                "openings": openings,
            })
        data["room_counter"] = self._room_counter
        data["opening_counter"] = self._opening_counter
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_project(self, path: str):
        """Load rooms and openings from a JSON file.

        Supports two formats:
        - Create-mode project files (have "rooms" key)
        - Export metadata.json files (have "units" key) -- reverse-engineers editable rooms
        """
        import json
        with open(path, "r") as f:
            data = json.load(f)

        # Clear current state
        for room in self._rooms:
            self._clear_room_graphics(room.id)
        self._rooms.clear()
        self._clear_drawing_items()
        self._selected_room_id = None
        self._selected_opening_id = None

        if "rooms" in data:
            self._load_native_project(data)
        elif "units" in data:
            base_dir = os.path.dirname(path)
            self._load_from_export(data, base_dir)
        else:
            raise ValueError("Unrecognized file format: expected 'rooms' or 'units' key.")

        self._render_all()
        self.rooms_changed.emit()

    def _load_native_project(self, data: dict):
        for rd in data["rooms"]:
            openings = []
            for od in rd["openings"]:
                linked = tuple(od["linked_opening"]) if od.get("linked_opening") else None
                openings.append(ManualOpening(
                    id=od["id"],
                    type=od["type"],
                    wall_index=od["wall_index"],
                    t=od["t"],
                    width=od["width"],
                    height=od["height"],
                    linked_opening=linked,
                ))
            vertices = [tuple(v) for v in rd["vertices"]]
            room = ManualRoom(
                id=rd["id"],
                label=rd["label"],
                vertices=vertices,
                openings=openings,
            )
            self._rooms.append(room)

        self._room_counter = data.get("room_counter", max((r.id for r in self._rooms), default=-1) + 1)
        self._opening_counter = data.get("opening_counter", 0)
        if self._rooms:
            max_oid = max((o.id for r in self._rooms for o in r.openings), default=-1)
            self._opening_counter = max(self._opening_counter, max_oid + 1)

    def _load_from_export(self, metadata: dict, base_dir: str):
        """Reverse-engineer editable rooms from exported room JSONs + metadata."""
        import json

        room_id = 0
        opening_id = 0

        for unit in metadata["units"]:
            for room_entry in unit["rooms"]:
                room_file = os.path.join(base_dir, room_entry["output_file"])
                with open(room_file, "r") as f:
                    room_data = json.load(f)

                label = room_entry.get("room_type", "")
                rotation_rad = room_entry.get("rotation_rad", 0.0)
                cx = room_entry["center_offset_m"]["x"]
                cz = room_entry["center_offset_m"]["z"]

                # bounds_bottom has [x, 0, z] in centered+rotated space
                bounds = room_data["bounds_bottom"]
                verts_3d = np.array([[v[0], v[2]] for v in bounds], dtype=np.float64)

                # Un-rotate
                if abs(rotation_rad) > 1e-6:
                    cos_a = math.cos(-rotation_rad)
                    sin_a = math.sin(-rotation_rad)
                    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    verts_3d = (rot @ verts_3d.T).T

                # Add back center offset (x_m, z_m)
                verts_3d[:, 0] += cx
                verts_3d[:, 1] += cz

                # Convert from (x_m, z_m) to canvas coords: canvas_x = x_m, canvas_y = -z_m
                vertices = [(round(v[0], 2), round(-v[1], 2)) for v in verts_3d]

                room = ManualRoom(id=room_id, label=label, vertices=vertices)

                # Reconstruct openings
                for od in room_data.get("openings", []):
                    o_type = od["type"]
                    pos_3d = np.array([od["pos"][0], od["pos"][2]], dtype=np.float64)
                    size_3d = np.array([od["size"][0], od["size"][2]], dtype=np.float64)

                    # Un-rotate position
                    if abs(rotation_rad) > 1e-6:
                        cos_a = math.cos(-rotation_rad)
                        sin_a = math.sin(-rotation_rad)
                        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                        pos_3d = rot @ pos_3d
                        # Un-rotate size axes too
                        size_3d = np.abs(rot @ size_3d)

                    # Add back center and convert to canvas
                    world_x = pos_3d[0] + cx
                    world_z = pos_3d[1] + cz
                    canvas_x = world_x
                    canvas_y = -world_z

                    # Opening width = the larger of the two horizontal size components
                    # (the smaller one is the wall thickness)
                    opening_width = max(size_3d[0], size_3d[1])
                    opening_height = od["size"][1]  # vertical (Y) size

                    # Find which wall this opening is on
                    best_wall = 0
                    best_t = 0.5
                    best_dist = float("inf")
                    click = np.array([canvas_x, canvas_y])

                    n = len(vertices)
                    for wi in range(n):
                        p1 = np.array(vertices[wi], dtype=np.float64)
                        p2 = np.array(vertices[(wi + 1) % n], dtype=np.float64)
                        edge = p2 - p1
                        edge_len = np.linalg.norm(edge)
                        if edge_len < 1e-9:
                            continue
                        t = np.dot(click - p1, edge) / (edge_len ** 2)
                        t = max(0.0, min(1.0, t))
                        closest = p1 + t * edge
                        dist = np.linalg.norm(click - closest)
                        if dist < best_dist:
                            best_dist = dist
                            best_wall = wi
                            best_t = t

                    # Clamp t
                    p1 = np.array(vertices[best_wall], dtype=np.float64)
                    p2 = np.array(vertices[(best_wall + 1) % n], dtype=np.float64)
                    wall_len = np.linalg.norm(p2 - p1)
                    if wall_len > 1e-9 and opening_width < wall_len * 0.95:
                        half_ratio = (opening_width / 2.0) / wall_len
                        best_t = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, best_t))

                        opening = ManualOpening(
                            id=opening_id,
                            type=o_type,
                            wall_index=best_wall,
                            t=best_t,
                            width=round(opening_width, 2),
                            height=round(opening_height, 2),
                        )
                        room.openings.append(opening)
                        opening_id += 1

                self._rooms.append(room)
                room_id += 1

        self._room_counter = room_id
        self._opening_counter = opening_id

    # --- Public API ---

    def get_selected_room(self) -> ManualRoom | None:
        return self._find_room(self._selected_room_id)

    def select_room(self, room_id: int):
        self._selected_room_id = room_id
        self._selected_opening_id = None
        self._update_selection_visuals()

    def delete_selected_room(self):
        if self._selected_room_id is None:
            return
        self._clear_room_graphics(self._selected_room_id)
        self._rooms = [r for r in self._rooms if r.id != self._selected_room_id]
        self._selected_room_id = None
        self._selected_opening_id = None
        self.rooms_changed.emit()

    def delete_selected_opening(self):
        if self._selected_opening_id is None or self._selected_room_id is None:
            return
        room = self._find_room(self._selected_room_id)
        if room is None:
            return
        opening = self._find_opening(room, self._selected_opening_id)
        if opening is None:
            return

        # Delete linked opening first
        if opening.linked_opening is not None:
            linked_room_id, linked_opening_id = opening.linked_opening
            linked_room = self._find_room(linked_room_id)
            if linked_room:
                linked_room.openings = [o for o in linked_room.openings if o.id != linked_opening_id]
                self._clear_opening_graphics(linked_opening_id)
                self._render_room(linked_room)

        room.openings = [o for o in room.openings if o.id != self._selected_opening_id]
        self._selected_opening_id = None
        self._render_room(room)
        self.rooms_changed.emit()
        self.status_message.emit("Opening deleted.")

    def adjust_selected_opening_width(self, delta_m: float):
        if self._selected_opening_id is None or self._selected_room_id is None:
            return
        room = self._find_room(self._selected_room_id)
        if room is None:
            return
        opening = self._find_opening(room, self._selected_opening_id)
        if opening is None:
            return

        n = len(room.vertices)
        i = opening.wall_index
        p1 = np.array(room.vertices[i], dtype=np.float64)
        p2 = np.array(room.vertices[(i + 1) % n], dtype=np.float64)
        wall_len = np.linalg.norm(p2 - p1)

        new_width = max(0.2, min(wall_len * 0.95, opening.width + delta_m))
        opening.width = round(new_width / 0.05) * 0.05

        # Re-clamp t
        half_ratio = (opening.width / 2.0) / wall_len
        opening.t = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, opening.t))

        self._render_room(room)

        # Propagate to linked opening
        self._sync_linked_opening(room, opening)

        self.rooms_changed.emit()
        self.status_message.emit(f"Opening width: {opening.width:.2f}m")

    def _find_room(self, room_id: int | None) -> ManualRoom | None:
        if room_id is None:
            return None
        for r in self._rooms:
            if r.id == room_id:
                return r
        return None

    def _find_opening(self, room: ManualRoom, opening_id: int | None) -> ManualOpening | None:
        if opening_id is None:
            return None
        for o in room.openings:
            if o.id == opening_id:
                return o
        return None

    def _find_opening_globally(self, opening_id: int) -> tuple[ManualRoom | None, ManualOpening | None]:
        for room in self._rooms:
            for o in room.openings:
                if o.id == opening_id:
                    return room, o
        return None, None

    def _sync_linked_opening(self, source_room: ManualRoom, source_opening: ManualOpening):
        """Sync width from source opening to its linked counterpart, recomputing t on the other wall."""
        if source_opening.linked_opening is None:
            return
        linked_room_id, linked_opening_id = source_opening.linked_opening
        linked_room = self._find_room(linked_room_id)
        if linked_room is None:
            return
        linked_opening = self._find_opening(linked_room, linked_opening_id)
        if linked_opening is None:
            return

        # Compute world position of the source opening center
        n = len(source_room.vertices)
        i = source_opening.wall_index
        a1 = np.array(source_room.vertices[i], dtype=np.float64)
        a2 = np.array(source_room.vertices[(i + 1) % n], dtype=np.float64)
        a_vec = a2 - a1
        a_len = np.linalg.norm(a_vec)
        if a_len < 1e-9:
            return
        center_world = a1 + (a_vec / a_len) * (source_opening.t * a_len)

        # Project onto the linked wall
        m = len(linked_room.vertices)
        j = linked_opening.wall_index
        b1 = np.array(linked_room.vertices[j], dtype=np.float64)
        b2 = np.array(linked_room.vertices[(j + 1) % m], dtype=np.float64)
        b_vec = b2 - b1
        b_len = np.linalg.norm(b_vec)
        if b_len < 1e-9:
            return
        b_dir = b_vec / b_len

        new_t = np.dot(center_world - b1, b_dir) / b_len
        half_ratio = (source_opening.width / 2.0) / b_len
        new_t = max(half_ratio + 0.01, min(1.0 - half_ratio - 0.01, new_t))

        linked_opening.t = new_t
        linked_opening.width = source_opening.width
        self._render_room(linked_room)

    # --- Event handlers ---

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            x_m, y_m = self._to_meters(scene_pos)

            if self._tool == "draw_walls":
                self._add_vertex(x_m, y_m)
            elif self._tool in ("add_door", "add_window"):
                otype = "door" if self._tool == "add_door" else "window"
                self._add_opening_at(x_m, y_m, otype)
            elif self._tool == "select":
                if not self._try_start_resize(x_m, y_m):
                    self._select_at(x_m, y_m)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        x_m, y_m = self._to_meters(scene_pos)

        # Preview line while drawing walls
        if self._tool == "draw_walls" and self._current_vertices:
            x_m, y_m = self._snap(x_m, y_m)
            lx, ly = self._current_vertices[-1]
            sx1, sy1 = self._to_scene(lx, ly)
            sx2, sy2 = self._to_scene(x_m, y_m)
            line = QLineF(QPointF(sx1, sy1), QPointF(sx2, sy2))
            if self._preview_line is not None:
                self._preview_line.setLine(line)
            else:
                self._preview_line = self._scene.addLine(line, WALL_PEN_PREVIEW)
                self._preview_line.setZValue(3)
            event.accept()
            return

        # Resize dragging
        if self._resize_edge is not None:
            self._do_resize(x_m, y_m)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            if self._tool in ("draw_walls", "add_door", "add_window"):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._resize_edge is not None:
            self._finish_resize()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            if self._tool == "draw_walls" and self._current_vertices:
                self._cancel_drawing()
                event.accept()
                return
        if event.key() == Qt.Key.Key_Delete:
            if self._selected_opening_id is not None:
                self.delete_selected_opening()
            elif self._selected_room_id is not None:
                self.delete_selected_room()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self.adjust_selected_opening_width(0.1)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Minus:
            self.adjust_selected_opening_width(-0.1)
            event.accept()
            return
        super().keyPressEvent(event)
