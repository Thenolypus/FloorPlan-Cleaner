import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QDockWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QComboBox, QPushButton, QListWidget, QListWidgetItem,
    QMessageBox, QStatusBar, QToolBar, QLineEdit,
)
from PySide6.QtGui import QAction

from .canvas import FloorPlanCanvas, SELECTED_ROOM_COLOR, LABELED_ROOM_COLOR
from .svg_parser import SvgParser
from .flood_fill import FloodFiller
from .export import Exporter
from .preprocess import center_svg
from .models import Room, ApartmentUnit


ROOM_TYPES = ["bedroom", "livingroom/diningroom", "all", "bathroom", "balcony"]
TARGET_LONGEST_SIDE = 2000


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FloorPlan Cleaner")
        self.resize(1200, 800)

        # State
        self._svg_path: str | None = None
        self._input_name: str | None = None
        self._parser: SvgParser | None = None
        self._filler: FloodFiller | None = None
        self._rooms: list[Room] = []
        self._units: list[ApartmentUnit] = []
        self._unit_counter = 1
        self._room_counter = 0
        self._scale: float = 1.0

        # Canvas
        self._canvas = FloorPlanCanvas()
        self.setCentralWidget(self._canvas)
        self._canvas.room_clicked.connect(self._on_room_clicked)

        # Sidebar
        self._setup_sidebar()

        # Toolbar
        self._setup_toolbar()

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open an SVG file to begin.")

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open SVG", self)
        open_action.triggered.connect(self._open_svg)
        toolbar.addAction(open_action)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self._reset)
        toolbar.addAction(reset_action)

    def _setup_sidebar(self):
        dock = QDockWidget("Room Panel", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)

        # Room type selection
        layout.addWidget(QLabel("Room Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(ROOM_TYPES)
        layout.addWidget(self._type_combo)

        # Assign label button
        assign_btn = QPushButton("Assign Label")
        assign_btn.clicked.connect(self._assign_label)
        layout.addWidget(assign_btn)

        # Room list
        layout.addWidget(QLabel("Rooms (current unit):"))
        self._room_list = QListWidget()
        self._room_list.currentItemChanged.connect(self._on_room_selection_changed)
        layout.addWidget(self._room_list)

        # Delete room button
        delete_btn = QPushButton("Delete Room")
        delete_btn.clicked.connect(self._delete_room)
        layout.addWidget(delete_btn)

        # Separator
        layout.addWidget(self._make_separator())

        # Save unit button
        save_unit_btn = QPushButton("Save Unit")
        save_unit_btn.clicked.connect(self._save_unit)
        layout.addWidget(save_unit_btn)

        # Unit list
        layout.addWidget(QLabel("Saved Units:"))
        self._unit_list = QListWidget()
        layout.addWidget(self._unit_list)

        # Separator
        layout.addWidget(self._make_separator())

        # Room height input
        layout.addWidget(QLabel("Room Height (m):"))
        self._height_input = QLineEdit("2.8")
        layout.addWidget(self._height_input)

        # Export button
        export_btn = QPushButton("Export All")
        export_btn.clicked.connect(self._export_all)
        layout.addWidget(export_btn)

        layout.addStretch()

        dock.setWidget(container)
        dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _make_separator(self) -> QWidget:
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet("background-color: #ccc;")
        return sep

    def _open_svg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SVG Floor Plan", "", "SVG Files (*.svg)"
        )
        if not path:
            return

        self._reset()
        self._input_name = os.path.splitext(os.path.basename(path))[0]
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")

        # Preprocess: center SVG content by cropping viewBox to content bounds
        processed_path = center_svg(path)
        self._svg_path = processed_path

        # Parse SVG
        self._parser = SvgParser(processed_path)
        self._parser.parse()
        vb = self._parser.viewbox

        # Compute scale
        self._scale = TARGET_LONGEST_SIDE / max(vb[2], vb[3])

        # Build flood filler
        self._filler = FloodFiller(vb, self._scale)
        wall_types = {"IfcWall", "IfcWallStandardCase"}
        wall_elements = [e for e in self._parser.elements if e.ifc_type in wall_types]
        self._filler.build_boundary_raster(
            svg_path=processed_path,
            wall_elements=wall_elements,
            door_elements=self._parser.get_doors(),
            window_elements=self._parser.get_windows(),
        )

        # Display SVG
        self._canvas.load_svg(processed_path, TARGET_LONGEST_SIDE)

        n_walls = len(self._parser.get_elements_by_type("IfcWall")) + len(self._parser.get_elements_by_type("IfcWallStandardCase"))
        n_doors = len(self._parser.get_doors())
        n_windows = len(self._parser.get_windows())
        self.statusBar().showMessage(
            f"Loaded: {os.path.basename(path)} | "
            f"Walls: {n_walls}, Doors: {n_doors}, Windows: {n_windows}"
        )

    def _reset(self):
        self._svg_path = None
        self._input_name = None
        self._parser = None
        self._filler = None
        self._rooms.clear()
        self._units.clear()
        self._unit_counter = 1
        self._room_counter = 0
        self._room_list.clear()
        self._unit_list.clear()
        self._canvas._scene.clear()
        self._canvas._overlay_items.clear()
        self._canvas._base_pixmap_item = None
        self.statusBar().showMessage("Open an SVG file to begin.")

    def _on_room_clicked(self, px: int, py: int):
        if self._filler is None:
            return

        # Check if we already have a room at this pixel
        for room in self._rooms:
            if room.unit_id is None and room.flood_mask[py, px]:
                self.statusBar().showMessage("Already selected this room. Click elsewhere or delete it.")
                return

        mask = self._filler.fill_at(px, py)
        if mask is None:
            self.statusBar().showMessage("No room detected at click point (boundary or too small/large).")
            return

        # Create room
        room_id = self._room_counter
        self._room_counter += 1

        # Compute SVG bbox
        ys, xs = mask.nonzero()
        svg_x, svg_y = self._filler.pixel_to_svg(int(xs.min()), int(ys.min()))
        svg_x2, svg_y2 = self._filler.pixel_to_svg(int(xs.max()), int(ys.max()))
        bbox_svg = (svg_x, svg_y, svg_x2 - svg_x, svg_y2 - svg_y)

        room = Room(id=room_id, label="", flood_mask=mask, bbox_svg=bbox_svg)
        self._rooms.append(room)

        # Add overlay
        self._canvas.add_room_overlay(room_id, mask)

        # Add to room list (display number is 1-based list position)
        display_num = self._room_list.count() + 1
        item = QListWidgetItem(f"Room {display_num} (unlabeled)")
        item.setData(Qt.ItemDataRole.UserRole, room_id)
        self._room_list.addItem(item)
        self._room_list.setCurrentItem(item)

        self.statusBar().showMessage(f"Room {display_num} selected. Assign a label from the sidebar.")

    def _on_room_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        if previous is not None:
            prev_id = previous.data(Qt.ItemDataRole.UserRole)
            prev_room = self._find_room(prev_id)
            if prev_room is not None and prev_room.unit_id is None:
                color = LABELED_ROOM_COLOR if prev_room.label else SELECTED_ROOM_COLOR
                self._canvas.add_room_overlay(prev_id, prev_room.flood_mask, color=color)

        if current is not None:
            cur_id = current.data(Qt.ItemDataRole.UserRole)
            cur_room = self._find_room(cur_id)
            if cur_room is not None and cur_room.unit_id is None:
                self._canvas.add_room_overlay(cur_id, cur_room.flood_mask, color=SELECTED_ROOM_COLOR)

    def _assign_label(self):
        current_item = self._room_list.currentItem()
        if current_item is None:
            self.statusBar().showMessage("Select a room from the list first.")
            return

        room_id = current_item.data(Qt.ItemDataRole.UserRole)
        room = self._find_room(room_id)
        if room is None:
            return

        label = self._type_combo.currentText()
        room.label = label
        display_num = self._room_list.row(current_item) + 1
        current_item.setText(f"Room {display_num} ({label})")
        self.statusBar().showMessage(f"Room {display_num} labeled as '{label}'.")

    def _delete_room(self):
        current_item = self._room_list.currentItem()
        if current_item is None:
            return

        room_id = current_item.data(Qt.ItemDataRole.UserRole)
        self._canvas.remove_room_overlay(room_id)
        self._rooms = [r for r in self._rooms if r.id != room_id]
        self._room_list.takeItem(self._room_list.row(current_item))

        # Renumber remaining items
        self._renumber_room_list()
        self.statusBar().showMessage("Room deleted.")

    def _save_unit(self):
        current_rooms = [r for r in self._rooms if r.unit_id is None]
        if not current_rooms:
            self.statusBar().showMessage("No rooms to save. Select rooms first.")
            return

        unit_id = self._unit_counter
        unit = ApartmentUnit(id=unit_id, room_ids=[r.id for r in current_rooms])
        self._units.append(unit)

        # Assign unit_id to rooms and update overlays
        for room in current_rooms:
            room.unit_id = unit_id
            self._canvas.update_room_overlay_color(room.id, room.flood_mask, unit_id)

        # Update unit list
        room_labels = ", ".join(r.label or "unlabelled" for r in current_rooms)
        self._unit_list.addItem(f"Unit {unit_id}: {len(current_rooms)} rooms ({room_labels})")

        # Clear room list (they're now saved)
        self._room_list.clear()

        self._unit_counter += 1
        self.statusBar().showMessage(f"Unit {unit_id} saved with {len(current_rooms)} rooms.")

    def _export_all(self):
        if not self._units:
            self.statusBar().showMessage("No units to export. Save at least one unit first.")
            return

        # Parse height
        try:
            height_m = float(self._height_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Height", "Room height must be a number.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        self.statusBar().showMessage("Exporting...")

        exporter = Exporter()
        exporter.export_all(
            svg_path=self._svg_path,
            input_name=self._input_name,
            rooms=self._rooms,
            units=self._units,
            filler=self._filler,
            height_m=height_m,
            output_dir=output_dir,
        )

        self.statusBar().showMessage(f"Export complete: {output_dir}")
        QMessageBox.information(self, "Export Complete", f"Exported to:\n{output_dir}")

    def _refresh_unit_list_display(self):
        self._unit_list.clear()
        for unit in self._units:
            unit_rooms = [r for r in self._rooms if r.unit_id == unit.id]
            room_labels = ", ".join(r.label or "unlabelled" for r in unit_rooms)
            self._unit_list.addItem(
                f"Unit {unit.id}: {len(unit_rooms)} rooms ({room_labels})"
            )

    def _renumber_room_list(self):
        for i in range(self._room_list.count()):
            item = self._room_list.item(i)
            room_id = item.data(Qt.ItemDataRole.UserRole)
            room = self._find_room(room_id)
            label_str = f"({room.label})" if room and room.label else "(unlabeled)"
            item.setText(f"Room {i + 1} {label_str}")

    def _find_room(self, room_id: int) -> Room | None:
        for r in self._rooms:
            if r.id == room_id:
                return r
        return None
