import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QDockWidget, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QComboBox, QPushButton, QListWidget, QListWidgetItem,
    QMessageBox, QStatusBar, QToolBar, QLineEdit, QStackedWidget,
    QButtonGroup, QRadioButton,
)
from PySide6.QtGui import QAction

from .canvas import FloorPlanCanvas, SELECTED_ROOM_COLOR, LABELED_ROOM_COLOR
from .create_canvas import CreateModeCanvas
from .svg_parser import SvgParser
from .flood_fill import FloodFiller
from .export import Exporter
from .preprocess import center_svg
from .models import Room, ApartmentUnit
from .room_splitter import simplify_contour_for_extend


ROOM_TYPES = ["bedroom", "livingroom", "diningroom", "all", "bathroom", "balcony"]
TARGET_LONGEST_SIDE = 2000

# Mode constants
MODE_SVG = 0
MODE_CREATE = 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FloorPlan Cleaner")
        self.resize(1200, 800)

        # Current mode
        self._mode = MODE_SVG

        # --- SVG mode state ---
        self._svg_path: str | None = None
        self._input_name: str | None = None
        self._parser: SvgParser | None = None
        self._filler: FloodFiller | None = None
        self._rooms: list[Room] = []
        self._units: list[ApartmentUnit] = []
        self._unit_counter = 1
        self._room_counter = 0
        self._scale: float = 1.0

        # --- Canvases ---
        self._stacked = QStackedWidget()
        self.setCentralWidget(self._stacked)

        # SVG canvas
        self._canvas = FloorPlanCanvas()
        self._canvas.room_clicked.connect(self._on_room_clicked)
        self._canvas.boundary_extend_confirmed.connect(self._on_boundary_extended)
        self._stacked.addWidget(self._canvas)  # index 0

        # Create canvas
        self._create_canvas = CreateModeCanvas()
        self._create_canvas.status_message.connect(lambda msg: self.statusBar().showMessage(msg))
        self._create_canvas.room_created.connect(self._on_create_room_created)
        self._create_canvas.rooms_changed.connect(self._on_create_rooms_changed)
        self._stacked.addWidget(self._create_canvas)  # index 1

        # Sidebar
        self._setup_sidebar()

        # Toolbar
        self._setup_toolbar()

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open an SVG file or create a unit from scratch.")

    def _setup_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open SVG", self)
        open_action.triggered.connect(self._open_svg)
        toolbar.addAction(open_action)

        create_action = QAction("Create Unit", self)
        create_action.triggered.connect(self._enter_create_mode)
        toolbar.addAction(create_action)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self._reset)
        toolbar.addAction(reset_action)

    def _setup_sidebar(self):
        dock = QDockWidget("Panel", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)

        # Use a stacked widget for the sidebar too
        self._sidebar_stack = QStackedWidget()

        # --- SVG mode sidebar (index 0) ---
        svg_container = QWidget()
        svg_layout = QVBoxLayout(svg_container)
        self._build_svg_sidebar(svg_layout)
        self._sidebar_stack.addWidget(svg_container)

        # --- Create mode sidebar (index 1) ---
        create_container = QWidget()
        create_layout = QVBoxLayout(create_container)
        self._build_create_sidebar(create_layout)
        self._sidebar_stack.addWidget(create_container)

        dock.setWidget(self._sidebar_stack)
        dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_svg_sidebar(self, layout: QVBoxLayout):
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

        # Extend boundary button
        extend_btn = QPushButton("Extend Boundary")
        extend_btn.clicked.connect(self._extend_boundary)
        layout.addWidget(extend_btn)

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
        self._height_input = QLineEdit("2.6")
        layout.addWidget(self._height_input)

        # Export button
        export_btn = QPushButton("Export All")
        export_btn.clicked.connect(self._export_all)
        layout.addWidget(export_btn)

        layout.addStretch()

    def _build_create_sidebar(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("-- Create Mode --"))

        # Tools
        layout.addWidget(QLabel("Tool:"))
        self._create_tool_group = QButtonGroup(self)

        tools = [
            ("Draw Walls", "draw_walls"),
            ("Add Door", "add_door"),
            ("Add Window", "add_window"),
            ("Select / Resize", "select"),
        ]
        for label, tool_name in tools:
            rb = QRadioButton(label)
            rb.setProperty("tool_name", tool_name)
            rb.toggled.connect(self._on_create_tool_changed)
            self._create_tool_group.addButton(rb)
            layout.addWidget(rb)
            if tool_name == "draw_walls":
                rb.setChecked(True)

        layout.addWidget(self._make_separator())

        # Room type
        layout.addWidget(QLabel("Room Type:"))
        self._create_type_combo = QComboBox()
        self._create_type_combo.addItems(ROOM_TYPES)
        layout.addWidget(self._create_type_combo)

        assign_btn = QPushButton("Assign Label")
        assign_btn.clicked.connect(self._create_assign_label)
        layout.addWidget(assign_btn)

        # Room list
        layout.addWidget(QLabel("Rooms:"))
        self._create_room_list = QListWidget()
        self._create_room_list.currentItemChanged.connect(self._on_create_room_selection_changed)
        layout.addWidget(self._create_room_list)

        # Delete buttons
        btn_row = QHBoxLayout()
        del_room_btn = QPushButton("Del Room")
        del_room_btn.clicked.connect(self._create_delete_room)
        btn_row.addWidget(del_room_btn)

        del_opening_btn = QPushButton("Del Opening")
        del_opening_btn.clicked.connect(self._create_canvas.delete_selected_opening)
        btn_row.addWidget(del_opening_btn)
        layout.addLayout(btn_row)

        # Resize controls
        layout.addWidget(self._make_separator())
        layout.addWidget(QLabel("Opening Size:"))
        size_row = QHBoxLayout()
        shrink_btn = QPushButton("-")
        shrink_btn.setFixedWidth(40)
        shrink_btn.clicked.connect(lambda: self._create_canvas.adjust_selected_opening_width(-0.1))
        size_row.addWidget(shrink_btn)
        grow_btn = QPushButton("+")
        grow_btn.setFixedWidth(40)
        grow_btn.clicked.connect(lambda: self._create_canvas.adjust_selected_opening_width(0.1))
        size_row.addWidget(grow_btn)
        layout.addLayout(size_row)

        layout.addWidget(self._make_separator())

        # Room height
        layout.addWidget(QLabel("Room Height (m):"))
        self._create_height_input = QLineEdit("2.6")
        layout.addWidget(self._create_height_input)

        # Export
        export_btn = QPushButton("Export Unit")
        export_btn.clicked.connect(self._create_export)
        layout.addWidget(export_btn)

        # Save / Load
        layout.addWidget(self._make_separator())
        save_btn = QPushButton("Save Project")
        save_btn.clicked.connect(self._create_save)
        layout.addWidget(save_btn)

        load_btn = QPushButton("Load Project")
        load_btn.clicked.connect(self._create_load)
        layout.addWidget(load_btn)

        # Back to SVG mode
        layout.addWidget(self._make_separator())
        back_btn = QPushButton("Back to SVG Mode")
        back_btn.clicked.connect(self._enter_svg_mode)
        layout.addWidget(back_btn)

        layout.addStretch()

    def _make_separator(self) -> QWidget:
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet("background-color: #ccc;")
        return sep

    # --- Mode switching ---

    def _enter_create_mode(self):
        self._mode = MODE_CREATE
        self._stacked.setCurrentIndex(1)
        self._sidebar_stack.setCurrentIndex(1)
        self.statusBar().showMessage(
            "Create mode: Draw walls to form rooms. Click near first vertex to close polygon."
        )

    def _enter_svg_mode(self):
        self._mode = MODE_SVG
        self._stacked.setCurrentIndex(0)
        self._sidebar_stack.setCurrentIndex(0)
        self.statusBar().showMessage("SVG mode. Open an SVG file to begin.")

    # --- Create mode handlers ---

    def _on_create_tool_changed(self, checked: bool):
        if not checked:
            return
        btn = self._create_tool_group.checkedButton()
        if btn:
            tool = btn.property("tool_name")
            self._create_canvas.set_tool(tool)

    def _on_create_room_created(self):
        self._refresh_create_room_list()

    def _on_create_rooms_changed(self):
        self._refresh_create_room_list()

    def _refresh_create_room_list(self):
        self._create_room_list.blockSignals(True)
        selected_id = None
        sel_room = self._create_canvas.get_selected_room()
        if sel_room:
            selected_id = sel_room.id

        self._create_room_list.clear()
        for i, room in enumerate(self._create_canvas.rooms):
            label_str = f"({room.label})" if room.label else "(unlabeled)"
            n_openings = len(room.openings)
            text = f"Room {i + 1} {label_str}"
            if n_openings:
                text += f" [{n_openings} openings]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, room.id)
            self._create_room_list.addItem(item)
            if room.id == selected_id:
                self._create_room_list.setCurrentItem(item)

        self._create_room_list.blockSignals(False)

    def _on_create_room_selection_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None):
        if current is not None:
            room_id = current.data(Qt.ItemDataRole.UserRole)
            self._create_canvas.select_room(room_id)

    def _create_assign_label(self):
        current = self._create_room_list.currentItem()
        if current is None:
            self.statusBar().showMessage("Select a room first.")
            return
        room_id = current.data(Qt.ItemDataRole.UserRole)
        room = self._create_canvas._find_room(room_id)
        if room is None:
            return
        room.label = self._create_type_combo.currentText()
        self._refresh_create_room_list()
        self.statusBar().showMessage(f"Room labeled as '{room.label}'.")

    def _create_delete_room(self):
        self._create_canvas.delete_selected_room()

    def _create_export(self):
        rooms = self._create_canvas.rooms
        if not rooms:
            self.statusBar().showMessage("No rooms to export.")
            return

        unlabeled = [r for r in rooms if not r.label]
        if unlabeled:
            self.statusBar().showMessage(f"{len(unlabeled)} room(s) unlabeled. Label all rooms before exporting.")
            return

        try:
            height_m = float(self._create_height_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Height", "Room height must be a number.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return

        self.statusBar().showMessage("Exporting...")

        exporter = Exporter()
        exporter.export_manual_unit(
            rooms=rooms,
            height_m=height_m,
            output_dir=output_dir,
        )

        self.statusBar().showMessage(f"Export complete: {output_dir}")
        QMessageBox.information(self, "Export Complete", f"Exported to:\n{output_dir}")

    def _create_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Manual Project", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._create_canvas.save_project(path)
        self.statusBar().showMessage(f"Project saved: {path}")

    def _create_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Manual Project", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._enter_create_mode()
        self._create_canvas.load_project(path)
        self._refresh_create_room_list()
        self.statusBar().showMessage(f"Project loaded: {path} ({len(self._create_canvas.rooms)} rooms)")

    # ===========================================================
    # SVG mode logic (unchanged from original)
    # ===========================================================

    def _open_svg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SVG Floor Plan", "", "SVG Files (*.svg)"
        )
        if not path:
            return

        self._enter_svg_mode()
        self._reset_svg()
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
        self._reset_svg()

    def _reset_svg(self):
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
        door_elements = self._parser.get_doors() if self._parser else None
        window_elements = self._parser.get_windows() if self._parser else None
        exporter.export_all(
            svg_path=self._svg_path,
            input_name=self._input_name,
            rooms=self._rooms,
            units=self._units,
            filler=self._filler,
            height_m=height_m,
            output_dir=output_dir,
            door_elements=door_elements,
            window_elements=window_elements,
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

    # --- Boundary extension ---

    def _extend_boundary(self):
        current_item = self._room_list.currentItem()
        if current_item is None:
            self.statusBar().showMessage("Select a room first.")
            return

        room_id = current_item.data(Qt.ItemDataRole.UserRole)
        room = self._find_room(room_id)
        if room is None or room.unit_id is not None:
            self.statusBar().showMessage("Cannot extend a saved room.")
            return

        contour = simplify_contour_for_extend(room.flood_mask)
        self._canvas.enter_extend_mode(room_id, contour, room.flood_mask)
        self.statusBar().showMessage(
            "Click on a room edge to select it. Drag to extend. Click again to confirm. Esc to cancel."
        )

    def _on_boundary_extended(self, room_id: int, new_mask, ext_info: dict):
        room = self._find_room(room_id)
        if room is None:
            return

        room.flood_mask = new_mask

        # Recompute SVG bbox
        ys, xs = new_mask.nonzero()
        svg_x, svg_y = self._filler.pixel_to_svg(int(xs.min()), int(ys.min()))
        svg_x2, svg_y2 = self._filler.pixel_to_svg(int(xs.max()), int(ys.max()))
        room.bbox_svg = (svg_x, svg_y, svg_x2 - svg_x, svg_y2 - svg_y)

        # Store extension geometry in SVG coords so export can shift openings
        p1_svg = self._filler.pixel_to_svg(*ext_info["edge_p1_px"])
        p2_svg = self._filler.pixel_to_svg(*ext_info["edge_p2_px"])
        room.boundary_extensions.append({
            "edge_p1_svg": p1_svg,
            "edge_p2_svg": p2_svg,
            "normal_svg": ext_info["normal"],  # direction is the same in SVG space
            "offset_svg": ext_info["offset_px"] / self._scale,
        })

        # Update overlay
        color = LABELED_ROOM_COLOR if room.label else SELECTED_ROOM_COLOR
        self._canvas.add_room_overlay(room_id, new_mask, color=color)

        self.statusBar().showMessage("Boundary extended.")
