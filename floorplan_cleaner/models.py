from dataclasses import dataclass, field
import numpy as np


@dataclass
class IfcElement:
    uuid: str
    ifc_type: str
    ifc_name: str
    ifc_guid: str
    material: str
    layer: str  # "projection" or "cut"
    paths: list  # list of list of (x, y) tuples in SVG coords
    bbox: tuple  # (x, y, w, h) in SVG coords


@dataclass
class Room:
    id: int
    label: str  # "bedroom", "livingroom", "diningroom", "all", "bathroom", "balcony", or "" if unlabeled
    flood_mask: np.ndarray  # boolean mask at raster resolution
    bbox_svg: tuple  # (x, y, w, h) in SVG coordinate space
    unit_id: int | None = None
    split_from: int | None = None  # id of original room this was split from
    split_line_px: tuple | None = None  # ((x1,y1),(x2,y2)) in pixel coords for mask generation
    # Each entry: {"edge_p1_svg": (x,y), "edge_p2_svg": (x,y),
    #              "normal_svg": (nx,ny), "offset_svg": float}
    boundary_extensions: list = field(default_factory=list)


@dataclass
class ApartmentUnit:
    id: int
    room_ids: list = field(default_factory=list)


@dataclass
class ManualOpening:
    id: int
    type: str  # "door" or "window"
    wall_index: int  # index into room's vertices list (wall from vertices[i] to vertices[i+1])
    t: float  # 0-1 position along the wall segment (center of opening)
    width: float  # meters (along the wall)
    height: float  # meters (vertical, for 3D export)
    linked_opening: tuple | None = None  # (room_id, opening_id) of the mirrored opening on shared wall


@dataclass
class ManualRoom:
    id: int
    label: str
    vertices: list = field(default_factory=list)  # [(x_m, y_m), ...] in meters, closed polygon
    openings: list = field(default_factory=list)  # list of ManualOpening
