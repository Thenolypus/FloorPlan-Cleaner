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
    label: str  # "bedroom", "livingroom/diningroom", "all", or "" if unlabeled
    flood_mask: np.ndarray  # boolean mask at raster resolution
    bbox_svg: tuple  # (x, y, w, h) in SVG coordinate space
    unit_id: int | None = None
    split_from: int | None = None  # id of original room this was split from
    split_line_px: tuple | None = None  # ((x1,y1),(x2,y2)) in pixel coords for mask generation


@dataclass
class ApartmentUnit:
    id: int
    room_ids: list = field(default_factory=list)
