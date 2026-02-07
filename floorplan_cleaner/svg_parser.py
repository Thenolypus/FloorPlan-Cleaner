import re
import xml.etree.ElementTree as ET
from .models import IfcElement


SVG_NS = "http://www.w3.org/2000/svg"
IFC_NS = "http://www.ifcopenshell.org/ns"

IFC_TYPES_OF_INTEREST = {
    "IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcWindow",
    "IfcSlab", "IfcColumn", "IfcSpace",
}


def parse_path_d(d: str) -> list[tuple[float, float]]:
    """Parse an SVG path d-attribute into a list of (x, y) coordinates.

    Handles M, L, Z (absolute) and m, l (relative) commands.
    """
    coords = []
    current_x, current_y = 0.0, 0.0
    tokens = re.findall(r'[MmLlZz]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', d)

    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in ('M', 'm', 'L', 'l'):
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlZz':
                x_str = tokens[i]
                y_str = tokens[i + 1]
                x, y = float(x_str), float(y_str)
                if cmd in ('m', 'l'):
                    x += current_x
                    y += current_y
                current_x, current_y = x, y
                coords.append((x, y))
                i += 2
                # Implicit repeat: after M -> implicit L, after m -> implicit l
                if cmd == 'M':
                    cmd = 'L'
                elif cmd == 'm':
                    cmd = 'l'
        elif cmd in ('Z', 'z'):
            i += 1
        else:
            i += 1

    return coords


def compute_bbox(all_coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Compute (x, y, width, height) bounding box from a list of coordinates."""
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


class SvgParser:
    def __init__(self, svg_path: str):
        self.svg_path = svg_path
        self.elements: list[IfcElement] = []
        self.viewbox: tuple[float, float, float, float] = (0, 0, 0, 0)
        self.width_mm: float = 0
        self.height_mm: float = 0

    def parse(self) -> list[IfcElement]:
        tree = ET.parse(self.svg_path)
        root = tree.getroot()

        # Parse viewBox
        vb = root.get("viewBox", "0 0 0 0")
        parts = vb.split()
        self.viewbox = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))

        # Parse dimensions
        w_str = root.get("width", "0")
        h_str = root.get("height", "0")
        self.width_mm = float(re.sub(r'[^\d.]', '', w_str))
        self.height_mm = float(re.sub(r'[^\d.]', '', h_str))

        # Find all <g> elements with IFC class attributes
        self._parse_group(root)

        return self.elements

    def _parse_group(self, parent):
        for elem in parent:
            tag = elem.tag
            # Strip namespace if present
            if tag.startswith('{'):
                tag = tag.split('}', 1)[1]

            if tag == 'g':
                cls_attr = elem.get("class", "")
                classes = cls_attr.split()

                # Check for IFC type
                ifc_type = None
                for c in classes:
                    if c in IFC_TYPES_OF_INTEREST:
                        ifc_type = c
                        break

                if ifc_type is not None:
                    self._parse_ifc_element(elem, ifc_type, classes)
                else:
                    # Recurse into non-IFC groups (section containers etc.)
                    self._parse_group(elem)

    def _parse_ifc_element(self, elem, ifc_type: str, classes: list[str]):
        uuid = elem.get("id", "")
        ifc_name = elem.get(f"{{{IFC_NS}}}name", "")
        ifc_guid = elem.get(f"{{{IFC_NS}}}guid", "")

        # Determine layer
        layer = "projection" if "projection" in classes else "cut" if "cut" in classes else "unknown"

        # Determine material
        material = ""
        for c in classes:
            if c.startswith("material-"):
                material = c
                break

        # Parse child path elements
        paths = []
        all_coords = []
        for child in elem.iter(f"{{{SVG_NS}}}path"):
            d = child.get("d", "")
            if d:
                coords = parse_path_d(d)
                if coords:
                    paths.append(coords)
                    all_coords.extend(coords)

        if not all_coords:
            return

        bbox = compute_bbox(all_coords)

        self.elements.append(IfcElement(
            uuid=uuid,
            ifc_type=ifc_type,
            ifc_name=ifc_name,
            ifc_guid=ifc_guid,
            material=material,
            layer=layer,
            paths=paths,
            bbox=bbox,
        ))

    def get_elements_by_type(self, ifc_type: str) -> list[IfcElement]:
        return [e for e in self.elements if e.ifc_type == ifc_type]

    def get_boundary_elements(self) -> list[IfcElement]:
        """Get all elements that form room boundaries (walls, doors, columns) in the cut layer."""
        boundary_types = {"IfcWall", "IfcWallStandardCase", "IfcDoor", "IfcColumn"}
        return [e for e in self.elements
                if e.ifc_type in boundary_types and e.layer == "cut"]

    def get_doors(self) -> list[IfcElement]:
        return [e for e in self.elements if e.ifc_type == "IfcDoor"]

    def get_windows(self) -> list[IfcElement]:
        return [e for e in self.elements if e.ifc_type == "IfcWindow"]
