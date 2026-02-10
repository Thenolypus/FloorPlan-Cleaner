import numpy as np
import cv2
from .models import IfcElement
from .flood_fill import FloodFiller


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the outer polygon boundary of a boolean flood_mask.

    Returns an Nx2 array of (x, y) pixel coordinates.
    """
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for room mask")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)  # shape (N, 2), columns are (x, y)


def snap_to_contour(point: tuple[int, int], contour: np.ndarray) -> tuple[int, int]:
    """Find the nearest contour vertex to the given pixel coordinate."""
    dists = np.sum((contour - np.array(point)) ** 2, axis=1)
    idx = np.argmin(dists)
    return (int(contour[idx, 0]), int(contour[idx, 1]))


def split_mask(
    mask: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Split a boolean flood_mask into two halves along the line from p1 to p2.

    Returns (half_a, half_b) as boolean arrays with the same shape as mask.
    Raises ValueError if the line does not divide the room into two parts.
    """
    h, w = mask.shape
    line_img = np.zeros((h, w), dtype=np.uint8)
    cv2.line(line_img, p1, p2, 255, thickness=3)

    cut = mask.copy()
    cut[line_img > 0] = False

    num_labels, labels = cv2.connectedComponents(cut.astype(np.uint8))
    # num_labels includes background (0), real components are 1..num_labels-1

    component_sizes = []
    for label_id in range(1, num_labels):
        component_sizes.append((int(np.sum(labels == label_id)), label_id))
    component_sizes.sort(reverse=True)

    if len(component_sizes) < 2:
        raise ValueError("Split line does not divide the room into two parts")

    main_a_id = component_sizes[0][1]
    main_b_id = component_sizes[1][1]

    half_a = labels == main_a_id
    half_b = labels == main_b_id

    # Assign remaining tiny fragments to the nearest main component
    kernel = np.ones((5, 5), dtype=np.uint8)
    for i in range(2, len(component_sizes)):
        frag_id = component_sizes[i][1]
        frag_mask = labels == frag_id
        dilated_a = cv2.dilate(half_a.astype(np.uint8), kernel) > 0
        dilated_b = cv2.dilate(half_b.astype(np.uint8), kernel) > 0
        overlap_a = int(np.sum(frag_mask & dilated_a))
        overlap_b = int(np.sum(frag_mask & dilated_b))
        if overlap_a >= overlap_b:
            half_a = half_a | frag_mask
        else:
            half_b = half_b | frag_mask

    return half_a, half_b


def compute_bbox_svg(mask: np.ndarray, filler: FloodFiller) -> tuple:
    """Recompute the SVG-space bounding box from a boolean mask."""
    ys, xs = mask.nonzero()
    svg_x, svg_y = filler.pixel_to_svg(int(xs.min()), int(ys.min()))
    svg_x2, svg_y2 = filler.pixel_to_svg(int(xs.max()), int(ys.max()))
    return (svg_x, svg_y, svg_x2 - svg_x, svg_y2 - svg_y)


def assign_elements_to_halves(
    elements: list[IfcElement],
    half_a: np.ndarray,
    half_b: np.ndarray,
    filler: FloodFiller,
) -> tuple[list[IfcElement], list[IfcElement]]:
    """Assign doors/windows to correct half based on spatial overlap.

    Returns (elements_for_a, elements_for_b).
    """
    h, w = half_a.shape

    # Dilated versions for catching wall-zone elements
    kernel = np.ones((15, 15), dtype=np.uint8)
    dilated_a = cv2.dilate(half_a.astype(np.uint8), kernel) > 0
    dilated_b = cv2.dilate(half_b.astype(np.uint8), kernel) > 0

    elems_a = []
    elems_b = []

    for elem in elements:
        # Render element paths into a temporary mask
        elem_mask = np.zeros((h, w), dtype=np.uint8)
        for path_coords in elem.paths:
            pts = np.array(
                [
                    (
                        int((x - filler.viewbox[0]) * filler.scale),
                        int((y - filler.viewbox[1]) * filler.scale),
                    )
                    for x, y in path_coords
                ],
                dtype=np.int32,
            )
            if len(pts) < 2:
                continue
            cv2.polylines(elem_mask, [pts], isClosed=False, color=255, thickness=3)

        elem_bool = elem_mask > 0

        # Check overlap with each half
        overlap_a = int(np.sum(elem_bool & half_a))
        overlap_b = int(np.sum(elem_bool & half_b))

        # If no direct overlap, try dilated versions
        if overlap_a == 0 and overlap_b == 0:
            overlap_a = int(np.sum(elem_bool & dilated_a))
            overlap_b = int(np.sum(elem_bool & dilated_b))

        if overlap_a >= overlap_b:
            elems_a.append(elem)
        else:
            elems_b.append(elem)

    return elems_a, elems_b
