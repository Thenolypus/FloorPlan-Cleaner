import os
import re
import tempfile


def _extract_content_bbox(svg_text: str) -> tuple[float, float, float, float] | None:
    """Extract bounding box of all path coordinates in the SVG body (outside <defs>).

    Returns (min_x, min_y, max_x, max_y) or None if no coordinates found.
    """
    # Only look at content after </defs> to ignore marker/pattern definitions
    defs_end = svg_text.find("</defs>")
    body = svg_text[defs_end:] if defs_end != -1 else svg_text

    xs = []
    ys = []

    # Extract coordinates from path d attributes (M/L commands)
    for m in re.finditer(r'[ML]([\d.]+),([\d.]+)', body):
        xs.append(float(m.group(1)))
        ys.append(float(m.group(2)))

    # Extract coordinates from line elements
    for m in re.finditer(r'<line[^>]*>', body):
        line = m.group(0)
        for attr in ('x1', 'x2'):
            am = re.search(rf'{attr}="([^"]*)"', line)
            if am:
                xs.append(float(am.group(1)))
        for attr in ('y1', 'y2'):
            am = re.search(rf'{attr}="([^"]*)"', line)
            if am:
                ys.append(float(am.group(1)))

    # Extract coordinates from rect elements
    for m in re.finditer(r'<rect[^>]*>', body):
        rect = m.group(0)
        xm = re.search(r' x="([^"]*)"', rect)
        ym = re.search(r' y="([^"]*)"', rect)
        wm = re.search(r'width="([^"]*)"', rect)
        hm = re.search(r'height="([^"]*)"', rect)
        if xm and ym and wm and hm:
            rx, ry = float(xm.group(1)), float(ym.group(1))
            rw, rh = float(wm.group(1)), float(hm.group(1))
            xs.extend([rx, rx + rw])
            ys.extend([ry, ry + rh])

    # Extract coordinates from circle elements
    for m in re.finditer(r'<circle[^>]*>', body):
        circ = m.group(0)
        cxm = re.search(r'cx="([^"]*)"', circ)
        cym = re.search(r'cy="([^"]*)"', circ)
        rm = re.search(r' r="([^"]*)"', circ)
        if cxm and cym and rm:
            cx, cy, r = float(cxm.group(1)), float(cym.group(1)), float(rm.group(1))
            xs.extend([cx - r, cx + r])
            ys.extend([cy - r, cy + r])

    if not xs or not ys:
        return None

    return (min(xs), min(ys), max(xs), max(ys))


def center_svg(svg_path: str) -> str:
    """Crop an SVG's viewBox to tightly fit its visual content.

    Parses element coordinates from the SVG body (outside <defs>) to find
    the actual content bounding box, then rewrites the viewBox and
    width/height so the SVG edges align with the content.

    Returns the path to the preprocessed SVG file (temp file).
    """
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    vb_match = re.search(r'viewBox="([^"]*)"', svg_text)
    if not vb_match:
        raise ValueError("SVG has no viewBox attribute")

    parts = vb_match.group(1).split()
    vb_x, vb_y, vb_w, vb_h = (
        float(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
    )

    bbox = _extract_content_bbox(svg_text)
    if bbox is None:
        print("[preprocess] No content coordinates found, skipping centering.")
        return svg_path

    min_x, min_y, max_x, max_y = bbox
    content_w = max_x - min_x
    content_h = max_y - min_y

    # Add a small margin (1% of content size)
    margin = max(content_w, content_h) * 0.01
    new_vb_x = min_x - margin
    new_vb_y = min_y - margin
    new_vb_w = content_w + 2 * margin
    new_vb_h = content_h + 2 * margin

    # Update viewBox in SVG text
    new_vb_str = f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}"
    svg_text = re.sub(r'viewBox="[^"]*"', f'viewBox="{new_vb_str}"', svg_text)

    # Update width and height (preserve mm units and data-scale relationship)
    w_match = re.search(r'width="([\d.]+)mm"', svg_text)
    h_match = re.search(r'height="([\d.]+)mm"', svg_text)
    if w_match and h_match:
        old_w_mm = float(w_match.group(1))
        mm_per_unit = old_w_mm / vb_w
        new_w_mm = new_vb_w * mm_per_unit
        new_h_mm = new_vb_h * mm_per_unit
        svg_text = re.sub(r'width="[\d.]+mm"', f'width="{new_w_mm}mm"', svg_text)
        svg_text = re.sub(r'height="[\d.]+mm"', f'height="{new_h_mm}mm"', svg_text)

    # Write to a temp file
    suffix = os.path.splitext(svg_path)[1]
    fd, output_path = tempfile.mkstemp(suffix=suffix, prefix="centered_")
    os.close(fd)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_text)

    print(
        f"[preprocess] Centered SVG: viewBox {vb_x:.1f},{vb_y:.1f},{vb_w:.1f},{vb_h:.1f}"
        f" -> {new_vb_x:.1f},{new_vb_y:.1f},{new_vb_w:.1f},{new_vb_h:.1f}"
    )

    return output_path