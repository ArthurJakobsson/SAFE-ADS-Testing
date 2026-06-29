"""nuPlan-style 18-channel BEV renderer.

This is a self-contained renderer for the UniScene-v2 nuPlan BEV layout. The stock
`occupancy_generation/utils/map_visualizer.py::visualize_map` in UniScene_v2_recon hard-codes
*nuScenes* class names (`drivable_area`, `ped_crossing`, ...), which mis-index the nuPlan
18-channel tensor produced here. We reproduce its color/compose style but with the correct
nuPlan channel layout taken verbatim from:
    UniScene_v2_recon/occupancy_generation/data_preprocess/nuplan/bev_preprocess/nuplan_dataset.py
        self.classes         -> static  channels 0..7
        self.object_classes  -> dynamic channels 8..14
    + 3 divider line channels 15..16..17 (broken-white / crosswalk-line / solid-white)

The layout (18 channels, channel-first (C, H, W)):
    0  intersections
    1  generic_drivable_areas
    2  walkways
    3  carpark_areas
    4  crosswalks
    5  lane_group_connectors
    6  lane_groups_polygons
    7  road_segments
    8  vehicle
    9  bicycle
    10 pedestrian
    11 traffic_cone
    12 barrier
    13 czone_sign
    14 generic_object
    15 lane_divider (broken white)
    16 crosswalk_line
    17 road_divider (solid white / center)
"""
from typing import List, Optional, Sequence
import numpy as np
from PIL import Image, ImageDraw

# ---- canonical nuPlan channel names (index == channel) -------------------------------------
STATIC_CLASSES = [
    "intersections", "generic_drivable_areas", "walkways", "carpark_areas",
    "crosswalks", "lane_group_connectors", "lane_groups_polygons", "road_segments",
]
OBJECT_CLASSES = [
    "vehicle", "bicycle", "pedestrian", "traffic_cone", "barrier", "czone_sign", "generic_object",
]
DIVIDER_CLASSES = ["lane_divider", "crosswalk_line", "road_divider"]

ALL_CLASSES = STATIC_CLASSES + OBJECT_CLASSES + DIVIDER_CLASSES
assert len(ALL_CLASSES) == 18

# channel index helpers
CH = {name: i for i, name in enumerate(ALL_CLASSES)}

COLORS = {
    "background":             (37, 37, 38),     # dark editor grey
    # static
    "intersections":          (84, 90, 130),    # muted indigo
    "generic_drivable_areas": (120, 130, 150),  # slate blue-grey (road)
    "walkways":               (227, 26, 28),    # red
    "carpark_areas":          (255, 127, 0),    # orange
    "crosswalks":             (251, 154, 153),  # light red
    "lane_group_connectors":  (102, 140, 110),  # muted green
    "lane_groups_polygons":   (110, 120, 140),  # slate (lane area)
    "road_segments":          (90, 95, 105),    # darker grey
    # dividers
    "lane_divider":           (200, 200, 200),  # light grey dashes
    "crosswalk_line":         (245, 245, 245),  # near white
    "road_divider":           (255, 210, 0),    # yellow center line
    # dynamic
    "vehicle":                (255, 158, 0),    # orange
    "bicycle":                (220, 20, 60),    # crimson
    "pedestrian":             (0, 120, 230),    # blue
    "traffic_cone":           (47, 79, 79),     # dark slate
    "barrier":                (112, 128, 144),  # slate grey
    "czone_sign":             (233, 150, 70),   # dark salmon
    "generic_object":         (160, 160, 160),  # grey
    # render-only highlight (not a channel)
    "ego":                    (60, 220, 90),    # bright green outline
}

# draw order: paint area-like static first, then lane areas, then special zones, then lines,
# then dynamic agents on top.
_STATIC_PAINT_ORDER = [
    "road_segments", "generic_drivable_areas", "lane_groups_polygons",
    "lane_group_connectors", "carpark_areas", "intersections", "walkways", "crosswalks",
]
_DIVIDER_PAINT_ORDER = ["lane_divider", "crosswalk_line", "road_divider"]
_DYNAMIC_PAINT_ORDER = OBJECT_CLASSES  # vehicle..generic_object


def visualize_map_nuplan(
    bev: np.ndarray,
    target_size: int = 512,
    ego_canvas_polygon: Optional[Sequence] = None,
    rotate90: bool = True,
) -> np.ndarray:
    """Render an 18-channel nuPlan BEV tensor to an RGB uint8 image.

    Args:
        bev: (18, H, W) array (0/1 masks). int8/uint8/bool all fine.
        target_size: long-edge output size (nearest-neighbour upscaled).
        ego_canvas_polygon: optional (4,2) polygon in *canvas pixel* coords to outline as ego.
        rotate90: rotate the final image 90deg (matches the stock visualizer's orientation).
    Returns:
        (target_size, target_size, 3) uint8 image.
    """
    bev = np.asarray(bev)
    assert bev.ndim == 3 and bev.shape[0] == 18, f"expected (18,H,W), got {bev.shape}"
    _, h, w = bev.shape
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = COLORS["background"]

    def paint(name):
        m = bev[CH[name]].astype(bool)
        if m.any():
            canvas[m] = COLORS[name]

    for name in _STATIC_PAINT_ORDER:
        paint(name)
    for name in _DIVIDER_PAINT_ORDER:
        paint(name)
    for name in _DYNAMIC_PAINT_ORDER:
        paint(name)

    img = Image.fromarray(canvas)

    # optional ego outline (drawn in canvas coords before resize)
    if ego_canvas_polygon is not None:
        draw = ImageDraw.Draw(img)
        pts = [(float(x), float(y)) for x, y in ego_canvas_polygon]
        draw.polygon(pts, outline=COLORS["ego"])
        # thicken the outline a touch
        draw.line(pts + [pts[0]], fill=COLORS["ego"], width=1)

    # upscale (nearest keeps masks crisp), then optional rotate to match stock orientation
    ratio = target_size / max(w, h)
    img = img.resize((int(w * ratio), int(h * ratio)), resample=Image.NEAREST)
    if rotate90:
        img = img.rotate(90)
    return np.asarray(img)[..., :3]


def legend_items(used_only: Optional[List[str]] = None):
    """Return [(name, '#rrggbb'), ...] for building an HTML legend."""
    names = used_only if used_only is not None else ALL_CLASSES + ["ego"]
    out = []
    for n in names:
        r, g, b = COLORS[n]
        out.append((n, f"#{r:02x}{g:02x}{b:02x}"))
    return out
