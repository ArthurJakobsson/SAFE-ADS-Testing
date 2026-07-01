"""Post-process SAFE BEVs into the UniScene-v2 visual style, and render the ego-centric BEV of
*any* vehicle in the scene.

Two things:
  1. **Restyle** — SAFE's `map_visualizer_nuplan.visualize_map_nuplan` uses a dark editor theme.
     UniScene-v2's `visualize_map` (uniscenev2_video/mmdet_plugin/core/utils/visualize.py) uses a
     light `(240,240,240)` background with the ColorBrewer `MAP_PALETTE` / nuScenes `OBJECT_PALETTE`.
     `render_uniscene()` colours SAFE's 18-channel nuPlan tensor with those exact palettes (mapping
     nuPlan layer names -> the closest UniScene class colour).
  2. **Ego view** — UniScene BEVs are ego-centric (ego at the canvas centre, heading up). SAFE BEVs
     are centred on the collision origin. `_rasterize_ego()` re-rasterizes the scene in a chosen
     vehicle's frame (translate to that vehicle, rotate so its heading points up), so we can render
     the ego BEV of *any* car in frame, per animation frame.

CLI (from Framework/):
    python ADS_Testing/BEV_Synthesis/uniscene_bev.py \
        --dsl Experiment_results/DSL_results_<ts>/DSL_extraction_results.pkl \
        --meta Experiment_results/Meta_Message_results_<ts>/meta_data_results.pkl --out ./ADS_Testing/BEV_Synthesis/output

Writes, per case:  <id>_uniscene.gif/.png  (scene view, UniScene style)
                   <id>_ego_V<k>.gif/.png  (ego-centric UniScene view of vehicle k)
"""
import argparse
import glob
import math
import os
import pickle

import numpy as np
from PIL import Image

import bev_from_dsl as B
import map_visualizer_nuplan as mv
from map_visualizer_nuplan import CH

# --- UniScene-v2 palettes (verbatim from uniscenev2_video/.../core/utils/visualize.py) ----------
UNISCENE_BG = (240, 240, 240)
MAP_PALETTE = {
    "drivable_area": (166, 206, 227), "road_segment": (31, 120, 180), "road_block": (178, 223, 138),
    "lane": (51, 160, 44), "ped_crossing": (251, 154, 153), "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111), "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214), "lane_divider": (106, 61, 154), "divider": (106, 61, 154),
}
OBJECT_PALETTE = {
    "car": (255, 158, 0), "truck": (255, 99, 71), "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0), "trailer": (255, 140, 0), "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99), "bicycle": (220, 20, 60), "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}
EGO_COLOR = (0, 200, 0)  # ego-vehicle highlight outline

# SAFE/nuPlan channel name -> UniScene class colour (semantic correspondence)
NUPLAN_TO_COLOR = {
    "generic_drivable_areas": MAP_PALETTE["drivable_area"],
    "road_segments":          MAP_PALETTE["road_segment"],
    "lane_groups_polygons":   MAP_PALETTE["lane"],
    "lane_group_connectors":  MAP_PALETTE["road_block"],
    "intersections":          MAP_PALETTE["road_block"],
    "walkways":               MAP_PALETTE["walkway"],
    "carpark_areas":          MAP_PALETTE["carpark_area"],
    "crosswalks":             MAP_PALETTE["ped_crossing"],
    "lane_divider":           MAP_PALETTE["lane_divider"],
    "road_divider":           MAP_PALETTE["road_divider"],
    "crosswalk_line":         MAP_PALETTE["stop_line"],
    "vehicle":                OBJECT_PALETTE["car"],
    "bicycle":                OBJECT_PALETTE["bicycle"],
    "pedestrian":             OBJECT_PALETTE["pedestrian"],
    "traffic_cone":           OBJECT_PALETTE["traffic_cone"],
    "barrier":                OBJECT_PALETTE["barrier"],
    "czone_sign":             OBJECT_PALETTE["truck"],
    "generic_object":         (150, 150, 150),
}

CANVAS, PATCH = (200, 200), (100, 100)

# nuPlan look: the carriageway fills (road_segments / lane_groups) are covered by the light-blue
# `drivable_area`, so the road reads light-blue (as in canonical UniScene nuPlan BEVs); ramps,
# intersections, crossings and dividers are accents painted on top, then vehicles last.
_NUPLAN_PAINT_ORDER = [
    "road_segments", "lane_groups_polygons", "generic_drivable_areas",   # -> light-blue road base
    "lane_group_connectors", "intersections", "carpark_areas", "walkways", "crosswalks",
    "road_divider", "lane_divider", "crosswalk_line",
] + list(mv._DYNAMIC_PAINT_ORDER)


def render_uniscene(bev18, target_size=512, ego_poly=None, rotate90=False):
    """Colour an 18-channel nuPlan BEV tensor in the UniScene-v2 style (light bg + palettes)."""
    bev18 = np.asarray(bev18)
    _, h, w = bev18.shape
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = UNISCENE_BG
    for name in _NUPLAN_PAINT_ORDER:
        if name in NUPLAN_TO_COLOR:
            m = bev18[CH[name]].astype(bool)
            if m.any():
                canvas[m] = NUPLAN_TO_COLOR[name]
    img = Image.fromarray(canvas)
    if ego_poly is not None:
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        pts = [(float(x), float(y)) for x, y in ego_poly]
        d.line(pts + [pts[0]], fill=EGO_COLOR, width=2)
    ratio = target_size / max(w, h)
    img = img.resize((int(w * ratio), int(h * ratio)), resample=Image.NEAREST)
    if rotate90:
        img = img.rotate(90)
    return np.asarray(img)[..., :3]


def _rasterize_ego(scene, states, ego_idx, canvas=CANVAS, patch=PATCH):
    """Re-rasterize the scene in vehicle `ego_idx`'s frame: that vehicle at the centre, heading up.

    states: list of (cx, cy, yaw, dx, dy, is_ego) for this frame (world coords).
    """
    affine = B.lidar2canvas(canvas, patch)
    H, W = canvas
    ex, ey, eyaw = states[ego_idx][0], states[ego_idx][1], states[ego_idx][2]
    # rotate the world so the ego heading maps to image-up (-row). alpha = -pi/2 - eyaw.
    alpha = -math.pi / 2.0 - eyaw
    ca, sa = math.cos(alpha), math.sin(alpha)
    R = np.array([[ca, -sa], [sa, ca]])
    origin = np.array([ex, ey])

    def tf(pts):
        return (np.asarray(pts, dtype=float) - origin) @ R.T

    bev = np.zeros((18, H, W), dtype=np.uint8)
    for name, polys in scene.static_polys.items():
        if name in CH:
            bev[CH[name]] = B._draw_filled(bev[CH[name]], [tf(p) for p in polys], affine)
    for line, name in scene.divider_lines:
        if name in CH:
            bev[CH[name]] = B._draw_line(bev[CH[name]], tf(line), affine,
                                         width=2, dashed=(name == "lane_divider"))
    ego_poly = None
    for i, (cx, cy, yaw, dx, dy, _is_ego) in enumerate(states):
        corners = tf(B.box_corners_bev(cx, cy, yaw, dx, dy))
        bev[CH["vehicle"]] = B._draw_filled(bev[CH["vehicle"]], [corners], affine)
        if i == ego_idx:
            ego_poly = B.project(corners, affine)
    return bev, ego_poly


def _save_gif(imgs, path, fps, rep_idx):
    imgs[rep_idx].save(path.replace(".gif", ".png"))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, disposal=2)


def build_case(case_id, dsl, road_type, out_dir, frames=16, fps=8):
    """Write the UniScene scene view + one ego view per vehicle for a case."""
    scene = B.build_scene(dsl, road_type)
    seq = B.simulate(scene, frames, 1.0 / fps)
    B._stage_collision(scene, seq)                  # match the main BEV (staged collision)
    cf = B._latch_impact(seq, scene.agents)         # ... and the held impact
    rep = cf if cf is not None else frames // 2

    # 1) scene view (collision-origin frame), UniScene-styled — pure restyle of the raster
    base, affine = B.rasterize_static(scene, CANVAS, PATCH)
    imgs = []
    for states in seq:
        bevf = base.copy()
        ego = B.draw_agents(bevf, states, affine)
        imgs.append(Image.fromarray(render_uniscene(bevf, 512, ego, rotate90=True)))
    _save_gif(imgs, os.path.join(out_dir, f"{case_id}_uniscene.gif"), fps, rep)

    # 2) ego-centric view of every vehicle
    labels = [a.label or f"V{i + 1}" for i, a in enumerate(scene.agents)]
    for ego_idx, label in enumerate(labels):
        imgs = []
        for states in seq:
            bev, ego_poly = _rasterize_ego(scene, states, ego_idx)
            imgs.append(Image.fromarray(render_uniscene(bev, 512, ego_poly, rotate90=False)))
        _save_gif(imgs, os.path.join(out_dir, f"{case_id}_ego_{label}.gif"), fps, rep)
    print(f"[uniscene] {case_id}: scene + {len(labels)} ego view(s) {labels} (impact f{cf})")
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsl", default="")
    ap.add_argument("--meta", default="")
    ap.add_argument("--out", default="./output")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.dsl:
        dsl = sorted(glob.glob("Experiment_results/DSL_results_*/DSL_extraction_results.pkl"),
                     key=os.path.getmtime)[-1]
    else:
        dsl = args.dsl
    with open(dsl, "rb") as f:
        dsls = pickle.load(f)
    road_by_case = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, "rb") as f:
            for row in pickle.load(f):
                road_by_case[str(row[-1])] = row[0]

    for d in dsls:
        cid = str(d.get("Scenario"))
        road_type = road_by_case.get(cid) or d.get("Road network", {}).get("Road type") or "Straight"
        build_case(cid, d, road_type, args.out, frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
