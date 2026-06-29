"""Synthesize nuPlan-style 18-channel BEV rasters from SAFE crash DSLs.

SAFE extracts a DSL per crash (Actors / Road network / Env / Scenario). There is no
georeferenced nuPlan map for these abstract crash sketches, so instead of querying a real
map we *procedurally* build a schematic scene from the DSL and rasterize it into the SAME
18-channel layout UniScene-v2 uses for nuPlan BEVs (see map_visualizer_nuplan.py for the
channel spec). We reuse UniScene's `lidar2canvas` affine and its polygon-projection idiom
verbatim; only the static/divider map layers are drawn procedurally (no nuplan-devkit / mmdet3d).

Outputs per crash:
    <out>/<case_id>.npz   -> np.savez_compressed(..., gt_bev_masks=(18,H,W) int8)   # uniscene key
    <out>/<case_id>.png   -> human-viewable render
    <out>/<case_id>.json  -> small sidecar describing the synthesized scene (for the dashboard)

Usage:
    # demo (no LLM needed) -- renders one BEV per road type:
    python bev_from_dsl.py --demo --out ./output
    # from a real pipeline run:
    python bev_from_dsl.py --dsl /path/DSL_extraction_results.pkl \
                           --meta /path/meta_data_results.pkl --out ./output
"""
import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from map_visualizer_nuplan import CH, visualize_map_nuplan, legend_items

# ------------------------------------------------------------------------------------------
# geometry constants
LANE_W = 3.5          # metres per lane
EXT = 52.0            # roads extend just past the +-50 m canvas edge
VEHICLE_SIZE = {      # (length_along_heading, width) in metres, tied to SAFE's vehicle models
    "sedan": (4.6, 1.85), "suv": (4.8, 1.95), "minivan": (5.2, 2.0),
    "pickup": (5.5, 2.0), "semi truck": (12.0, 2.5), "semi": (12.0, 2.5),
    "truck": (12.0, 2.5), "van": (5.2, 2.0),
}
DEFAULT_SIZE = (4.7, 1.9)

# direction -> (yaw, unit heading vector, lateral 'keep-right' sign)
#   lateral sign picks which side of a two-way road the vehicle drives on (US, right-hand).
DIRECTION = {
    "W2E": (0.0,            (1.0, 0.0),  -1),   # east-bound: drive on south side (y<0)
    "E2W": (math.pi,        (-1.0, 0.0), +1),   # west-bound: north side (y>0)
    "S2N": (math.pi / 2,    (0.0, 1.0),  +1),   # north-bound: east side (x>0)
    "N2S": (-math.pi / 2,   (0.0, -1.0), -1),   # south-bound: west side (x<0)
}


# ------------------------------------------------------------------------------------------
# scene representation
@dataclass
class Agent:
    cx: float
    cy: float
    yaw: float
    dx: float
    dy: float
    cls: str = "vehicle"
    is_ego: bool = False
    label: str = ""
    # kinematics for the temporal rollout; (cx,cy) is the ~mid-sequence ("collision") pose.
    ux: float = 1.0
    uy: float = 0.0
    speed_mps: float = 11.0
    action: str = ""


@dataclass
class Scene:
    road_type: str
    static_polys: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    divider_lines: List[Tuple[np.ndarray, str]] = field(default_factory=list)  # (polyline, channel)
    agents: List[Agent] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def add_static(self, name: str, poly: np.ndarray):
        self.static_polys.setdefault(name, []).append(np.asarray(poly, dtype=float))


# ------------------------------------------------------------------------------------------
# rasterization (reuses UniScene-v2 lidar2canvas + polygon projection idiom)
def lidar2canvas(canvas=(200, 200), patch=(100, 100)) -> np.ndarray:
    """Affine copied from UniScene_v2 nuplan_dataset.py (self.lidar2canvas)."""
    H, W = canvas
    ph, pw = patch
    return np.array([[H / ph, 0, H / 2.0],
                     [0, W / pw, W / 2.0],
                     [0, 0, 1.0]])


def project(points_xy: np.ndarray, affine: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)  # (K,3)
    return (homo @ affine.T)[:, :2]


def box_corners_bev(cx, cy, yaw, dx, dy) -> np.ndarray:
    """4 bottom corners of an oriented box in lidar x/y (replaces mmdet3d corners)."""
    l, w = dx / 2.0, dy / 2.0
    local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world = local @ R.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


def _draw_filled(mask: np.ndarray, polys_lidar, affine) -> np.ndarray:
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    for poly in polys_lidar:
        cv = project(poly, affine)
        draw.polygon(cv.round().astype(np.int32).flatten().tolist(), fill=1)
    return np.array(img)


def _draw_line(mask: np.ndarray, polyline_lidar, affine, width=2, dashed=False) -> np.ndarray:
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    cv = project(polyline_lidar, affine)
    for i in range(len(cv) - 1):
        p0, p1 = cv[i], cv[i + 1]
        if not dashed:
            draw.line([tuple(p0), tuple(p1)], fill=1, width=width)
        else:
            seg = p1 - p0
            length = float(np.hypot(*seg))
            if length < 1e-6:
                continue
            unit = seg / length
            step = 6.0  # px dash period
            d = 0.0
            while d < length:
                a = p0 + unit * d
                b = p0 + unit * min(d + step * 0.5, length)
                draw.line([tuple(a), tuple(b)], fill=1, width=width)
                d += step
    return np.array(img)


def rasterize_static(scene: Scene, canvas=(200, 200), patch=(100, 100)):
    """Draw the time-invariant map layers (static 0-7 + dividers 15-17) once."""
    H, W = canvas
    affine = lidar2canvas(canvas, patch)
    bev = np.zeros((18, H, W), dtype=np.uint8)
    for name, polys in scene.static_polys.items():
        if name in CH:
            bev[CH[name]] = _draw_filled(bev[CH[name]], polys, affine)
    for line, name in scene.divider_lines:
        if name in CH:
            bev[CH[name]] = _draw_line(bev[CH[name]], line, affine,
                                       width=2, dashed=(name == "lane_divider"))
    return bev, affine


def draw_agents(bev, agent_states, affine):
    """Rasterize agents into channel 8. agent_states: (cx, cy, yaw, dx, dy, is_ego) tuples."""
    ego_canvas_poly = None
    for (cx, cy, yaw, dx, dy, is_ego) in agent_states:
        corners = box_corners_bev(cx, cy, yaw, dx, dy)
        bev[CH["vehicle"]] = _draw_filled(bev[CH["vehicle"]], [corners], affine)
        if is_ego:
            ego_canvas_poly = project(corners, affine)
    return ego_canvas_poly


def rasterize(scene: Scene, canvas=(200, 200), patch=(100, 100)):
    bev, affine = rasterize_static(scene, canvas, patch)
    states = [(a.cx, a.cy, a.yaw, a.dx, a.dy, a.is_ego) for a in scene.agents]
    ego_canvas_poly = draw_agents(bev, states, affine)
    return bev, ego_canvas_poly


# ------------------------------------------------------------------------------------------
# DSL parsing
def _first(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    # case-insensitive fallback
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low and low[k.lower()] not in (None, ""):
            return low[k.lower()]
    return None


def parse_actors(dsl: dict) -> List[dict]:
    field_ = dsl.get("Actors", dsl.get("actors", []))
    out = []

    def add(vd):
        if not isinstance(vd, dict):
            return
        model = _first(vd, ["Model", "model"])
        init = _first(vd, ["Initial_position", "Initial position", "initial_position"])
        action = _first(vd, ["Actions", "Action", "action"])
        speed = _first(vd, ["Speed", "speed"])
        if model is None and init is None:
            return
        out.append({"model": model, "init": init, "action": action, "speed": speed})

    if isinstance(field_, dict):
        for _, v in field_.items():
            add(v)
    elif isinstance(field_, list):
        for el in field_:
            if isinstance(el, dict):
                if any(str(k).lower().startswith("vehicle") for k in el.keys()):
                    for _, v in el.items():
                        add(v)
                else:
                    add(el)
    return out


def norm_direction(init) -> str:
    if not init:
        return "W2E"
    s = str(init).upper().replace(" ", "")
    for key in DIRECTION:
        if key in s:
            return key
    return "W2E"


def size_for_model(model) -> Tuple[float, float]:
    if not model:
        return DEFAULT_SIZE
    return VEHICLE_SIZE.get(str(model).strip().lower(), DEFAULT_SIZE)


def get_num_lanes(dsl: dict) -> int:
    rn = dsl.get("Road network", dsl.get("Road_network", {})) or {}
    val = _first(rn, ["Number of lanes", "Number_of_lanes", "num_lanes"])
    try:
        n = int(str(val).strip().split()[0])
    except Exception:
        n = 2
    return max(2, min(n, 8))


def get_stem_direction(dsl: dict) -> str:
    rn = dsl.get("Road network", dsl.get("Road_network", {})) or {}
    val = _first(rn, ["Stem road direction", "Stem_road_direction", "stem"])
    return str(val).strip().capitalize() if val else "South"


# ------------------------------------------------------------------------------------------
# scene geometry per road type
def _primary_axis(actors: List[dict]) -> str:
    dirs = [norm_direction(a["init"]) for a in actors]
    if any(d in ("W2E", "E2W") for d in dirs):
        return "EW"
    if any(d in ("N2S", "S2N") for d in dirs):
        return "NS"
    return "EW"


def _band(axis: str, n_lanes: int):
    half = n_lanes * LANE_W / 2.0
    if axis == "EW":
        poly = np.array([[-EXT, -half], [EXT, -half], [EXT, half], [-EXT, half]])
    else:
        poly = np.array([[-half, -EXT], [-half, EXT], [half, EXT], [half, -EXT]])
    return poly, half


def _band_dividers(axis: str, n_lanes: int):
    """boundaries between lanes; nearest-to-centre boundary is the (solid) road_divider."""
    half = n_lanes * LANE_W / 2.0
    lines = []
    for i in range(1, n_lanes):
        b = -half + i * LANE_W
        name = "road_divider" if abs(b) < 1e-6 else "lane_divider"
        if axis == "EW":
            poly = np.array([[-EXT, b], [EXT, b]])
        else:
            poly = np.array([[b, -EXT], [b, EXT]])
        lines.append((poly, name))
    # if even #lanes the centreline coincides with a boundary already named road_divider;
    # if odd, add an explicit centre road_divider so two-way separation is visible
    if n_lanes % 2 == 1:
        if axis == "EW":
            lines.append((np.array([[-EXT, 0.0], [EXT, 0.0]]), "road_divider"))
        else:
            lines.append((np.array([[0.0, -EXT], [0.0, EXT]]), "road_divider"))
    return lines


def _add_band(scene: Scene, axis: str, n_lanes: int):
    poly, half = _band(axis, n_lanes)
    for name in ("generic_drivable_areas", "road_segments", "lane_groups_polygons"):
        scene.add_static(name, poly)
    scene.divider_lines.extend(_band_dividers(axis, n_lanes))
    return half


def _curved_centerline(R=80.0, span_deg=60.0, npts=24):
    span = math.radians(span_deg)
    phis = np.linspace(-span / 2, span / 2, npts)
    return np.stack([R * np.sin(phis), R - R * np.cos(phis)], axis=1)  # passes through origin


def _offset_polyline(cl: np.ndarray, d: float) -> np.ndarray:
    out = []
    for i in range(len(cl)):
        a = cl[max(0, i - 1)]
        b = cl[min(len(cl) - 1, i + 1)]
        t = b - a
        n = np.array([-t[1], t[0]])
        n = n / (np.linalg.norm(n) + 1e-9)
        out.append(cl[i] + n * d)
    return np.array(out)


def build_scene(dsl: dict, road_type: str) -> Scene:
    road_type = (road_type or "Straight").strip()
    actors = parse_actors(dsl)
    n_lanes = get_num_lanes(dsl)
    scene = Scene(road_type=road_type, notes={"n_lanes": n_lanes, "n_actors": len(actors)})

    if road_type == "Curve":
        cl = _curved_centerline()
        road_half = n_lanes * LANE_W / 2.0
        left = _offset_polyline(cl, road_half)
        right = _offset_polyline(cl, -road_half)
        strip = np.concatenate([left, right[::-1]], axis=0)
        for name in ("generic_drivable_areas", "road_segments", "lane_groups_polygons"):
            scene.add_static(name, strip)
        scene.divider_lines.append((cl, "road_divider"))
        for i in range(1, n_lanes):
            off = -road_half + i * LANE_W
            if abs(off) < 1e-6:
                continue
            scene.divider_lines.append((_offset_polyline(cl, off), "lane_divider"))
        _place_agents(scene, actors, axis="EW", road_type=road_type, centerline=cl)
        return scene

    if road_type == "Intersection":
        _add_band(scene, "EW", n_lanes)
        _add_band(scene, "NS", n_lanes)
        half = n_lanes * LANE_W / 2.0
        scene.add_static("intersections", np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]]))
        # crosswalks just outside each arm
        cw = 2.5
        scene.add_static("crosswalks", np.array(
            [[half, -half], [half + cw, -half], [half + cw, half], [half, half]]))
        scene.add_static("crosswalks", np.array(
            [[-half - cw, -half], [-half, -half], [-half, half], [-half - cw, half]]))
        scene.add_static("crosswalks", np.array(
            [[-half, half], [half, half], [half, half + cw], [-half, half + cw]]))
        scene.add_static("crosswalks", np.array(
            [[-half, -half - cw], [half, -half - cw], [half, -half], [-half, -half]]))
        _place_agents(scene, actors, axis="BOTH", road_type=road_type)
        return scene

    if road_type == "T-intersection":
        stem = get_stem_direction(dsl)
        scene.notes["stem"] = stem
        if stem in ("South", "North"):
            through, stem_axis = "EW", "NS"
        else:
            through, stem_axis = "NS", "EW"
        _add_band(scene, through, n_lanes)
        half = n_lanes * LANE_W / 2.0
        # stem half-band from the junction outward toward stem direction
        if stem == "South":
            stem_poly = np.array([[-half, -EXT], [-half, 0], [half, 0], [half, -EXT]])
        elif stem == "North":
            stem_poly = np.array([[-half, 0], [-half, EXT], [half, EXT], [half, 0]])
        elif stem == "East":
            stem_poly = np.array([[0, -half], [EXT, -half], [EXT, half], [0, half]])
        else:  # West
            stem_poly = np.array([[-EXT, -half], [0, -half], [0, half], [-EXT, half]])
        for name in ("generic_drivable_areas", "road_segments", "lane_groups_polygons"):
            scene.add_static(name, stem_poly)
        scene.add_static("intersections", np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]]))
        _place_agents(scene, actors, axis="BOTH", road_type=road_type)
        return scene

    if road_type == "Merging":
        half = _add_band(scene, "EW", n_lanes)
        # on-ramp merging from the south-west into the right (south) edge of the main road,
        # built as a proper lane-width band along a diagonal centreline
        ramp_cl = np.array([[-46.0, -26.0], [-26.0, -18.0], [-6.0, -half - 0.5]])
        ramp = np.concatenate(
            [_offset_polyline(ramp_cl, LANE_W / 2.0),
             _offset_polyline(ramp_cl, -LANE_W / 2.0)[::-1]], axis=0)
        for name in ("generic_drivable_areas", "road_segments", "lane_group_connectors"):
            scene.add_static(name, ramp)
        scene.divider_lines.append((ramp_cl, "lane_divider"))
        _place_agents(scene, actors, axis="EW", road_type=road_type)
        return scene

    # default: Straight
    axis = _primary_axis(actors)
    _add_band(scene, axis, n_lanes)
    _place_agents(scene, actors, axis=axis, road_type=road_type)
    return scene


def _place_agents(scene: Scene, actors: List[dict], axis: str, road_type: str,
                  centerline: Optional[np.ndarray] = None):
    """Place each actor in its travel lane, approaching the origin (collision point)."""
    by_dir: Dict[str, int] = {}
    for idx, a in enumerate(actors):
        d = norm_direction(a["init"])
        yaw, unit, side = DIRECTION[d]
        ux, uy = unit
        dx, dy = size_for_model(a["model"])
        lane_idx = by_dir.get(d, 0)
        by_dir[d] = lane_idx + 1

        lateral = (LANE_W / 2.0 + lane_idx * LANE_W) * side
        # longitudinal: nose ~ a few metres before origin, staggered per same-direction vehicle
        back = dx / 2.0 + 3.0 + lane_idx * 7.0

        if d in ("W2E", "E2W"):  # travel along x -> lateral is y
            cx = -ux * back
            cy = lateral
        else:                    # travel along y -> lateral is x
            cx = lateral
            cy = -uy * back

        is_ego = (idx == 0)  # Vehicle_1 is the case/ego vehicle in SAFE
        scene.agents.append(Agent(cx=cx, cy=cy, yaw=yaw, dx=dx, dy=dy,
                                  is_ego=is_ego, label=f"V{idx + 1}",
                                  ux=ux, uy=uy, speed_mps=_speed_mps(a["speed"]),
                                  action=str(a["action"] or "")))


def _speed_mps(speed):
    try:
        mph = float(str(speed).lower().replace("mph", "").strip().split()[0])
    except Exception:
        mph = 25.0
    return max(3.0, min(mph, 75.0)) * 0.44704


def simulate(scene: Scene, frames: int, dt: float):
    """Kinematic rollout. (cx,cy) is the ~mid-sequence pose, so each agent starts back along
    its heading and the agents converge near the origin (the collision) mid-way through."""
    half = dt * frames / 2.0
    state = []
    for ag in scene.agents:
        v = ag.speed_mps
        state.append({
            "x": ag.cx - ag.ux * v * half, "y": ag.cy - ag.uy * v * half,
            "yaw": ag.yaw, "dx": ag.dx, "dy": ag.dy, "is_ego": ag.is_ego,
            "v": v, "act": (ag.action or "").lower(),
            "decel": v / max(0.1, 0.6 * frames * dt),
        })
    seq = []
    for _ in range(frames):
        seq.append([(s["x"], s["y"], s["yaw"], s["dx"], s["dy"], s["is_ego"]) for s in state])
        for s in state:
            _step(s, dt)
    return seq


def _step(s, dt):
    act = s["act"]
    if "stop" in act and "forward" not in act:          # decelerate to a halt
        s["v"] = max(0.0, s["v"] - s["decel"] * dt)
    d2o = math.hypot(s["x"], s["y"])
    if ("left" in act or "right" in act) and d2o < 14.0:  # turn through the junction
        rate = math.radians(55.0)
        s["yaw"] += (rate if "left" in act else -rate) * dt
    if ("lane" in act or "merg" in act) and d2o > 6.0:   # gentle one-lane lateral drift
        perpx, perpy = -math.sin(s["yaw"]), math.cos(s["yaw"])
        s["x"] += perpx * 1.2 * dt
        s["y"] += perpy * 1.2 * dt
    s["x"] += s["v"] * math.cos(s["yaw"]) * dt
    s["y"] += s["v"] * math.sin(s["yaw"]) * dt


# ------------------------------------------------------------------------------------------
# public entry
def synthesize_bev(dsl: dict, road_type: str, canvas=(200, 200), patch=(100, 100)):
    """Return (bev18 int8 (18,H,W), scene)."""
    scene = build_scene(dsl, road_type)
    bev, ego_poly = rasterize(scene, canvas, patch)
    scene.notes["ego_canvas_polygon"] = None if ego_poly is None else ego_poly.tolist()
    return bev.astype(np.int8), scene


def _save_case(case_id, dsl, road_type, out_dir, canvas, patch, frames=1, fps=6):
    import map_visualizer_nuplan as _mv
    scene = build_scene(dsl, road_type)
    base, affine = rasterize_static(scene, canvas, patch)
    npz_path = os.path.join(out_dir, f"{case_id}.npz")
    png_path = os.path.join(out_dir, f"{case_id}.png")
    gif_path = os.path.join(out_dir, f"{case_id}.gif")
    json_path = os.path.join(out_dir, f"{case_id}.json")

    animated = bool(frames and frames > 1)
    if animated:
        dt = 1.0 / fps
        seq = simulate(scene, frames, dt)
        stack = np.zeros((frames, 18, canvas[0], canvas[1]), dtype=np.int8)
        imgs = []
        for fi, states in enumerate(seq):
            bevf = base.copy()
            ego = draw_agents(bevf, states, affine)
            stack[fi] = bevf.astype(np.int8)
            imgs.append(Image.fromarray(
                visualize_map_nuplan(bevf, target_size=512, ego_canvas_polygon=ego)))
        np.savez_compressed(npz_path, gt_bev_masks=stack)          # (T,18,H,W)
        imgs[len(imgs) // 2].save(png_path)                        # representative frame
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, disposal=2)
        rep = stack[len(stack) // 2]
        bev_shape = list(stack.shape)
    else:
        bevf = base.copy()
        states = [(a.cx, a.cy, a.yaw, a.dx, a.dy, a.is_ego) for a in scene.agents]
        ego = draw_agents(bevf, states, affine)
        rep = bevf.astype(np.int8)
        np.savez_compressed(npz_path, gt_bev_masks=rep)            # (18,H,W)
        Image.fromarray(
            visualize_map_nuplan(rep, target_size=512, ego_canvas_polygon=ego)).save(png_path)
        bev_shape = list(rep.shape)

    used = [n for n in _mv.ALL_CLASSES if rep[CH[n]].any()]
    sidecar = {
        "case_id": str(case_id), "road_type": road_type,
        "n_lanes": scene.notes.get("n_lanes"), "stem": scene.notes.get("stem"),
        "animated": animated, "frames": int(frames) if animated else 1, "fps": fps,
        "agents": [{"label": a.label, "is_ego": a.is_ego,
                    "cx": round(a.cx, 2), "cy": round(a.cy, 2),
                    "yaw_deg": round(math.degrees(a.yaw), 1),
                    "speed_mph": round(a.speed_mps / 0.44704, 1),
                    "action": a.action, "size": [a.dx, a.dy]} for a in scene.agents],
        "channels_used": used,
        "npz": os.path.basename(npz_path), "png": os.path.basename(png_path),
        "gif": os.path.basename(gif_path) if animated else None,
        "bev_shape": bev_shape,
    }
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[bev] {case_id}: {'anim ' + str(frames) + 'f' if animated else 'static'} "
          f"shape={bev_shape} used={used}")
    return sidecar


def _demo_dsls():
    return [
        ("demo_straight_same", "Straight", {
            "Road network": {"Number of lanes": 4},
            "Actors": [{"Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 30},
                        "Vehicle_2": {"Model": "SUV", "Initial_position": "W2E",
                                      "Actions": "Stop", "Speed": 0}}]}),
        ("demo_straight_opp", "Straight", {
            "Road network": {"Number of lanes": 2},
            "Actors": [{"Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 40},
                        "Vehicle_2": {"Model": "Pickup", "Initial_position": "E2W",
                                      "Actions": "Enter the Wrong Way", "Speed": 35}}]}),
        ("demo_curve", "Curve", {
            "Road network": {"Number of lanes": 2},
            "Actors": [{"Vehicle_1": {"Model": "SUV", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 25}}]}),
        ("demo_intersection", "Intersection", {
            "Road network": {"Number of lanes": 4},
            "Actors": [{"Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 30},
                        "Vehicle_2": {"Model": "Minivan", "Initial_position": "S2N",
                                      "Actions": "Move forward", "Speed": 25}}]}),
        ("demo_t_intersection", "T-intersection", {
            "Road network": {"Number of lanes": 2, "Stem road direction": "South"},
            "Actors": [{"Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 30},
                        "Vehicle_2": {"Model": "SUV", "Initial_position": "S2N",
                                      "Actions": "turn left", "Speed": 15}}]}),
        ("demo_merging", "Merging", {
            "Road network": {"Number of lanes": 4},
            "Actors": [{"Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                                      "Actions": "Move forward", "Speed": 50},
                        "Vehicle_2": {"Model": "Semi Truck", "Initial_position": "W2E",
                                      "Actions": "Merge", "Speed": 45}}]}),
    ]


def main():
    ap = argparse.ArgumentParser(description="Synthesize nuPlan-style BEVs from SAFE DSLs.")
    ap.add_argument("--dsl", default="", help="DSL_extraction_results.pkl")
    ap.add_argument("--meta", default="", help="meta_data_results.pkl (for road_type per case)")
    ap.add_argument("--out", default="./output")
    ap.add_argument("--demo", action="store_true", help="render one BEV per road type, no LLM")
    ap.add_argument("--res", default="200", choices=["200", "400"],
                    help="canvas resolution (200->[-50,50]@0.5, 400->@0.25)")
    ap.add_argument("--frames", type=int, default=1,
                    help="temporal frames; >1 renders an animated GIF + a (T,18,H,W) npz stack")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    canvas = (200, 200) if args.res == "200" else (400, 400)
    patch = (100, 100)
    os.makedirs(args.out, exist_ok=True)
    manifest = []

    if args.demo:
        for case_id, road_type, dsl in _demo_dsls():
            manifest.append(_save_case(case_id, dsl, road_type, args.out, canvas, patch,
                                       frames=args.frames, fps=args.fps))
    else:
        if not args.dsl:
            raise SystemExit("provide --dsl <DSL_extraction_results.pkl> or --demo")
        with open(args.dsl, "rb") as f:
            dsls = pickle.load(f)
        road_by_case = {}
        if args.meta and os.path.exists(args.meta):
            with open(args.meta, "rb") as f:
                for row in pickle.load(f):
                    road_by_case[str(row[-1])] = row[0]
        for dsl in dsls:
            case_id = str(dsl.get("Scenario", f"case_{len(manifest)}"))
            road_type = road_by_case.get(case_id) or dsl.get("Road network", {}).get("Road type") or "Straight"
            manifest.append(_save_case(case_id, dsl, road_type, args.out, canvas, patch,
                                       frames=args.frames, fps=args.fps))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"cases": manifest,
                   "legend": legend_items()}, f, indent=2)
    print(f"[bev] wrote {len(manifest)} case(s) + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
