"""Export SAFE crash scenes as UniScene-v2 BEV->occ samples (one per *vehicle ego-view*).

UniScene's `Nuplan_Occ_bev_HR_mini` dataloader (uniscene/occupancy_generation/dataset/
dataload_util.py) consumes, per frame ("token"):
  * `<bev_root>/<token>.npz`  key `gt_bev_masks` -> (18, 400, 400) int8, ego-centric, 0.25 m/px
  * `<gts_root>/<token>.npy`  -> (400, 400, 32) occupancy   (built by build_ciren_uniscene_index.py)
and groups each car-view's consecutive tokens into a clip via a global index pkl.  The OccDiT model only reads
merged-static (ch 0-7 -> ch 1) + dynamic (ch 8-13); dividers 15-17 are NOT read by BEV->occ.

This module turns ONE SAFE scene (DSL + road type) into N car-views (one per vehicle), each a
dense clip (every timestep) rendered in *that vehicle's* ego frame (vehicle centred, heading up), reusing the
existing SAFE geometry/kinematics verbatim:
  - `bev_from_dsl.build_scene / simulate / _stage_collision / _latch_impact`  (scene + crash rollout)
  - `uniscene_bev._rasterize_ego`  (re-rasterize the world in a chosen vehicle's frame -> (18,H,W))

Because the exported `gt_boxes` use the SAME ego transform `_rasterize_ego` uses to *draw* the
vehicles, the box records and the rasterized vehicle channel (8) are consistent by construction.

Schema deltas handled vs. the stock SAFE renderer (output/*.npz):
  1. Resolution -> rasterize at canvas (400,400), patch (100,100) == generator's 0.25 m/px affine.
  2. Rank -> one (18,400,400) npz per timestep (not a (T,18,H,W) stack).
  3. Divider channels -> `to_generator_channels()` remaps SAFE's 15=lane/16=crosswalk/17=road onto
     the UniScene generator truth (15 = merged divider, 16 = ped_crossing, 17 = road boundary).
     (Harmless for BEV->occ, which ignores 15-17; kept for broader UniScene map-conditioning compat.)

Per case we also write `pkl_records.pkl` (numpy box/pose records the global indexer concatenates),
`boxes.npz` (audit), and human-inspectable scene + per-car GIF/PNGs.  Nothing here imports torch.
"""
import argparse
import glob
import math
import os
import pickle

import numpy as np
from PIL import Image

try:
    import cv2
except Exception:  # boundary (ch 17) falls back to a numpy edge if cv2 is unavailable
    cv2 = None

import bev_from_dsl as B
import uniscene_bev as U
from map_visualizer_nuplan import CH

# UniScene nuPlan grid: 400x400 @ patch 100 m  ->  0.25 m/px, range [-50, +50] m.
CANVAS = (400, 400)
PATCH = (100, 100)
DEFAULT_VEH_HEIGHT = 1.6   # fallback roof height (m) when the Model matches no class
GROUND_Z = 0.0
Z_RES = 0.25               # occ vertical resolution (m per z-bin); height(m) -> round(h / Z_RES) bins

# add_bev_layout drivable union (excludes walkways=2); used for the ch-17 boundary edge.
_DRIV_STATIC = [0, 1, 3, 4, 5, 6, 7]


# ---------------------------------------------------------------------------------------------
# divider remap (preserves prior UniScene-v2 generator-truth audit; harmless for BEV->occ)
def to_generator_channels(bev):
    """Remap SAFE's divider channels (15=lane_divider,16=crosswalk_line,17=road_divider) onto the
    UniScene-v2 generator truth (15=merged divider, 16=ped_crossing, 17=road boundary). Channels
    0-14 are untouched, so the BEV->occ model (which reads only 1,8-13) is unaffected."""
    out = bev.copy()
    lane = bev[CH["lane_divider"]].astype(bool)
    road = bev[CH["road_divider"]].astype(bool)
    out[15] = (lane | road).astype(bev.dtype)                 # all dividers merged (label 0)
    out[16] = bev[CH["crosswalk_line"]].astype(bev.dtype)     # ped_crossing line  (label 1)
    drivable = np.any(bev[_DRIV_STATIC], axis=0).astype(np.uint8)
    if cv2 is not None:
        edge = cv2.morphologyEx(drivable, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    else:
        edge = np.zeros_like(drivable)
        edge[1:-1, 1:-1] = (drivable[1:-1, 1:-1] &
                            ~(drivable[:-2, 1:-1] & drivable[2:, 1:-1] &
                              drivable[1:-1, :-2] & drivable[1:-1, 2:]))
    out[17] = edge.astype(bev.dtype)                          # road boundary / solid edge (label 2)
    return out


# ---------------------------------------------------------------------------------------------
# small helpers
def nuscenes_name(model) -> str:
    """SAFE vehicle 'Model' string -> a nuScenes-style class name (for gt_names / taxonomy remap).
    All map to nuPlan 'vehicle' downstream, so granularity here is cosmetic/provenance only."""
    m = str(model or "").strip().lower()
    if any(k in m for k in ("semi", "truck", "lorry", "tractor")):
        return "truck"
    if "bus" in m:
        return "bus"
    if "trailer" in m:
        return "trailer"
    if any(k in m for k in ("motorcycle", "motorbike", "moped")):
        return "motorcycle"
    if any(k in m for k in ("bicycle", "bike", "cyclist")):
        return "bicycle"
    return "car"


# One z value (roof height, m) tagged on every moving object, from its LLM-extracted 'Model'
# string (most-specific keyword first). This is a semantic class-prior table for now; the custom
# BEV creator will later let the LLM output heights directly, to verify against these.
CLASS_HEIGHT = [
    ("ambulance", 2.6), ("fire truck", 3.3), ("fire", 3.3), ("emergency", 2.6),
    ("semi", 3.9), ("tractor", 3.9), ("lorry", 3.5), ("trailer", 4.0),
    ("pickup", 1.9), ("minivan", 1.8), ("suv", 1.8), ("van", 2.2),
    ("bus", 3.2), ("box truck", 3.0), ("truck", 3.5),
    ("motorcycle", 1.5), ("motorbike", 1.5), ("moped", 1.5), ("scooter", 1.5),
    ("bicycle", 1.7), ("cyclist", 1.7), ("bike", 1.7),
    ("pedestrian", 1.75), ("person", 1.75),
    ("sedan", 1.5), ("coupe", 1.45), ("hatchback", 1.5), ("wagon", 1.55), ("car", 1.5),
]


def height_for_model(model) -> float:
    """Roof height (m) for a moving object from its 'Model' string; default if no class matches."""
    m = str(model or "").lower()
    for key, h in CLASS_HEIGHT:
        if key in m:
            return h
    return DEFAULT_VEH_HEIGHT


def yaw_to_quat(yaw) -> list:
    """Rotation of `yaw` about +z as pyquaternion-order [w, x, y, z] (loader uses Quaternion(...))."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _ego_transform(ex, ey, eyaw):
    """The exact transform `uniscene_bev._rasterize_ego` applies: rotate world so the ego heading
    points to image-up (-row), ego at origin.  Returns (R, origin, alpha)."""
    alpha = -math.pi / 2.0 - eyaw
    ca, sa = math.cos(alpha), math.sin(alpha)
    R = np.array([[ca, -sa], [sa, ca]])
    origin = np.array([ex, ey], dtype=float)
    return R, origin, alpha


def ego_frame_states(states, ego_idx):
    """Map every agent world state into vehicle `ego_idx`'s frame.
    states: list of (cx, cy, yaw, dx, dy, is_ego).  Returns the same tuples in ego coords."""
    ex, ey, eyaw = states[ego_idx][0], states[ego_idx][1], states[ego_idx][2]
    R, origin, alpha = _ego_transform(ex, ey, eyaw)
    out = []
    for (cx, cy, yaw, dx, dy, is_ego) in states:
        c = (np.array([cx, cy], dtype=float) - origin) @ R.T
        out.append((float(c[0]), float(c[1]), float(yaw + alpha), float(dx), float(dy), bool(is_ego)))
    return out


def states_to_nuscenes_boxes(ego_states, names, heights):
    """(N,7) nuScenes gt_boxes [x, y, z, w, l, h, yaw] + gt_names, in the ego frame.
    SAFE lays length (dx) along heading and width (dy) lateral, matching nuScenes l/w. The h column
    carries each object's per-class roof height; z is the box centre (height/2 off the ground)."""
    boxes = np.zeros((len(ego_states), 7), dtype=np.float32)
    gt_names = []
    for i, (cx, cy, yaw, dx, dy, _is_ego) in enumerate(ego_states):
        h = heights[i] if i < len(heights) else DEFAULT_VEH_HEIGHT
        boxes[i] = [cx, cy, GROUND_Z + h / 2.0, dy, dx, h, yaw]   # w=dy, l=dx, per-object h
        gt_names.append(names[i] if i < len(names) else "car")
    return boxes, gt_names


def rasterize_ego_18ch(scene, states, ego_idx) -> np.ndarray:
    """(18,400,400) uint8 ego-centric BEV of vehicle `ego_idx` for one frame, divider-remapped."""
    bev, _ego_poly = U._rasterize_ego(scene, states, ego_idx, canvas=CANVAS, patch=PATCH)
    return to_generator_channels(bev)


def ego_veh_height_map(scene, states, ego_idx, heights) -> np.ndarray:
    """(400,400) float32 per-pixel vehicle roof-height (m), ego-centric, pixel-aligned with BEV ch 8.
    Each agent's footprint is stamped with its height (max where footprints overlap). Lets the occ
    extruder give every object its own z extent instead of a single constant."""
    ex, ey, eyaw = states[ego_idx][0], states[ego_idx][1], states[ego_idx][2]
    R, origin, _alpha = _ego_transform(ex, ey, eyaw)
    affine = B.lidar2canvas(CANVAS, PATCH)
    hmap = np.zeros(CANVAS, dtype=np.float32)
    for i, (cx, cy, yaw, dx, dy, _is_ego) in enumerate(states):
        ce = (B.box_corners_bev(cx, cy, yaw, dx, dy) - origin) @ R.T   # footprint in ego frame
        m = B._draw_filled(np.zeros(CANVAS, dtype=np.uint8), [ce], affine).astype(bool)
        h = heights[i] if i < len(heights) else DEFAULT_VEH_HEIGHT
        hmap[m] = np.maximum(hmap[m], h)
    return hmap


# ---------------------------------------------------------------------------------------------
# scene -> rollout
def simulate_scene(dsl, road_type, frames=5, fps=2, step=4):
    """Build the scene; simulate at FINE resolution (dt/step) so collision staging + latching
    reliably catch the overlap (a coarse 5-frame rollout can tunnel past fast-closing crashes),
    then subsample to `frames` uniformly-spaced output frames at output dt = 1/fps.

    Returns (scene, seq_out, impact_out, seq_fine, impact_fine):
      seq_out    : the `frames`-length clip we tokenise for UniScene (output dt uniform).
      impact_out : index into seq_out of the (held) crash, or None.
      seq_fine   : the dense rollout (for smooth human GIFs).
      impact_fine: index into seq_fine of the crash, or None.
    """
    scene = B.build_scene(dsl, road_type)
    dt_out = 1.0 / fps
    step = max(1, int(step))
    fine = (frames - 1) * step + 1
    dt_fine = dt_out / step
    seq_fine = B.simulate(scene, fine, dt_fine)
    B._stage_collision(scene, seq_fine)
    impact_fine = B._latch_impact(seq_fine, scene.agents)
    idxs = [j * step for j in range(frames)]                 # [0, step, 2*step, ...]
    seq_out = [seq_fine[i] for i in idxs]
    # first output frame at/after the impact shows the held crash: ceil(impact_fine / step)
    impact_out = None if impact_fine is None else min(frames - 1, (impact_fine + step - 1) // step)
    actors = B.parse_actors(dsl)
    scene.notes["nuscenes_names"] = [nuscenes_name(a.get("model")) for a in actors]
    scene.notes["models"] = [str(a.get("model") or "") for a in actors]            # raw LLM model
    scene.notes["heights"] = [height_for_model(a.get("model")) for a in actors]   # one z per object
    return scene, seq_out, impact_out, seq_fine, impact_fine


# ---------------------------------------------------------------------------------------------
# per car-view export
def export_car_views(case_token, scene, seq, *, bev_dir, dt):
    """Write one BEV npz per (vehicle, frame) token; return the records the global indexer needs.

    Returns (records, car_views):
      records  : list of per-token dicts (token, anns{gt_boxes,gt_names,gt_velocity_3d}, poses) in
                 clip order (car-view 0 frames 0..4, car-view 1 frames 0..4, ...).
      car_views: list of {label, ego_idx, tokens:[5 tokens]} (one per vehicle).
    """
    os.makedirs(bev_dir, exist_ok=True)
    n_agents = len(scene.agents)
    frames = len(seq)
    names_all = scene.notes.get("nuscenes_names") or []
    heights_all = scene.notes.get("heights") or []
    records, car_views = [], []

    for k in range(n_agents):
        label = scene.agents[k].label or f"V{k + 1}"
        ego_states_f = [ego_frame_states(seq[f], k) for f in range(frames)]
        boxes_f, names_f = [], None
        for f in range(frames):
            bx, nm = states_to_nuscenes_boxes(ego_states_f[f], names_all, heights_all)
            boxes_f.append(bx)
            if names_f is None:
                names_f = nm

        # ego-frame velocity by finite difference of box centres (last frame repeats previous)
        vel_f = []
        for f in range(frames):
            g = min(f + 1, frames - 1)
            if g == f:
                vel = np.zeros((n_agents, 3), dtype=np.float32)
            else:
                d = (boxes_f[g][:, :2] - boxes_f[f][:, :2]) / max(dt, 1e-6)
                vel = np.concatenate([d, np.zeros((n_agents, 1), dtype=np.float32)], axis=1).astype(np.float32)
            vel_f.append(vel)
        if frames >= 2:
            vel_f[-1] = vel_f[-2]

        tokens = []
        for f in range(frames):
            token = f"{case_token}_v{k + 1}_f{f}"
            tokens.append(token)
            bev = rasterize_ego_18ch(scene, seq[f], k).astype(np.int8)
            hmap = ego_veh_height_map(scene, seq[f], k, heights_all).astype(np.float16)
            np.savez_compressed(os.path.join(bev_dir, f"{token}.npz"),
                                gt_bev_masks=bev, gt_veh_height=hmap)

            wx, wy, wyaw = seq[f][k][0], seq[f][k][1], seq[f][k][2]   # car k world pose at frame f
            records.append({
                "token": token,
                "anns": {
                    "gt_boxes": boxes_f[f],
                    "gt_names": list(names_f),
                    "gt_velocity_3d": vel_f[f],
                },
                "ego2global_translation": [float(wx), float(wy), 0.0],
                "ego2global_rotation": yaw_to_quat(float(wyaw)),
                "lidar2ego_translation": [0.0, 0.0, 0.0],
                "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "driving_command": [0, 0, 0, 0],
            })
        car_views.append({"label": label, "ego_idx": k, "tokens": tokens})
    return records, car_views


# ---------------------------------------------------------------------------------------------
# human-inspectable renders (scene view + one ego view per car), UniScene-styled
def export_human_views(case_id, scene, seq, impact, sample_dir, fps):
    scene_dir = os.path.join(sample_dir, "scene")
    ego_dir = os.path.join(sample_dir, "ego")
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(ego_dir, exist_ok=True)
    rep = impact if impact is not None else len(seq) // 2

    base, affine = B.rasterize_static(scene, U.CANVAS, U.PATCH)
    imgs = []
    for states in seq:
        bevf = base.copy()
        ego = B.draw_agents(bevf, states, affine)
        imgs.append(Image.fromarray(U.render_uniscene(bevf, 512, ego, rotate90=True)))
    U._save_gif(imgs, os.path.join(scene_dir, f"{case_id}_uniscene.gif"), fps, rep)

    labels = [a.label or f"V{i + 1}" for i, a in enumerate(scene.agents)]
    for ego_idx, label in enumerate(labels):
        imgs = []
        for states in seq:
            bev, ego_poly = U._rasterize_ego(scene, states, ego_idx)   # 200x200 is fine for viewing
            imgs.append(Image.fromarray(U.render_uniscene(bev, 512, ego_poly, rotate90=False)))
        U._save_gif(imgs, os.path.join(ego_dir, f"{case_id}_ego_{label}.gif"), fps, rep)
    return labels


# ---------------------------------------------------------------------------------------------
# top-level per-case entry
def export_case(case_id, case_token, dsl, road_type, *, bev_dir, sample_dir,
                frames=5, fps=2, step=4, human_fps=8, write_human=True):
    """Full per-case export. Writes BEV npz (per token), pkl_records.pkl, boxes.npz, human views.
    Returns a summary dict for provenance.

    We tokenise the DENSE rollout (every timestep, the same frames the human GIF uses), not the
    coarse `frames`-length subsample -- so each car-view clip is temporally dense. The clip length
    is len(seq_fine) = (frames-1)*step+1 at dt = (1/fps)/step (UniScene's Tframe is configurable).
    """
    os.makedirs(sample_dir, exist_ok=True)
    scene, _seq_out, _impact_out, seq_fine, impact_fine = simulate_scene(
        dsl, road_type, frames=frames, fps=fps, step=step)
    seq, impact = seq_fine, impact_fine               # dense: tokenise every timestep
    dt = (1.0 / fps) / max(1, step)
    records, car_views = export_car_views(case_token, scene, seq, bev_dir=bev_dir, dt=dt)

    with open(os.path.join(sample_dir, "pkl_records.pkl"), "wb") as f:
        pickle.dump({"case_id": case_id, "case_token": case_token,
                     "records": records, "car_views": car_views,
                     "road_type": road_type, "impact_frame": impact, "dt": dt,
                     "nuscenes_names": scene.notes.get("nuscenes_names"),
                     "models": scene.notes.get("models"),
                     "heights": scene.notes.get("heights")}, f)

    np.savez_compressed(os.path.join(sample_dir, "boxes.npz"),
                        **{r["token"]: r["anns"]["gt_boxes"] for r in records})

    if write_human:                                  # smooth GIFs from the dense rollout
        export_human_views(case_id, scene, seq_fine, impact_fine, sample_dir, human_fps)

    return {
        "n_vehicles": len(scene.agents),
        "impact_frame": impact,
        "frames": len(seq), "fps": round(1.0 / dt, 2), "dt": dt,
        "single_vehicle": len(scene.agents) <= 1,
        "car_views": [{"label": cv["label"], "ego_idx": cv["ego_idx"], "tokens": cv["tokens"]}
                      for cv in car_views],
        "n_tokens": len(records),
    }


# ---------------------------------------------------------------------------------------------
# standalone smoke test (no LLM): run every demo DSL through the exporter and check shapes.
def _selftest(out_root="/tmp/uniscene_export_selftest"):
    import shutil
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    bev_dir = os.path.join(out_root, "bev")
    ok = True
    for case_id, road_type, dsl in B._demo_dsls():
        token = "ck_" + case_id.replace("demo_", "d")[:24]
        sample_dir = os.path.join(out_root, "samples", case_id)
        summary = export_case(case_id, token, dsl, road_type,
                              bev_dir=bev_dir, sample_dir=sample_dir, frames=5, fps=2)
        with open(os.path.join(sample_dir, "pkl_records.pkl"), "rb") as f:
            recs = pickle.load(f)["records"]
        case_ok = (len(recs) == summary["n_vehicles"] * summary["frames"])
        for r in recs:
            m = np.load(os.path.join(bev_dir, f"{r['token']}.npz"))["gt_bev_masks"]
            case_ok = case_ok and (m.shape == (18, 400, 400) and m.dtype == np.int8
                                   and set(np.unique(m)).issubset({0, 1}) and m[CH["vehicle"]].any())
            case_ok = case_ok and (r["anns"]["gt_boxes"].shape[1] == 7)
        ok = ok and case_ok
        print(f"[selftest] {case_id:22s} road={road_type:14s} "
              f"vehicles={summary['n_vehicles']} tokens={len(recs)} "
              f"impact={summary['impact_frame']} {'OK' if case_ok else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL", "->", out_root)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Export SAFE scenes as UniScene BEV->occ tokens.")
    ap.add_argument("--selftest", action="store_true", help="run the no-LLM demo smoke test")
    ap.add_argument("--dsl", default="", help="DSL_extraction_results.pkl")
    ap.add_argument("--meta", default="", help="meta_data_results.pkl (road_type per case)")
    ap.add_argument("--bev-dir", default="./uniscene_bev")
    ap.add_argument("--sample-root", default="./uniscene_samples")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    if args.selftest or not args.dsl:
        _selftest()
        return
    with open(args.dsl, "rb") as f:
        dsls = pickle.load(f)
    road_by_case = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, "rb") as f:
            for row in pickle.load(f):
                road_by_case[str(row[-1])] = row[0]
    for d in dsls:
        cid = str(d.get("Scenario"))
        rt = road_by_case.get(cid) or d.get("Road network", {}).get("Road type") or "Straight"
        export_case(cid, f"ck_{cid}", d, rt,
                    bev_dir=args.bev_dir, sample_dir=os.path.join(args.sample_root, cid),
                    frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
