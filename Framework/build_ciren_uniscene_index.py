#!/usr/bin/env python3
"""Stage B: extrude per-token occupancy + assemble the global UniScene index pkl.

Run AFTER build_ciren_uniscene_dataset.py.  Core (occ + pkl) is pure numpy; --aux / --cross-check
additionally import mmdet3d via UniScene's bridge_nuplan, so run those in the `uniscene_occ` env.

For every token (one per vehicle-view per frame) it:
  1. Extrudes a coarse 3D occupancy from the token's BEV:  occ[(400,400,32)] uint8, class ids
     (nuplan-occ.yaml: 0=free, 1=background/road, 2=vehicle).  The OccDiT loader requires a per-token
     occ file even for pure generation; it is used for the `occ_meta` density vector + IoU (the
     diffusion itself starts from noise when LAMBDA_NOISE_PRIOR=0).  A road slab (drivable footprint,
     low z-bins) + vehicle boxes (ch8 footprint, ~0..1.7 m) give a non-degenerate, BEV-aligned occ.
  2. Assembles the global index pkl consumed by `Nuplan_Occ_bev_HR_mini`:
       {infos:[{token, anns{gt_boxes,gt_names,gt_velocity_3d}, ego2global_*, lidar2ego_*,
                driving_command}], scene_tokens:[[dense tok]/car-view], clip_infos:[[dense idx]/car-view]}
  3. (--aux)         splice gt_aux_bev (7,400,400) into each token npz via bridge rasterize_aux.
  4. (--cross-check) report vehicle-channel IoU of SAFE's ch8 vs bridge rasterize_dynamic.

The occ files are COMPRESSED .npz (key 'occ'); the (400,400,32) uint8 occ is ~99.5% zeros so each is
~7 KB (~700x smaller than a raw .npy), i.e. ~0.4 GB for the whole CIREN pass instead of ~332 GB.  The
loader (Nuplan_Occ_bev_HR_mini) reads <token>.npz['occ'] and falls back to a legacy <token>.npy.
<out_root>/gts is still symlinked to /mnt/diskA (now barely used).

Examples (uniscene_occ env):
  PYTHONPATH=/home/arthur/uniscene_stack python build_ciren_uniscene_index.py \
      --out-root /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2 \
      --gts-dir  /mnt/diskA/CIREN_uniscene_v2_gts --aux --cross-check
  # core only (no mmdet3d):
  python build_ciren_uniscene_index.py --out-root /tmp/ds --gts-dir /tmp/ds/gts
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

# nuplan-occ.yaml class ids
OCC_FREE, OCC_BG, OCC_VEHICLE = 0, 1, 2


def extrude_occ(bev18, z_bins=32, road_z=2, veh_z=7, veh_height_map=None, z_res=0.25):
    """Coarse 3D occupancy (H, W, z_bins) uint8 from an (18,H,W) BEV, BEV-pixel-aligned.

    background(1) under the merged drivable footprint (static ch 0-7) for the lowest `road_z` bins;
    vehicle(2) under the vehicle footprint. If `veh_height_map` (H,W float metres, the exporter's
    gt_veh_height) is given, every pixel extrudes to its OWN roof height (round(h / z_res) bins), so
    each object gets its tagged z extent; otherwise it falls back to the constant `veh_z` bins.
    """
    H, W = bev18.shape[1], bev18.shape[2]
    occ = np.zeros((H, W, z_bins), dtype=np.uint8)
    drivable = np.any(bev18[0:8].astype(bool), axis=0)     # merged static map -> road/ground
    occ[drivable, :road_z] = OCC_BG
    if veh_height_map is not None:                         # per-object height (one z per object)
        vz = np.clip(np.rint(np.asarray(veh_height_map, np.float32) / z_res), 0, z_bins).astype(np.int32)
        below = np.arange(z_bins)[None, None, :] < vz[:, :, None]   # (H,W,z) below each pixel's roof
        occ[below] = OCC_VEHICLE                           # cars overwrite the slab under them
    else:
        occ[bev18[8].astype(bool), :veh_z] = OCC_VEHICLE
    return occ


def build_index(out_root):
    """Concatenate every sample's pkl_records.pkl into the global infos/scene_tokens/clip_infos."""
    samples = sorted(glob.glob(os.path.join(out_root, "samples", "*", "*", "pkl_records.pkl")))
    infos, scene_tokens, clip_infos = [], [], []
    tok2idx = {}
    for sp in samples:
        with open(sp, "rb") as f:
            rec = pickle.load(f)
        for r in rec["records"]:
            tok = r["token"]
            if tok in tok2idx:
                raise ValueError(f"duplicate token {tok} (from {sp})")
            tok2idx[tok] = len(infos)
            infos.append({
                "token": tok,
                "anns": r["anns"],
                "ego2global_translation": r["ego2global_translation"],
                "ego2global_rotation": r["ego2global_rotation"],
                "lidar2ego_translation": r["lidar2ego_translation"],
                "lidar2ego_rotation": r["lidar2ego_rotation"],
                "driving_command": r["driving_command"],
            })
        for cv in rec["car_views"]:
            toks = list(cv["tokens"])
            scene_tokens.append(toks)
            clip_infos.append([tok2idx[t] for t in toks])
    return {"infos": infos, "scene_tokens": scene_tokens, "clip_infos": clip_infos,
            "original_info_count": len(infos)}, samples


def gen_occ(gts_dir, bev_dir, tokens, z_bins, road_z, veh_z, overwrite, z_res=0.25):
    """Write each token's occupancy as a COMPRESSED .npz (key 'occ').

    The extruded (400,400,32) uint8 occ is ~99.5% zeros, so np.savez_compressed shrinks it ~700x
    (~5.1 MB raw .npy -> ~7 KB) with ZERO loss.  Over the full CIREN pass this turns ~332 GB of occ
    into ~0.4 GB.  The OccDiT loader (Nuplan_Occ_bev_HR_mini) reads <token>.npz['occ'] and falls back
    to a legacy <token>.npy, so this stays a drop-in, generator-independent GT snapshot on disk.
    """
    os.makedirs(gts_dir, exist_ok=True)
    n = 0
    for tok in tokens:
        out = os.path.join(gts_dir, f"{tok}.npz")
        if os.path.exists(out) and not overwrite:
            continue
        npz = np.load(os.path.join(bev_dir, f"{tok}.npz"))
        hmap = npz["gt_veh_height"] if "gt_veh_height" in npz.files else None   # per-object heights
        occ = extrude_occ(npz["gt_bev_masks"], z_bins, road_z, veh_z,
                          veh_height_map=hmap, z_res=z_res)
        np.savez_compressed(out, occ=occ)
        n += 1
    return n


def add_aux(bev_dir, infos):
    """Splice gt_aux_bev (7,400,400) float32 into each token npz (mmdet3d via bridge_nuplan)."""
    from scenario_tools.bridge_nuplan import rasterize as R
    for info in infos:
        tok = info["token"]
        boxes = np.asarray(info["anns"]["gt_boxes"], dtype=np.float32)
        aux = R.rasterize_aux(R.to_nuplan_frame(boxes)).astype(np.float32)
        npz = os.path.join(bev_dir, f"{tok}.npz")
        data = dict(np.load(npz))
        data["gt_aux_bev"] = aux
        np.savez_compressed(npz, **data)


def cross_check(bev_dir, infos, n=8):
    """Vehicle-channel IoU: SAFE's rasterized ch8 vs bridge rasterize_dynamic of the exported boxes."""
    from scenario_tools.bridge_nuplan import rasterize as R
    ious = []
    for info in infos[:n]:
        tok = info["token"]
        boxes = np.asarray(info["anns"]["gt_boxes"], dtype=np.float32)
        labels = R.remap_labels(info["anns"]["gt_names"])
        dyn = R.rasterize_dynamic(R.to_nuplan_frame(boxes), labels)        # (7,400,400)
        bev = np.load(os.path.join(bev_dir, f"{tok}.npz"))["gt_bev_masks"]
        a, b = bev[8].astype(bool), dyn[0].astype(bool)                    # vehicle channels
        ious.append((a & b).sum() / max(1, (a | b).sum()))
    print(f"[cross-check] vehicle-channel IoU vs bridge (n={len(ious)}): "
          f"{[round(float(x), 3) for x in ious]} mean={float(np.mean(ious)):.3f}")
    print("[cross-check] (low IoU => box convention disagrees; SAFE raster is the deliverable, "
          "investigate before trusting bridge dynamic.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--bev-dir", default="", help="default <out_root>/bev")
    ap.add_argument("--gts-dir", default="", help="default <out_root>/gts (point at /mnt/diskA)")
    ap.add_argument("--pkl-out", default="", help="default <out_root>/index/ciren_uniscene_clip_infos.pkl")
    ap.add_argument("--z-bins", type=int, default=32)
    ap.add_argument("--road-z", type=int, default=2, help="z-bins of background road slab")
    ap.add_argument("--veh-z", type=int, default=7, help="fallback z-bins of vehicle extrusion when a token has no gt_veh_height")
    ap.add_argument("--z-res", type=float, default=0.25, help="occ vertical resolution (m/z-bin); height(m) -> round(h/z-res) bins")
    ap.add_argument("--aux", action="store_true", help="add gt_aux_bev (needs uniscene_occ/mmdet3d)")
    ap.add_argument("--cross-check", action="store_true", help="IoU check vs bridge (needs mmdet3d)")
    ap.add_argument("--overwrite-occ", action="store_true")
    ap.add_argument("--uniscene-root", default="/home/arthur/uniscene_stack")
    args = ap.parse_args()

    bev_dir = args.bev_dir or os.path.join(args.out_root, "bev")
    gts_dir = args.gts_dir or os.path.join(args.out_root, "gts")
    pkl_out = args.pkl_out or os.path.join(args.out_root, "index", "ciren_uniscene_clip_infos.pkl")
    os.makedirs(os.path.dirname(pkl_out), exist_ok=True)

    index, samples = build_index(args.out_root)
    tokens = [i["token"] for i in index["infos"]]
    print(f"[index] {len(samples)} samples -> {len(tokens)} tokens, "
          f"{len(index['clip_infos'])} car-view clips")

    n_occ = gen_occ(gts_dir, bev_dir, tokens, args.z_bins, args.road_z, args.veh_z,
                    args.overwrite_occ, z_res=args.z_res)
    print(f"[occ] wrote {n_occ} new compressed occ .npz -> {gts_dir} (shape ({args.z_bins} z), "
          f"~7 KB/token, existing skipped unless --overwrite-occ)")

    # Write the index pkl FIRST: it is the deliverable and needs only numpy. The optional
    # diagnostics below need mmdet3d (uniscene_occ env) and must NEVER block the pkl/occ.
    with open(pkl_out, "wb") as f:
        pickle.dump(index, f)
    print(f"[index] wrote pkl -> {pkl_out}  (infos={len(index['infos'])}, "
          f"clips={len(index['clip_infos'])})")

    if args.aux or args.cross_check:
        if args.uniscene_root not in sys.path:
            sys.path.insert(0, args.uniscene_root)
        try:
            if args.cross_check:
                cross_check(bev_dir, index["infos"])
            if args.aux:
                add_aux(bev_dir, index["infos"])
                print("[aux] spliced gt_aux_bev into every token npz")
        except ImportError as e:
            print(f"[warn] --aux/--cross-check need the uniscene_occ env (mmdet3d), not the host: "
                  f"{e}. Skipped — the pkl + occ are already complete. To run them, use the "
                  f"uniscene-occ Docker image (see run_ciren_dataset.sh 'index-docker').")


if __name__ == "__main__":
    main()
