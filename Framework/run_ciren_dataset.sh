#!/usr/bin/env bash
# Build the CIREN -> SAFE -> UniScene-v2 BEV->occ dataset, end to end.
#
# Two stages in two envs:
#   Stage A (SAFE/vLLM env, CWD=Framework): LLM extraction + per-vehicle BEV tokens + human GIFs
#   Stage B (uniscene_occ env, mmdet3d):    extrude occ + global index pkl (+ aux/cross-check)
#
# One-time setup (needs sudo; you are in the sudo group). The output root lives on /mnt/disk2; occ is
# written COMPRESSED (.npz, ~7 KB/token, ~0.4 GB total) to /mnt/diskA and symlinked in as gts/.
#
#   sudo mkdir -p /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2 /mnt/diskA/CIREN_uniscene_v2_gts
#   sudo chown arthur:arthur /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2 /mnt/diskA/CIREN_uniscene_v2_gts
#   ln -sfn /mnt/diskA/CIREN_uniscene_v2_gts /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2/gts
#
# Usage:
#   bash run_ciren_dataset.sh pilot        # 15 cases, both datasets, then index+occ (+cross-check)
#   bash run_ciren_dataset.sh full         # all ok cases (CIREN2017 then legacy), resumable
#   bash run_ciren_dataset.sh index        # (re)build occ + pkl over whatever is in OUT_ROOT
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-/mnt/disk2/CIREN_dataset/CIREN_uniscene_v2}"
GTS_DIR="${GTS_DIR:-/mnt/diskA/CIREN_uniscene_v2_gts}"
UNISCENE_ROOT="${UNISCENE_ROOT:-/home/arthur/uniscene_stack}"
JOBS="${JOBS:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

stage_a() {  # args passed straight to the driver
  echo "[run] Stage A (SAFE/vLLM): $*"
  python3 build_ciren_uniscene_dataset.py --out-root "$OUT_ROOT" "$@"
}

stage_b() {  # occ + pkl on the HOST (pure numpy; no mmdet3d). --aux/--cross-check are skipped here.
  echo "[run] Stage B (host): occ + index"
  PYTHONPATH="$UNISCENE_ROOT" python3 build_ciren_uniscene_index.py \
      --out-root "$OUT_ROOT" --gts-dir "$GTS_DIR" "$@"
}

stage_b_docker() {  # occ + pkl + aux/cross-check INSIDE the uniscene-occ image (has mmdet3d)
  echo "[run] Stage B (uniscene-occ docker): occ + index + $*"
  docker run --rm \
    -v "$OUT_ROOT":/pilot -v "$GTS_DIR":/gts \
    -v /home/arthur/SAFE-ADS-Testing:/safe -v "$UNISCENE_ROOT":/uniscene_src \
    -e PYTHONPATH=/uniscene_src -w /safe/Framework \
    uniscene-occ python build_ciren_uniscene_index.py \
      --out-root /pilot --gts-dir /gts "$@"
}

# NOTE on occ/aux: the OccDiT BEV->occ loader (Nuplan_Occ_bev_HR_mini) reads ONLY gt_bev_masks +
# the occ .npz + ego poses; it does NOT read infos[].anns.gt_boxes nor gt_aux_bev. So the dataset is
# "directly usable" from Stage B's core (occ + pkl) alone. gt_aux_bev (--aux) is forward-compat only
# and uses the mmdet3d bridge convention, which disagrees with SAFE's raster for ROTATED boxes
# (verified: IoU 1.0 for axis-aligned/ego boxes, lower for turned vehicles), so it is OFF by default.
# --cross-check is a diagnostic (prints vehicle-channel IoU vs the bridge); SAFE's raster is authoritative.
#
# Per-object height: Stage A tags every moving object with a roof height (uniscene_export.height_for_model,
# a semantic class prior) and writes a per-pixel `gt_veh_height` map into each token npz; Stage B's
# extrude_occ then gives each object its OWN z extent (round(h / --z-res) bins) instead of a constant
# veh_z. Existing occ .npy are skipped unless --overwrite-occ, so we pass it whenever heights change.
case "${1:-pilot}" in
  pilot)
    stage_a --datasets CIREN2017 legacy --pilot 15 --jobs "$JOBS" --resume
    stage_b --overwrite-occ        # regenerate occ with per-object heights (gt_veh_height)
    ;;
  full)
    stage_a --datasets CIREN2017 --jobs "$JOBS" --resume
    stage_a --datasets legacy     --jobs "$JOBS" --resume
    stage_b --overwrite-occ        # per-object-height occ
    ;;
  index)            # host: occ + pkl only
    shift || true
    stage_b "$@"
    ;;
  index-docker)     # uniscene-occ image: occ + pkl + optional --aux/--cross-check
    shift || true
    stage_b_docker "$@"
    ;;
  *)
    echo "usage: bash run_ciren_dataset.sh [pilot|full|index|index-docker]"; exit 1;;
esac
echo "[run] done. dataset -> $OUT_ROOT  (occ -> $GTS_DIR)"
