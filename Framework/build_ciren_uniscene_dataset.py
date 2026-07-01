#!/usr/bin/env python3
"""Batch driver: CIREN crash cases  ->  SAFE (meta/DSL/conflict)  ->  per-vehicle UniScene BEV tokens.

Runs in the SAFE/vLLM env with CWD = Framework/ (the SAFE stage functions hardcode relative paths
like ./Crash_dataset/{id}/ and ./Knowledge_base/...).  For EACH "ok" case in the CIREN manifests it:

  1. Stage 1  meta  = Meta_Message_Extraction.meta_msg_extraction(...)        [road_type, car_num, dir]
              + message_validation (retry once on "not consistent")
  2. Stage 2  prompts = Prompts_Generation.generate_<roadtype>(...)            -> {prompts}/{id}/*.txt
  3. Stage 3  dsl   = Scenario_Representation_Extraction.get_dsl(...)          (the SAFE DSL)
              + dsl_validation (retry once)
  3b.Conflict dsl["Conflict"] = conflict_augment.extract_conflict(...)        (deterministic at-fault)
  4. Export  uniscene_export.export_case(...)  -> per-vehicle DENSE BEV npz token clips + human GIFs
  5. Provenance: source symlinks + safe/{meta,dsl}.json + raw LLM txt + provenance.json

Idempotent/resumable (--resume skips cases whose provenance + all token npz already exist; never
re-rolls a finished case's stochastic DSL).  Per-case failures are logged and skipped, never abort
the batch.  The global index pkl + extruded occ are built afterwards by build_ciren_uniscene_index.py.

Examples (from Framework/, after `source ../serving/env.sh`):
  python build_ciren_uniscene_dataset.py --datasets CIREN2017 legacy \
      --out-root /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2 --pilot 15 --jobs 2
  python build_ciren_uniscene_dataset.py --datasets legacy \
      --out-root /mnt/disk2/CIREN_dataset/CIREN_uniscene_v2 --jobs 3 --resume
  python build_ciren_uniscene_dataset.py --datasets CIREN2017 --out-root /tmp/ds --pilot 3 --dry-run
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# --- locate repo dirs and make SAFE's relative paths resolve -------------------------------------
FRAMEWORK = os.path.dirname(os.path.abspath(__file__))
BEV_SYNTH = os.path.join(FRAMEWORK, "ADS_Testing", "BEV_Synthesis")
sys.path.insert(0, FRAMEWORK)
sys.path.insert(0, BEV_SYNTH)
os.chdir(FRAMEWORK)

import Meta_Message_Extraction as MME          # noqa: E402
import Prompts_Generation as PG                # noqa: E402
import Scenario_Representation_Extraction as SRE  # noqa: E402
import conflict_augment as CA                  # noqa: E402
import uniscene_export as UE                   # noqa: E402

SOURCE_ROOT = "/mnt/disk2/CIREN_dataset"
DATASET_DIRS = {"CIREN2017": "Crash_dataset_CIREN2017", "legacy": "Crash_dataset_CIREN_legacy"}
DATASET_TAG = {"CIREN2017": "c17", "legacy": "lg"}
VALID_ROADS = ["Straight", "Curve", "Intersection", "T-intersection", "Merging"]
STAGE_DIR = os.path.join(FRAMEWORK, "Crash_dataset")   # where get_dsl looks: ./Crash_dataset/{id}/
_GEN = {"Straight": PG.generate_straight, "Curve": PG.generate_curve,
        "Intersection": PG.generate_intersection, "T-intersection": PG.generate_t_intersection,
        "Merging": PG.generate_merging}
_csv_lock = threading.Lock()
_stage_lock = threading.Lock()


# --- helpers -------------------------------------------------------------------------------------
def load_meta_kb():
    """Few-shot Meta-stage assets, read once (constant across cases)."""
    return (MME.encode_image("./Knowledge_base/Meta_prompts/example_sketch.jpg"),
            open("./Knowledge_base/Meta_prompts/example_summary.txt", encoding="utf-8").read(),
            open("./Knowledge_base/Meta_prompts/p1.txt", encoding="utf-8").read(),
            open("./Knowledge_base/Meta_prompts/p2.txt", encoding="utf-8").read(),
            open("./Knowledge_base/Meta_prompts/system.txt", encoding="utf-8").read())


def norm_road_type(rt):
    s = str(rt or "").strip().lower()
    if "t-int" in s or "t int" in s or "t-junction" in s or s.startswith("t-"):
        return "T-intersection"
    if "inter" in s or "junction" in s or "crossroad" in s:
        return "Intersection"
    if "curv" in s or "bend" in s:
        return "Curve"
    if "merg" in s or "ramp" in s:
        return "Merging"
    for r in VALID_ROADS:           # exact canonical match
        if s == r.lower():
            return r
    return "Straight"               # safe default (also avoids get_dsl UnboundLocalError)


def norm_direction(d):
    return "same direction" if "same" in str(d or "").lower() else "opposite direction"


def make_case_token(tag, case_id):
    return f"{tag}_{re.sub(r'[^0-9A-Za-z]', '', str(case_id))}"


def manifest_ok_cases(dataset_name):
    p = os.path.join(SOURCE_ROOT, DATASET_DIRS[dataset_name], "manifest.json")
    data = json.load(open(p))
    entries = data if isinstance(data, list) else next(
        (v for v in data.values() if isinstance(v, list)), [])
    return [str(e["case_id"]) for e in entries if e.get("status") == "ok"]


def stage_case(case_id, real_case_dir):
    """Symlink Framework/Crash_dataset/{case_id} -> the real CIREN case dir so get_dsl resolves."""
    with _stage_lock:
        os.makedirs(STAGE_DIR, exist_ok=True)
        link = os.path.join(STAGE_DIR, case_id)
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            return link                        # a real dir with this id already exists (demo) -> leave it
        os.symlink(real_case_dir, link)
    return link


def unstage_case(case_id):
    link = os.path.join(STAGE_DIR, case_id)
    if os.path.islink(link):
        try:
            os.remove(link)
        except OSError:
            pass


def stub_meta_dsl(case_id):
    """Deterministic offline stand-in for the LLM stages (--dry-run): a 2-car head-on."""
    meta = ["Straight", 2, "same direction"]
    dsl = {
        "Actors": {
            "Vehicle_1": {"Model": "Sedan", "Initial_position": "W2E",
                          "Actions": "Move forward", "Speed": "30 mph"},
            "Vehicle_2": {"Model": "SUV", "Initial_position": "E2W",
                          "Actions": "cross the centerline", "Speed": "35 mph"}},
        "Road network": {"Road type": "Straight", "Number of lanes": 2},
        "Environment": {"Weather": "Clear"},
        "Scenario": case_id,
        "Conflict": {"at_fault_vehicle": "Vehicle_2", "struck_vehicle": "Vehicle_1",
                     "impact_type": "head-on", "point_of_impact": "front",
                     "description": "dry-run stub"}}
    return meta, dsl


# --- SAFE stage wrappers (validation + one retry, mirroring the originals) ------------------------
def run_meta(case_id, sketch_b64, summary, kb, model, raw_dir):
    ex_sketch, ex_summary, p1, p2, system_info = kb
    meta = MME.meta_msg_extraction(sketch_b64, summary, ex_sketch, ex_summary,
                                   p1, p2, case_id, raw_dir, model)
    try:
        ok = MME.message_validation(system_info, meta, sketch_b64, summary, case_id, raw_dir, model)
    except Exception:
        ok = 1
    if ok == 0:
        meta = MME.meta_msg_extraction(sketch_b64, summary, ex_sketch, ex_summary,
                                       p1, p2, case_id, raw_dir, model)
    return meta


def run_dsl(case_id, prompts_root, road_type, direction, model, raw_dir):
    raw, dsl = SRE.get_dsl(case_id, prompts_root, road_type, direction, model, raw_dir)
    try:
        ok = SRE.dsl_validation(raw, case_id, model, raw_dir)
    except Exception:
        ok = 1
    if ok == 0:
        raw, dsl = SRE.get_dsl(case_id, prompts_root, road_type, direction, model, raw_dir)
    dsl["Scenario"] = case_id
    return dsl


# --- outputs / provenance ------------------------------------------------------------------------
def _symlink(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)


def is_complete(prov_path, bev_dir):
    if not os.path.exists(prov_path):
        return False
    try:
        prov = json.load(open(prov_path))
        return all(os.path.exists(os.path.join(bev_dir, os.path.basename(p)))
                   for p in prov.get("bev_npz", {}).values())
    except Exception:
        return False


def write_outputs(sample_dir, safe_dir, dataset_name, case_id, case_token,
                  sketch_path, summary_path, meta, road_type, direction, dsl, exp, args):
    _symlink(sketch_path, os.path.join(sample_dir, "source", "Sketch.jpg"))
    _symlink(summary_path, os.path.join(sample_dir, "source", "Summary.txt"))
    json.dump({"road_type_raw": meta[0], "car_num": meta[1], "drive_dir_raw": meta[2],
               "road_type": road_type, "direction": direction},
              open(os.path.join(safe_dir, "meta.json"), "w"), indent=2)
    json.dump(dsl, open(os.path.join(safe_dir, "dsl.json"), "w"), indent=2, default=str)

    tokens = {cv["label"]: cv["tokens"] for cv in exp["car_views"]}
    bev_npz = {tok: f"bev/{tok}.npz" for cv in exp["car_views"] for tok in cv["tokens"]}
    ego_views = {cv["label"]: {"gif": f"ego/{case_id}_ego_{cv['label']}.gif",
                               "png": f"ego/{case_id}_ego_{cv['label']}.png"}
                 for cv in exp["car_views"]}
    prov = {
        "source_dataset": DATASET_DIRS[dataset_name],
        "case_id": case_id, "case_token": case_token,
        "source_sketch": os.path.abspath(sketch_path),
        "source_summary": os.path.abspath(summary_path),
        "safe_model": args.model,
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": bool(args.dry_run),
        "road_type": road_type, "road_type_raw": meta[0],
        "car_num": meta[1], "drive_dir": meta[2], "direction": direction,
        "n_vehicles": exp["n_vehicles"], "single_vehicle": exp["single_vehicle"],
        "impact_frame": exp["impact_frame"], "frames": exp["frames"],
        "fps": exp["fps"], "frame_dt": exp["dt"],
        "conflict": dsl.get("Conflict", {}),
        "vehicles": [{"label": cv["label"], "ego_idx": cv["ego_idx"]} for cv in exp["car_views"]],
        "tokens": tokens, "bev_npz": bev_npz,
        "human_views": {"scene_gif": f"scene/{case_id}_uniscene.gif",
                        "scene_png": f"scene/{case_id}_uniscene.png", "ego": ego_views},
    }
    json.dump(prov, open(os.path.join(sample_dir, "provenance.json"), "w"), indent=2)


# --- per-case orchestration ----------------------------------------------------------------------
def process_case(dataset_name, case_id, args, kb):
    ds_dir = os.path.join(SOURCE_ROOT, DATASET_DIRS[dataset_name])
    src_dir = os.path.join(ds_dir, case_id)
    sketch_path = os.path.join(src_dir, "Sketch.jpg")
    summary_path = os.path.join(src_dir, "Summary.txt")
    sample_dir = os.path.join(args.out_root, "samples", DATASET_DIRS[dataset_name], case_id)
    safe_dir = os.path.join(sample_dir, "safe")
    bev_dir = os.path.join(args.out_root, "bev")
    prompts_root = os.path.join(args.out_root, "_work", "prompts")
    case_token = make_case_token(DATASET_TAG[dataset_name], case_id)
    row = {"dataset": dataset_name, "case_id": case_id, "status": "?", "n_tokens": 0, "note": ""}

    if args.resume and not args.overwrite and is_complete(
            os.path.join(sample_dir, "provenance.json"), bev_dir):
        row["status"] = "skip_done"
        return row
    if not (os.path.exists(sketch_path) and os.path.getsize(sketch_path) > 0
            and os.path.exists(summary_path) and os.path.getsize(summary_path) > 0):
        row["status"] = "skip_missing"
        return row

    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(prompts_root, exist_ok=True)
    try:
        if args.dry_run:
            meta, dsl = stub_meta_dsl(case_id)
            road_type, direction = "Straight", "same direction"
        else:
            sketch_b64 = MME.encode_image(sketch_path)
            summary = open(summary_path, encoding="utf-8", errors="ignore").read()
            meta = run_meta(case_id, sketch_b64, summary, kb, args.model, safe_dir)
            road_type = norm_road_type(meta[0])
            direction = norm_direction(meta[2])
            _GEN[road_type](meta[1], direction, case_id, prompts_root)
            stage_case(case_id, src_dir)
            try:
                dsl = run_dsl(case_id, prompts_root, road_type, direction, args.model, safe_dir)
                dsl["Conflict"] = CA.extract_conflict(case_id, ds_dir, args.model, safe_dir)
            finally:
                unstage_case(case_id)

        exp = UE.export_case(case_id, case_token, dsl, road_type,
                             bev_dir=bev_dir, sample_dir=sample_dir,
                             frames=args.frames, fps=args.fps, step=args.step,
                             human_fps=args.human_fps, write_human=not args.no_human)
        write_outputs(sample_dir, safe_dir, dataset_name, case_id, case_token,
                      sketch_path, summary_path, meta, road_type, direction, dsl, exp, args)
        row["status"] = "ok"
        row["n_tokens"] = exp["n_tokens"]
        row["note"] = f"{road_type} v{exp['n_vehicles']} impact={exp['impact_frame']}"
    except Exception as e:
        row["status"] = "fail"
        row["note"] = f"{type(e).__name__}: {e}"[:200]
        try:
            open(os.path.join(safe_dir, "ERROR.txt"), "w").write(traceback.format_exc())
        except Exception:
            pass
    return row


def append_csv(log_path, row):
    with _csv_lock:
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts", "dataset", "case_id", "status", "n_tokens", "note"])
            if new:
                w.writeheader()
            w.writerow({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **row})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["CIREN2017", "legacy"], choices=list(DATASET_DIRS))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pilot", type=int, default=0, help="process only the first N ok cases total")
    ap.add_argument("--limit", type=int, default=0, help="alias for --pilot (cap total cases)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--model", default=os.environ.get("SAFE_MODEL", "Qwen/Qwen3-VL-32B-Instruct"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--step", type=int, default=4, help="fine-sim subsample factor (collision capture)")
    ap.add_argument("--human-fps", type=int, default=8)
    ap.add_argument("--no-human", action="store_true", help="skip GIF/PNG (faster, smaller)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stub the LLM stages (offline plumbing test)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_root, "bev"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "index"), exist_ok=True)
    log_path = os.path.join(args.out_root, "index", "build_log.csv")

    work = [(ds, cid) for ds in args.datasets for cid in manifest_ok_cases(ds)]
    cap = args.pilot or args.limit
    if cap:
        work = work[:cap]
    kb = None if args.dry_run else load_meta_kb()

    print(f"[driver] {len(work)} case(s) over {args.datasets} | jobs={args.jobs} "
          f"| model={args.model} | dry_run={args.dry_run} | out={args.out_root}")
    counts = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(process_case, ds, cid, args, kb): (ds, cid) for ds, cid in work}
        for i, fut in enumerate(as_completed(futs), 1):
            ds, cid = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"dataset": ds, "case_id": cid, "status": "fail", "n_tokens": 0, "note": str(e)[:200]}
            append_csv(log_path, row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{i}/{len(work)}] {ds}/{cid}: {row['status']} "
                  f"({row['n_tokens']} tok) {row['note']}")
    print(f"[driver] done. status counts: {counts}  log -> {log_path}")


if __name__ == "__main__":
    main()
