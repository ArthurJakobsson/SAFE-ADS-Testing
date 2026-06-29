#!/usr/bin/env python3
"""
ciren_modern.py -- scrape the MODERN CIREN crash viewer (2017+ relaunch, ~359 cases).

Unlike the legacy nass-CIREN viewer (6-digit CaseIDs, JPG sketches, handled by
ciren_scraper.py), the modern /ciren/ SPA is backed by a JSON API and stores scene
diagrams as PDFs. This module produces the SAME drop-in dataset format:

    <out>/<case>/Summary.txt   -- crash narrative (from cirenCrashSummary.summary, de-HTML'd)
    <out>/<case>/Sketch.jpg    -- scene diagram (scene PDF page 1, rendered to JPEG)

API flow (all over curl_cffi Firefox impersonation -- the host JA3-blocks plain requests):
    list   : POST {API}/ciren/cases/search           body {"filters":[]}  -> [{cirenId,...}]
    detail : GET  {API}/Ciren/GetCirenCrashDetails?cirenId=N  -> narrative + caseId + caseNumber
    scene  : POST {API}/ciren/GetSceneDiagram?caseID={caseId} -> [{objectid, filename(.pdf)}]
             GET  {API}/ciren/scenefiles/download/{caseId}?objectId={objid} -> PDF bytes
where API = https://crashviewer.nhtsa.dot.gov/api

Idempotent, rate-limited, writes a dashboard-compatible manifest.json.
"""
from __future__ import annotations

import argparse
import html as _html
import io
import json
import os
import random
import re
import sys
import time

from curl_cffi import requests as creq

HOST = "https://crashviewer.nhtsa.dot.gov"
API = HOST + "/api"
JH = {"Content-Type": "application/json", "Accept": "application/json"}
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(TOOLS_DIR), "Crash_dataset_CIREN2017")


def make_session():
    return creq.Session(impersonate="firefox")


def list_cases(s):
    r = s.post(f"{API}/ciren/cases/search", headers=JH, data=json.dumps({"filters": []}), timeout=120)
    r.raise_for_status()
    return r.json()  # list of {cirenId, make, model, modelYear, sex}


def html_to_text(html: str) -> str:
    """Convert the summary HTML to readable text: <br><br> -> blank line, <br> -> space,
    strip remaining tags, unescape entities, collapse intra-paragraph whitespace."""
    if not html:
        return ""
    t = re.sub(r"(?i)<br\s*/?>\s*<br\s*/?>", "\n\n", html)
    t = re.sub(r"(?i)<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = _html.unescape(t)
    paras = [re.sub(r"\s+", " ", p).strip() for p in t.split("\n\n")]
    return "\n\n".join(p for p in paras if p)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "case"


def render_pdf_first_page_to_jpeg(pdf_bytes: bytes, dpi: int = 130) -> bytes:
    import fitz  # PyMuPDF
    from PIL import Image
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def fetch_case(s, ciren_id):
    """Return dict(caseid, caseno, narrative, jpg|None, note)."""
    d = s.get(f"{API}/Ciren/GetCirenCrashDetails?cirenId={ciren_id}", timeout=90).json()
    cs = d.get("cirenCrashSummary") or {}
    summ = d.get("cirenSummary") or {}
    narrative = html_to_text(cs.get("summary") or summ.get("crashSummary") or "")
    caseid = summ.get("caseId") or cs.get("caseId")
    caseno = cs.get("caseNumber") or summ.get("caseNumber") or str(ciren_id)

    jpg, note = None, ""
    if caseid is not None:
        sd = s.post(f"{API}/ciren/GetSceneDiagram?caseID={caseid}", headers=JH, timeout=90).json()
        objid = sd[0].get("objectid") if sd else None
        if objid:
            pdf = s.get(f"{API}/ciren/scenefiles/download/{caseid}?objectId={objid}", timeout=150)
            if pdf.status_code == 200 and pdf.content[:4] == b"%PDF":
                try:
                    jpg = render_pdf_first_page_to_jpeg(pdf.content)
                except Exception as exc:  # noqa: BLE001
                    note = f"pdf render failed: {exc}"
            else:
                note = f"scene not a PDF (HTTP {pdf.status_code})"
        else:
            note = "no scene diagram object"
    else:
        note = "no caseId"
    return dict(caseid=caseid, caseno=caseno, narrative=narrative, jpg=jpg, note=note)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scrape modern CIREN (2017+) into <case>/Summary.txt + Sketch.jpg")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--delay", type=float, default=1.5, help="base delay between cases (s); jitter added")
    ap.add_argument("--max", type=int, default=None, help="cap number of cases")
    ap.add_argument("--ids", help="comma-separated cirenIds to scrape (default: all from search)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    manifest_path = args.manifest or os.path.join(args.out, "manifest.json")
    s = make_session()

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        print("[ciren-modern] listing cases ...", flush=True)
        ids = [c["cirenId"] for c in list_cases(s)]
        print(f"[ciren-modern] {len(ids)} cases", flush=True)
    if args.max:
        ids = ids[: args.max]

    results = []
    n = len(ids)
    for i, cid in enumerate(ids, 1):
        case_dir = None
        try:
            # peek folder name only after we know caseno; first check by cirenId marker file
            info = fetch_case(s, cid)
            folder = sanitize(info["caseno"])
            case_dir = os.path.join(args.out, folder)
            summary_path = os.path.join(case_dir, "Summary.txt")
            sketch_path = os.path.join(case_dir, "Sketch.jpg")

            if not args.overwrite and os.path.exists(summary_path) and os.path.exists(sketch_path):
                status, note = "skip", "exists"
            elif not info["narrative"]:
                status, note = "fail", "no narrative"
            elif info["jpg"] is None:
                status, note = "fail", info["note"] or "no sketch"
            else:
                os.makedirs(case_dir, exist_ok=True)
                with open(summary_path, "w", encoding="utf-8") as fh:
                    fh.write(info["narrative"])
                with open(sketch_path, "wb") as fh:
                    fh.write(info["jpg"])
                status, note = "ok", info["note"]
            results.append({
                "case_id": folder, "cirenId": cid, "caseid": info["caseid"],
                "status": status, "note": note,
                "summary_path": os.path.join(case_dir, "Summary.txt") if case_dir else "",
                "sketch_path": os.path.join(case_dir, "Sketch.jpg") if case_dir else "",
                "summary_chars": len(info["narrative"]),
                "sketch_bytes": len(info["jpg"]) if info["jpg"] else 0,
            })
            print(f"[{i}/{n}] cirenId={cid} {folder} -> {status}" + (f" ({note})" if note else ""), flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": str(cid), "cirenId": cid, "status": "fail", "note": f"error: {exc}"})
            print(f"[{i}/{n}] cirenId={cid} -> fail ({exc})", flush=True)

        if i < n:
            time.sleep(max(0.0, args.delay) + random.uniform(0, args.delay * 0.4))

    # merge manifest
    existing = {}
    if os.path.exists(manifest_path):
        try:
            for e in json.load(open(manifest_path)):
                existing[str(e.get("case_id"))] = e
        except Exception:
            pass
    for r in results:
        existing[str(r["case_id"])] = r
    json.dump(sorted(existing.values(), key=lambda e: str(e["case_id"])),
              open(manifest_path, "w"), indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    sk = sum(1 for r in results if r["status"] == "skip")
    fa = sum(1 for r in results if r["status"] == "fail")
    print(f"[ciren-modern] done: {ok} ok, {sk} skip, {fa} fail -> {args.out}", flush=True)
    return 0 if fa == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
