#!/usr/bin/env python3
"""
ciren_scraper.py -- orchestration for the LegacyCIREN (2005-2016) scraper (Track B).

For each case id it produces, drop-in compatible with the existing pipeline:

    <out>/<case_id>/Summary.txt   -- crash narrative, single prose paragraph
    <out>/<case_id>/Sketch.jpg    -- crash scene diagram/sketch, re-encoded JPEG

It is idempotent (skips a case if both files already exist), polite
(rate-limited with jitter, descriptive User-Agent), and logs every
success/skip/failure to stdout and to scrape_log.csv. After a run it writes a
machine-readable manifest.json the dashboard can consume.

Usage examples
--------------
    # Specific ids
    python ciren_scraper.py --ids 109204,115565,117021

    # A range (inclusive), capped at 50 attempts
    python ciren_scraper.py --id-range 109000-110000 --max 50

    # Custom output dir + politeness
    python ciren_scraper.py --ids 109204 --out /tmp/out --delay 5

    # Force re-download even if files exist
    python ciren_scraper.py --ids 109204 --overwrite

    # Use the Selenium fallback (if Akamai is 403-ing plain requests)
    python ciren_scraper.py --ids 109204 --selenium

Verified endpoints (2026-06-26): see README.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import ciren_parse as cp
from ciren_client import (
    DEFAULT_BASE_URL,
    DEFAULT_USER_AGENT,
    CirenClient,
    CirenClientError,
)

LOG = logging.getLogger("ciren.scraper")

DEFAULT_OUT = "/home/adteam/Documents/SAFE-ADS-Testing/Framework/Crash_dataset"

# Manifest + log live next to the scraper (in _scratch_out by default for the
# scratch validation, but configurable). The dashboard reads manifest.json.
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRATCH = os.path.join(TOOLS_DIR, "_scratch_out")


@dataclass
class CaseResult:
    case_id: str
    status: str            # "ok" | "skip" | "fail"
    note: str = ""
    summary_path: str = ""
    sketch_path: str = ""
    summary_chars: int = 0
    sketch_bytes: int = 0


# ---------------------------------------------------------------------------
# Core per-case logic
# ---------------------------------------------------------------------------

def scrape_one(client, case_id: str, out_dir: str, overwrite: bool) -> CaseResult:
    case_dir = os.path.join(out_dir, str(case_id))
    summary_path = os.path.join(case_dir, "Summary.txt")
    sketch_path = os.path.join(case_dir, "Sketch.jpg")

    # Idempotency: skip if both already exist.
    if not overwrite and os.path.exists(summary_path) and os.path.exists(sketch_path):
        return CaseResult(
            case_id=str(case_id),
            status="skip",
            note="both files already exist",
            summary_path=summary_path,
            sketch_path=sketch_path,
            summary_chars=_safe_size_chars(summary_path),
            sketch_bytes=_safe_size_bytes(sketch_path),
        )

    # 1) Fetch + parse the case XML.
    try:
        xml_text = client.fetch_case_xml(case_id)
    except CirenClientError as exc:
        return CaseResult(case_id=str(case_id), status="fail", note=f"fetch xml: {exc}")

    # Guard: a non-existent case returns the SPA shell / error page, not <case>.
    if "<case" not in xml_text.lower():
        return CaseResult(
            case_id=str(case_id),
            status="fail",
            note="no <case> element in response (case may not exist or site returned an error page)",
        )

    narrative = cp.extract_narrative(xml_text)
    sketch = cp.extract_scene_sketch(xml_text)

    if not narrative:
        return CaseResult(case_id=str(case_id), status="fail", note="no <summary> narrative found")
    if not sketch:
        return CaseResult(
            case_id=str(case_id),
            status="fail",
            note="no <scenedrawings><scene> sketch found",
            summary_chars=len(narrative),
        )

    # 2) Download the sketch image bytes.
    try:
        img_bytes = client.download_image(sketch.image_id, case_id, sketch.version)
    except CirenClientError as exc:
        return CaseResult(
            case_id=str(case_id),
            status="fail",
            note=f"download sketch (ImageID={sketch.image_id}): {exc}",
            summary_chars=len(narrative),
        )

    # 3) Re-encode to JPEG via Pillow (normalize to RGB JPEG, matches dataset).
    try:
        jpeg_bytes = _to_jpeg(img_bytes)
    except Exception as exc:  # noqa: BLE001  -- surface any decode error as a failure
        return CaseResult(
            case_id=str(case_id),
            status="fail",
            note=f"re-encode JPEG failed (got {len(img_bytes)} bytes): {exc}",
            summary_chars=len(narrative),
        )

    # 4) Write outputs atomically-ish.
    os.makedirs(case_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(narrative)
    with open(sketch_path, "wb") as fh:
        fh.write(jpeg_bytes)

    return CaseResult(
        case_id=str(case_id),
        status="ok",
        note=f"ImageID={sketch.image_id}",
        summary_path=summary_path,
        sketch_path=sketch_path,
        summary_chars=len(narrative),
        sketch_bytes=len(jpeg_bytes),
    )


def _to_jpeg(raw: bytes) -> bytes:
    """Image.open(BytesIO(...)).convert('RGB').save(buf,'JPEG') -> bytes."""
    from PIL import Image  # lazy import so --help works without Pillow

    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, "JPEG", quality=92)
    return out.getvalue()


def _safe_size_chars(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as fh:
            return len(fh.read())
    except OSError:
        return 0


def _safe_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Run loop, logging, manifest
# ---------------------------------------------------------------------------

def run(ids: List[str], args) -> List[CaseResult]:
    if args.selenium:
        from ciren_client import SeleniumCirenClient

        client = SeleniumCirenClient(base_url=args.base_url, user_agent=args.user_agent)
    else:
        client = CirenClient(base_url=args.base_url, user_agent=args.user_agent)

    results: List[CaseResult] = []
    log_rows: List[dict] = []

    n = len(ids)
    for i, cid in enumerate(ids, 1):
        try:
            res = scrape_one(client, cid, args.out, args.overwrite)
        except KeyboardInterrupt:
            LOG.warning("interrupted by user")
            break
        except Exception as exc:  # noqa: BLE001 -- never let one case kill the run
            res = CaseResult(case_id=str(cid), status="fail", note=f"unexpected: {exc}")

        results.append(res)
        log_rows.append({"case_id": res.case_id, "status": res.status, "note": res.note})
        LOG.info(
            "[%d/%d] %s -> %s%s",
            i, n, res.case_id, res.status,
            f" ({res.note})" if res.note else "",
        )

        # Politeness: delay + jitter between *network* hits only.
        if res.status != "skip" and i < n:
            delay = max(0.0, args.delay) + random.uniform(0, args.delay * 0.3)
            time.sleep(delay)

    if getattr(client, "close", None):
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    write_log_csv(args.log, log_rows)
    write_manifest(args.manifest, results)
    return results


def write_log_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case_id", "status", "note"])
        if new:
            writer.writeheader()
        writer.writerows(rows)
    LOG.info("appended %d rows to %s", len(rows), path)


def write_manifest(path: str, results: List[CaseResult]) -> None:
    """Write/merge manifest.json: list of per-case dicts for the dashboard.

    Merges with any existing manifest (keyed by case_id) so successive runs
    accumulate rather than clobber.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    existing: dict[str, dict] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                for entry in json.load(fh):
                    existing[str(entry.get("case_id"))] = entry
        except (json.JSONDecodeError, OSError):
            existing = {}

    for r in results:
        existing[r.case_id] = {
            "case_id": r.case_id,
            "summary_path": r.summary_path,
            "sketch_path": r.sketch_path,
            "status": r.status,
            "summary_chars": r.summary_chars,
            "sketch_bytes": r.sketch_bytes,
        }

    merged = sorted(existing.values(), key=lambda e: str(e["case_id"]))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    LOG.info("wrote manifest with %d entries to %s", len(merged), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_ids(args) -> List[str]:
    ids: List[str] = []
    if args.ids:
        ids.extend(p.strip() for p in args.ids.split(",") if p.strip())
    if args.id_range:
        try:
            lo, hi = args.id_range.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
        except ValueError:
            raise SystemExit(f"--id-range must be LOW-HIGH integers, got {args.id_range!r}")
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        ids.extend(str(x) for x in range(lo_i, hi_i + 1))
    # De-dup, preserve order.
    seen = set()
    uniq = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    if args.max is not None:
        uniq = uniq[: args.max]
    return uniq


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scrape NHTSA LegacyCIREN cases into <case_id>/Summary.txt + Sketch.jpg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ids", help="comma-separated case ids, e.g. 109204,115565")
    p.add_argument("--id-range", help="inclusive id range, e.g. 109000-110000")
    p.add_argument("--out", default=DEFAULT_OUT, help="output dataset directory")
    p.add_argument("--delay", type=float, default=3.0, help="base delay (s) between network hits; jitter added")
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LegacyCIREN backend base URL")
    p.add_argument("--max", type=int, default=None, help="max number of cases to attempt")
    p.add_argument("--overwrite", action="store_true", help="re-fetch even if both files exist")
    p.add_argument("--selenium", action="store_true", help="use headless-Chrome fallback transport")
    p.add_argument("--log", default=os.path.join(DEFAULT_SCRATCH, "scrape_log.csv"),
                   help="CSV log path (case_id,status,note)")
    p.add_argument("--manifest", default=os.path.join(DEFAULT_SCRATCH, "manifest.json"),
                   help="manifest.json path for the dashboard")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ids = parse_ids(args)
    if not ids:
        LOG.error("no case ids given. Use --ids or --id-range.")
        return 2

    LOG.info("scraping %d case(s) -> %s", len(ids), args.out)
    results = run(ids, args)

    ok = sum(1 for r in results if r.status == "ok")
    skip = sum(1 for r in results if r.status == "skip")
    fail = sum(1 for r in results if r.status == "fail")
    LOG.info("done: %d ok, %d skipped, %d failed", ok, skip, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
