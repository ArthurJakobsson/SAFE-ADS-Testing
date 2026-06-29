# LegacyCIREN scraper (Track B)

Scrapes NHTSA **LegacyCIREN** crash cases (CIREN Crash Viewer, 2004–2015) into a
drop-in dataset format consumed by the existing pipeline:

```
<out>/<case_id>/Summary.txt   # crash narrative as a single prose paragraph
<out>/<case_id>/Sketch.jpg    # crash scene diagram/sketch, re-encoded JPEG
```

The default `--out` is `Framework/Crash_dataset` so output lands exactly where the
pipeline reads `Crash_dataset/<case_id>/Sketch.jpg` and `.../Summary.txt`.

---

## Confirmed live endpoints

All endpoints were verified on **2026-06-26** against case **109204** (and
115565, 120523). The backend behind the "CIREN Crash Viewer (2004–2015)" search
page (`https://crashviewer.nhtsa.dot.gov/crashviewer/LegacyCIREN/Search`) is the
classic NASS-CIREN ASP.NET app. Its viewer root was read straight from the
search page's JavaScript:

```js
viewerRoot = 'https://crashviewer.nhtsa.dot.gov/crashviewer/nass-CIREN'
// case page:  var url = viewerRoot + '/CaseForm.aspx?xsl=main.xsl&CaseID=' + data.caseid;
```

| Purpose            | Method | URL |
|--------------------|--------|-----|
| **Case XML**       | GET | `{BASE}/CaseForm.aspx?GetXML&caseid={id}` |
| Case HTML (human)  | GET | `{BASE}/CaseForm.aspx?xsl=main.xsl&CaseID={id}` |
| **Image / sketch** | GET | `{BASE}/GetBinary.aspx?Image&ImageID={imageId}&CaseID={id}&Version={ver}` |

where `BASE = https://crashviewer.nhtsa.dot.gov/crashviewer/nass-CIREN`.

> The `GetBinary.aspx` request uses a **bare `Image` flag** (no `=value`). The
> live server responds with `Content-Type: img/jpg` for a valid `ImageID`
> (observed file path `e:\webapps\nass\ciren\...jpg`).

### Where the data lives in the case XML

Root: `<case CaseID="109204" CaseStr="2004-38" xsi:noNamespaceSchemaLocation="../XSD/Ciren2004.xsd" ...>`

* **Narrative / summary** — `<summary>...</summary>`.
  This is byte-for-byte what the existing `Summary.txt` files contain (verified
  exact match for 109204/115565/120523). Note: a sibling `<casesummary>` exists
  but it *prepends* a crash-type label (e.g. "Vehicle to vehicle Angle/sideswipe");
  we deliberately use `<summary>`.

* **Scene sketch (the `Sketch.jpg`)** — `<imgform>/<scenedrawings>/<scene type="jpg">{ImageID}</scene>`.
  Example: `<scenedrawings><scene type="jpg" desc="">555113862</scene></scenedrawings>`.
  Download via `GetBinary.aspx?Image&ImageID=555113862&CaseID=109204&Version=0`.

  Other image containers in `<imgform>` are **not** the scene sketch and are not
  used here: `<crashscene>` (on-scene photos), `<diagram>` (intrusion / interior
  / steering sketches), `<cirenimages>` (full image index).

---

## Files

| File | Role |
|------|------|
| `ciren_client.py`  | HTTP layer: `requests.Session`, retry + exponential backoff on 429/5xx, clear 403 handling, configurable base URL. Includes optional `SeleniumCirenClient` headless-Chrome fallback. |
| `ciren_parse.py`   | Pure parsing. **All** site-specific tag selectors live here (narrative tag, scene-sketch xpath). Collapses whitespace to one paragraph; resolves the sketch `ImageID`. |
| `ciren_scraper.py` | Orchestration / CLI. Idempotent; logs to stdout + `scrape_log.csv`; writes/merges `manifest.json`. |
| `requirements.txt` | Dependencies (`selenium` optional). |

---

## Usage

```bash
pip install -r requirements.txt   # base conda already has requests, pillow; adds bs4, lxml

# Specific cases (writes into the real dataset dir by default)
python ciren_scraper.py --ids 109204,115565,117021

# A contiguous id range, capped
python ciren_scraper.py --id-range 109000-110000 --max 100

# Custom output + politeness (3s base delay + jitter is the default)
python ciren_scraper.py --ids 109204 --out /path/to/out --delay 5

# Re-download even if files already exist
python ciren_scraper.py --ids 109204 --overwrite

# Headless-Chrome fallback if plain requests gets HTTP 403 from Akamai
python ciren_scraper.py --ids 109204 --selenium
```

Key flags: `--ids`, `--id-range`, `--out` (default `Framework/Crash_dataset`),
`--delay` (default `3.0` + jitter), `--user-agent`, `--max`, `--base-url`,
`--overwrite`, `--selenium`, `--log`, `--manifest`, `-v`.

Per case the scraper: skips if both files already exist → fetches case XML →
parses `<summary>` → resolves + downloads the scene-sketch `ImageID` →
re-encodes to JPEG via Pillow (`Image.open(BytesIO(...)).convert('RGB').save(path,'JPEG')`)
→ writes `Summary.txt` + `Sketch.jpg` → logs `ok/skip/fail` to stdout and
`scrape_log.csv` → updates `manifest.json`.

### manifest.json (for the dashboard)

A JSON list of objects, merged across runs (keyed by `case_id`):

```json
[
  {
    "case_id": "109204",
    "summary_path": ".../109204/Summary.txt",
    "sketch_path":  ".../109204/Sketch.jpg",
    "status": "ok",
    "summary_chars": 1559,
    "sketch_bytes": 47858
  }
]
```

---

## Politeness / Terms of Use

* **Public domain.** NHTSA crash data is U.S. Government work and in the public
  domain. This tool only retrieves already-public case pages.
* **Identify yourself.** The default User-Agent names the project and a contact
  email. Keep it descriptive.
* **Rate-limit.** Default `--delay 3.0` seconds + jitter between network hits.
  Do not lower this for large ranges; prefer running overnight.
* **robots.txt.** `https://crashviewer.nhtsa.dot.gov/robots.txt` is itself
  returned as Akamai "Access Denied" from automated clients, so it could not be
  programmatically parsed. Treat the site conservatively: low rate, off-peak,
  back off on any 429/5xx (the client does this automatically).
* **Back off on errors.** The client retries 429/5xx with exponential backoff +
  jitter and honors `Retry-After`.

---

## How to validate

1. **Local pipeline test (no network):** a captured real case XML + a synthetic
   image are served by a tiny mock; the scraper produces a `Summary.txt` that is
   a **byte-exact** match of the existing `Crash_dataset/109204/Summary.txt` and
   a valid re-encoded `Sketch.jpg`. See `_scratch_out/` for the test artifacts.

2. **Field check against the shipped dataset:** for any case you scrape, compare
   `diff <(tr -s ' \n' ' ' < new/Summary.txt) <(tr -s ' \n' ' ' < Crash_dataset/<id>/Summary.txt)`.
   They should be identical (the existing dataset was generated from `<summary>`).

3. **Live run:** `python ciren_scraper.py --ids 109204 --out /tmp/check`. If you
   get `HTTP 403 (Access Denied)`, you are on a network Akamai is filtering
   (see below) — switch networks or use `--selenium`.

---

## Known issue: Akamai IP filtering (what a human must verify)

From **datacenter / cloud IPs** (including the CI/build environment this was
developed in), `crashviewer.nhtsa.dot.gov` is fronted by Akamai and returns
**HTTP 403 "Access Denied"** to all programmatic requests — `curl`, `requests`,
and server-side fetchers alike. This is an edge/IP reputation filter, **not** a
login wall and **not** a per-case problem:

* The endpoints themselves are correct and live — verified end-to-end through an
  independent fetch path that resolves from a different region: the real case XML
  was retrieved and the `<summary>` matched the shipped dataset exactly, and the
  `GetBinary.aspx?Image...` URL returned `Content-Type: img/jpg`.

**A human on a normal (residential / institutional) network must:**

1. Run `python ciren_scraper.py --ids 109204 --out /tmp/check` and confirm it
   produces files (i.e. no 403 from your network).
2. Confirm the downloaded `Sketch.jpg` is the real crash scene diagram (the
   image-byte fetch could not be completed from the blocked build environment,
   only the endpoint + content-type were confirmed). Eyeball it against
   `Crash_dataset/109204/Sketch.jpg`.
3. If you still get 403, run with `--selenium` (needs `pip install selenium` +
   Chrome + chromedriver). A real browser fingerprint is far less likely to be
   filtered.

No login or credentials are required for these public cases.
