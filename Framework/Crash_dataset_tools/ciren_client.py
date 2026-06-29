"""
ciren_client.py -- HTTP layer for the NHTSA LegacyCIREN crash viewer scraper (Track B).

This module owns *all* network I/O. It exposes a small, well-behaved client that
talks to the legacy NASS-CIREN ASP.NET backend that still powers the
"CIREN Crash Viewer (2004 - 2015)" search at:

    https://crashviewer.nhtsa.dot.gov/crashviewer/LegacyCIREN/Search

Confirmed live endpoints (verified 2026-06-26 against case 109204; see README.md):

    Case XML   GET  {BASE}/CaseForm.aspx?GetXML&caseid={id}
    Case HTML  GET  {BASE}/CaseForm.aspx?xsl=main.xsl&CaseID={id}
    Image/JPG  GET  {BASE}/GetBinary.aspx?Image&ImageID={imgid}&CaseID={id}&Version={ver}

where BASE defaults to:

    https://crashviewer.nhtsa.dot.gov/crashviewer/nass-CIREN

(`viewerRoot` literal found in the LegacyCIREN search page JavaScript:
 `viewerRoot = 'https://crashviewer.nhtsa.dot.gov/crashviewer/nass-CIREN'`).

Design notes
------------
* `requests.Session` is the default transport, with a descriptive User-Agent,
  retry + exponential backoff on 429/5xx, and a configurable base URL.
* The host sits behind Akamai. From some networks (notably cloud / datacenter
  IPs) Akamai returns HTTP 403 "Access Denied" to *all* programmatic requests.
  If you hit that, run from a normal/residential network, or use the
  Selenium fallback path (`SeleniumCirenClient`) which drives a real headless
  Chrome and is far less likely to be filtered.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

import requests

try:
    # curl_cffi presents a REAL browser TLS/JA3 + HTTP2 fingerprint. The host is
    # behind Akamai, which 403s plain requests/curl at the edge based on TLS
    # fingerprint (NOT IP) -- a genuine browser (or curl_cffi impersonation) passes.
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except Exception:  # pragma: no cover
    _HAS_CURL_CFFI = False

LOG = logging.getLogger("ciren.client")

# The legacy NASS-CIREN virtual directory (the real backend behind LegacyCIREN/Search).
DEFAULT_BASE_URL = "https://crashviewer.nhtsa.dot.gov/crashviewer/nass-CIREN"

DEFAULT_USER_AGENT = (
    "SAFE-ADS-Research-Scraper/1.0 (CIREN public-domain crash data; "
    "contact: ajakobss@andrew.cmu.edu) python-requests"
)

# Statuses we retry with backoff.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class CirenClientError(RuntimeError):
    """Raised for unrecoverable HTTP problems (after retries are exhausted)."""


class CirenClient:
    """Plain `requests` client for the LegacyCIREN backend."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 45.0,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        session=None,
        impersonate: str = "firefox",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.impersonate = impersonate

        if session is not None:
            self.session = session
        elif _HAS_CURL_CFFI and impersonate:
            # Let impersonation own the browser-consistent headers (incl. User-Agent):
            # overriding the UA would recreate a UA/JA3 mismatch that Akamai flags.
            self.session = cffi_requests.Session(impersonate=impersonate)
            self.session.headers.update({"Accept-Language": "en-US,en;q=0.9"})
        else:
            self.session = requests.Session()
            self.session.headers.update(
                {
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/jpeg,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )

    # -- URL builders ---------------------------------------------------------

    def case_xml_url(self, case_id: int | str) -> str:
        return f"{self.base_url}/CaseForm.aspx?GetXML&caseid={case_id}"

    def case_html_url(self, case_id: int | str) -> str:
        return f"{self.base_url}/CaseForm.aspx?xsl=main.xsl&CaseID={case_id}"

    def image_url(self, image_id: int | str, case_id: int | str, version: int | str = 0) -> str:
        # NOTE: the bare `Image` flag (no `=value`) is required -- this is how
        # the live backend distinguishes an image request. Verified: the server
        # responds with Content-Type "img/jpg" for a valid ImageID.
        return (
            f"{self.base_url}/GetBinary.aspx?Image"
            f"&ImageID={image_id}&CaseID={case_id}&Version={version}"
        )

    # -- low-level fetch with retry/backoff -----------------------------------

    def _get(self, url: str, *, want_binary: bool = False) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except Exception as exc:  # network-level failure (requests or curl_cffi)
                last_exc = exc
                self._sleep_backoff(attempt, reason=f"network error: {exc}")
                continue

            if resp.status_code in RETRY_STATUSES:
                retry_after = resp.headers.get("Retry-After")
                self._sleep_backoff(attempt, reason=f"HTTP {resp.status_code}", retry_after=retry_after)
                last_exc = CirenClientError(f"HTTP {resp.status_code} for {url}")
                continue

            if resp.status_code == 403:
                # Almost always Akamai edge filtering of this client IP, not a
                # per-case problem. Surface it clearly; do not retry forever.
                raise CirenClientError(
                    f"HTTP 403 (Access Denied) for {url}. The host is behind Akamai "
                    f"and is blocking this network/IP. Run from a different network "
                    f"or use the Selenium fallback (SeleniumCirenClient)."
                )

            if resp.status_code != 200:
                raise CirenClientError(f"HTTP {resp.status_code} for {url}")

            return resp

        raise CirenClientError(f"Giving up on {url}: {last_exc}")

    def _sleep_backoff(self, attempt: int, *, reason: str, retry_after: Optional[str] = None) -> None:
        if attempt >= self.max_retries:
            return
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.backoff_base ** attempt
        else:
            delay = self.backoff_base ** attempt
        delay += random.uniform(0, 0.75)  # jitter
        LOG.warning("retry %d/%d (%s); sleeping %.1fs", attempt + 1, self.max_retries, reason, delay)
        time.sleep(delay)

    # -- public API -----------------------------------------------------------

    def fetch_case_xml(self, case_id: int | str) -> str:
        """Return the case XML document text for a case id.

        The document is an XML tree whose root is `<case CaseID="..." ...>`
        following the CirenNNNN.xsd schema (e.g. Ciren2004.xsd).
        """
        url = self.case_xml_url(case_id)
        LOG.debug("GET case xml %s", url)
        resp = self._get(url)
        try:
            if not getattr(resp, "encoding", None):
                resp.encoding = "utf-8"
        except Exception:
            pass
        return resp.text

    def download_image(self, image_id: int | str, case_id: int | str, version: int | str = 0) -> bytes:
        """Download the raw image bytes (JPEG) for an ImageID."""
        url = self.image_url(image_id, case_id, version)
        LOG.debug("GET image %s", url)
        resp = self._get(url, want_binary=True)
        return resp.content


# ---------------------------------------------------------------------------
# Selenium fallback (OPTIONAL).
#
# Use this only if `CirenClient` keeps getting HTTP 403 from Akamai, or if the
# page ever moves to a JS/postback-rendered model. It drives a real headless
# Chrome (which carries a genuine browser fingerprint) to fetch the same URLs.
#
# Selenium is an OPTIONAL dependency -- it is imported lazily so the module
# loads fine without it.
# ---------------------------------------------------------------------------


class SeleniumCirenClient:
    """Headless-Chrome fallback that mirrors CirenClient's public API.

    Requires: `pip install selenium` and a matching chromedriver on PATH
    (or pass `driver_path`). Chrome/Chromium must be installed.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        driver_path: Optional[str] = None,
        page_load_timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.driver_path = driver_path
        self.page_load_timeout = page_load_timeout
        self._driver = None

    def _ensure_driver(self):
        if self._driver is not None:
            return self._driver
        from selenium import webdriver  # lazy import
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument(f"--user-agent={self.user_agent}")
        if self.driver_path:
            from selenium.webdriver.chrome.service import Service

            self._driver = webdriver.Chrome(service=Service(self.driver_path), options=opts)
        else:
            self._driver = webdriver.Chrome(options=opts)
        self._driver.set_page_load_timeout(self.page_load_timeout)
        return self._driver

    # Same builders as CirenClient.
    def case_xml_url(self, case_id):  # noqa: D401
        return f"{self.base_url}/CaseForm.aspx?GetXML&caseid={case_id}"

    def image_url(self, image_id, case_id, version=0):
        return (
            f"{self.base_url}/GetBinary.aspx?Image"
            f"&ImageID={image_id}&CaseID={case_id}&Version={version}"
        )

    def fetch_case_xml(self, case_id) -> str:
        driver = self._ensure_driver()
        driver.get(self.case_xml_url(case_id))
        # The XML is served as a document; grab the raw source.
        # Chrome wraps XML in a viewer, so prefer the <body> text / page_source.
        src = driver.page_source
        # Strip Chrome's XML pretty-printer wrapper if present.
        start = src.find("<case")
        return src[start:] if start != -1 else src

    def download_image(self, image_id, case_id, version=0) -> bytes:
        """Fetch image bytes by reusing Selenium's authenticated cookies in requests."""
        driver = self._ensure_driver()
        # Warm the session so Akamai/ASP.NET cookies exist, then reuse them.
        driver.get(self.base_url + "/")
        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
        sess = requests.Session()
        sess.headers["User-Agent"] = self.user_agent
        resp = sess.get(self.image_url(image_id, case_id, version), cookies=cookies, timeout=60)
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None
