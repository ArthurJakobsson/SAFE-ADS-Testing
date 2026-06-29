"""
ciren_parse.py -- pure parsing for the LegacyCIREN case XML.

Everything site-specific (which XML tags hold the narrative and the scene
sketch) lives HERE, so when NHTSA changes their schema you only edit one file.

Verified against the live case XML for cases 109204, 115565, 120523 on
2026-06-26 (each produced a byte-exact match of the existing
Framework/Crash_dataset/<id>/Summary.txt).

Case XML shape (root `<case CaseID="..." CaseStr="..." ...>`):

    <case CaseID="109204" CaseStr="2004-38" ...>
      ...
      <summary>Vehicle one (V1 - case vehicle) ...</summary>          <-- narrative
      ...
      <imgform>
        <scenedrawings>
          <scene type="jpg" desc="">555113862</scene>                 <-- SCENE SKETCH ImageID
        </scenedrawings>
        <crashscene>...</crashscene>      (on-scene photos, NOT the sketch)
        <diagram>...</diagram>            (intrusion/interior sketches)
      </imgform>
      ...
    </case>

The scene diagram / sketch (== the existing Sketch.jpg) is the single
`<imgform>/<scenedrawings>/<scene>` ImageID. We resolve its bytes via
`GetBinary.aspx?Image&ImageID=...&CaseID=...&Version=...` (see ciren_client).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from lxml import etree

# ---------------------------------------------------------------------------
# Site-specific selectors. Tags in the CIREN XML are lower-case. We match
# case-insensitively defensively, since schema casing has drifted over years.
# ---------------------------------------------------------------------------

# Narrative: the <summary> element directly under <case>. (There is also a
# <casesummary> which PREPENDS a short crash-type label like
# "Vehicle to vehicle Angle/sideswipe" -- we deliberately use <summary>, which
# is exactly what the existing dataset's Summary.txt files contain.)
NARRATIVE_TAGS = ("summary",)

# Scene sketch: <imgform>/<scenedrawings>/<scene>. We also accept a bare
# <scenedrawings>/<scene> anywhere, as a fallback.
SCENE_SKETCH_XPATH = (
    ".//imgform/scenedrawings/scene "
    "| .//scenedrawings/scene"
)


@dataclass
class SceneSketchRef:
    image_id: str
    ext: str = "jpg"       # from <scene type="jpg">
    version: int = 0       # GetBinary Version; scene drawings are version 0
    desc: str = ""

    def __bool__(self) -> bool:
        return bool(self.image_id)


def _parse_xml(xml_text: str) -> etree._Element:
    """Parse the case XML into an lxml tree (lower-cased tag namespace-free)."""
    # The document declares xsi:noNamespaceSchemaLocation but no default ns,
    # so tags are accessible directly. Use a recovering parser for robustness.
    parser = etree.XMLParser(recover=True, huge_tree=True, encoding="utf-8")
    # lxml wants bytes for encoding-declared docs.
    root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    if root is None:
        raise ValueError("Could not parse case XML (empty / malformed root)")
    return root


def _local(tag: str) -> str:
    return etree.QName(tag).localname.lower() if isinstance(tag, str) else ""


def extract_case_id(xml_text: str) -> Optional[str]:
    """Return the CaseID attribute from <case>, if present."""
    root = _parse_xml(xml_text)
    cid = root.get("CaseID") or root.get("caseid")
    return cid


def extract_narrative(xml_text: str) -> str:
    """Extract the crash narrative and collapse it to a single prose paragraph.

    Returns "" if no narrative element is found.
    """
    root = _parse_xml(xml_text)
    for tag in NARRATIVE_TAGS:
        # Case-insensitive search over all descendants.
        for el in root.iter():
            if _node_local(el) == tag:
                text = _collapse_whitespace(_all_text(el))
                if text:
                    return text
    return ""


def extract_scene_sketch(xml_text: str) -> Optional[SceneSketchRef]:
    """Resolve the scene diagram / sketch image reference.

    Looks for <imgform>/<scenedrawings>/<scene>...</scene> and returns its
    ImageID plus the `type`/`desc` attributes. Returns None if no scene sketch
    exists for the case.
    """
    root = _parse_xml(xml_text)

    # Walk to imgform -> scenedrawings -> scene (case-insensitive), then
    # fall back to any scenedrawings/scene.
    scene_el = _find_scene_under(root, ("imgform", "scenedrawings", "scene"))
    if scene_el is None:
        scene_el = _find_scene_under(root, ("scenedrawings", "scene"))
    if scene_el is None:
        return None

    image_id = (scene_el.text or "").strip()
    if not image_id:
        return None
    ext = (scene_el.get("type") or "jpg").strip().lower() or "jpg"
    desc = (scene_el.get("desc") or "").strip()
    return SceneSketchRef(image_id=image_id, ext=ext, version=0, desc=desc)


def list_all_scene_sketches(xml_text: str) -> List[SceneSketchRef]:
    """Return every <scenedrawings>/<scene> ref (usually one). Useful if a case
    has multiple scene drawings and you want to pick/fallback."""
    root = _parse_xml(xml_text)
    out: List[SceneSketchRef] = []
    for el in root.iter():
        if _node_local(el) == "scene" and _parent_local(el) == "scenedrawings":
            image_id = (el.text or "").strip()
            if image_id:
                out.append(
                    SceneSketchRef(
                        image_id=image_id,
                        ext=(el.get("type") or "jpg").strip().lower() or "jpg",
                        version=0,
                        desc=(el.get("desc") or "").strip(),
                    )
                )
    return out


# -- internal helpers --------------------------------------------------------

def _node_local(el) -> str:
    try:
        return etree.QName(el).localname.lower()
    except ValueError:
        return str(el.tag).lower()


def _parent_local(el) -> str:
    p = el.getparent()
    return _node_local(p) if p is not None else ""


def _find_scene_under(root, chain) -> Optional[object]:
    """Find the first element matching a tag chain (case-insensitive)."""
    current_level = [root]
    # Descend each named level.
    for depth, name in enumerate(chain):
        nxt = []
        for node in current_level:
            for child in node.iter():
                if child is node:
                    continue
                if _node_local(child) == name:
                    # Only keep children whose parent matches the previous chain
                    # element (for depth>0) -- but to stay simple and robust we
                    # only enforce the immediate parent for the final 'scene'.
                    nxt.append(child)
        current_level = nxt
        if not current_level:
            return None
    # For the final level, prefer one whose parent matches chain[-2].
    if len(chain) >= 2:
        for node in current_level:
            if _parent_local(node) == chain[-2]:
                return node
    return current_level[0] if current_level else None


def _all_text(el) -> str:
    """Concatenate all text inside an element (including nested), with spaces."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p for p in parts if p)


_WS_RE = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces -> one prose paragraph."""
    return _WS_RE.sub(" ", text).strip()
