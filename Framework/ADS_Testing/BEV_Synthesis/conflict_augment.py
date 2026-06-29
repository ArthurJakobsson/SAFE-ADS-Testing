"""Non-destructively augment a SAFE DSL with an explicit `Conflict` block.

The BEV synthesizer can place a crash deterministically if it knows which vehicle is at fault
and how the two collide. Rather than re-running Stage 3 (which re-rolls the stochastic DSL and can
flip directions/actions), this tool reads an *existing* `DSL_extraction_results.pkl`, makes ONE
extra multimodal call per case to extract the conflict, and writes a NEW pickle with a `Conflict`
key merged in. Every original DSL field (Actors, Initial_position, Road network, ...) is copied
verbatim — directions never change.

Usage (from Framework/):
    python ADS_Testing/BEV_Synthesis/conflict_augment.py \
        --dsl Experiment_results/DSL_results_<ts>/DSL_extraction_results.pkl --gpt "$SAFE_MODEL"

Writes Experiment_results/DSL_results_<newts>/DSL_extraction_results.pkl (+ <id>_conflict.txt raw).
Reads OPENAI_BASE_URL / OPENAI_API_KEY like the rest of the pipeline.
"""
import argparse
import base64
import json
import os
import pickle
import re
from datetime import datetime

from openai import OpenAI

CONFLICT_SYSTEM = (
    "You are a crash reconstruction expert. From the crash summary and the bird's-eye sketch, "
    "identify the collision. Respond with ONLY a JSON object (no prose) of the form:\n"
    '{"at_fault_vehicle": "Vehicle_1" or "Vehicle_2", '
    '"struck_vehicle": "Vehicle_1" or "Vehicle_2", '
    '"impact_type": one of ["head-on","rear-end","angle","sideswipe","T-bone","single-vehicle"], '
    '"point_of_impact": "<short phrase>", "description": "<one sentence>"}\n'
    "at_fault_vehicle is the one that initiated the conflict (crossed the centreline, turned across "
    "the other's path, merged improperly, ran a control device, etc.). Vehicle_1 is the case "
    "vehicle (V1).")


def _encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json_block(text):
    """Tolerant JSON extractor (same idiom as Stage 3)."""
    for pat in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1)
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("no JSON object in:\n" + text[:300])


def extract_conflict(case_id, dataset, model, raw_dir):
    sketch = _encode(os.path.join(dataset, str(case_id), "Sketch.jpg"))
    with open(os.path.join(dataset, str(case_id), "Summary.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        summary = f.read()
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": CONFLICT_SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Sketch:\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sketch}"}},
                {"type": "text", "text": f"\nSummary:\n{summary}"}]}],
        response_format={"type": "text"}, temperature=0.2, max_completion_tokens=300,
        top_p=1, frequency_penalty=0, presence_penalty=0)
    out = resp.choices[0].message.content
    with open(os.path.join(raw_dir, f"{case_id}_conflict.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    try:
        return json.loads(_extract_json_block(out))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsl", required=True, help="existing DSL_extraction_results.pkl")
    ap.add_argument("--gpt", default=os.environ.get("SAFE_MODEL", "gpt-4o"))
    ap.add_argument("--dataset", default="./Crash_dataset")
    ap.add_argument("--out", default="./Experiment_results")
    args = ap.parse_args()

    with open(args.dsl, "rb") as f:
        dsls = pickle.load(f)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.abspath(os.path.join(args.out, f"DSL_results_{ts}"))
    os.makedirs(out_dir, exist_ok=True)

    augmented = []
    for entry in dsls:
        cid = str(entry.get("Scenario", "?"))
        entry = dict(entry)  # copy; all original fields preserved verbatim
        entry["Conflict"] = extract_conflict(cid, args.dataset, args.gpt, out_dir)
        af = entry["Conflict"].get("at_fault_vehicle", "?")
        print(f"[conflict] {cid}: at_fault={af} impact={entry['Conflict'].get('impact_type','?')}")
        augmented.append(entry)

    out_pkl = os.path.join(out_dir, "DSL_extraction_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(augmented, f)
    print(f"[conflict] wrote {len(augmented)} augmented cases -> {out_pkl}")


if __name__ == "__main__":
    main()
