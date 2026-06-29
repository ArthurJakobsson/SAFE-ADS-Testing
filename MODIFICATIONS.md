# Modifications to the original SAFE framework

This file records every change made to the **original SAFE code** (the files uploaded in
`Upload SAFE codes` / `Update SAFE Framework`), as distinct from the bolt-on tooling added later
(the free-VLM serving stack, BEV synthesis, CIREN tools, and the debug dashboard — see
[`SETUP.md`](SETUP.md)). The bolt-on tooling is *not* part of original SAFE and is not listed here.

## Original SAFE files (for reference)
`Framework/Meta_Message_Extraction.py`, `Framework/Prompts_Generation.py`,
`Framework/Scenario_Representation_Extraction.py`, and `Framework/ADS_Testing/{BeamNG,MetaDrive}/`.

## Changes

### 1. `Framework/Scenario_Representation_Extraction.py` — add a `Conflict` descriptor (Stage 3)
**What:** Added a new function `extract_conflict(record, model, results_path)` and called it once
per case in `main()` (both validation branches), attaching the result to the per-case DSL dict as
a new top-level key `Conflict`:

```json
"Conflict": {
  "at_fault_vehicle": "Vehicle_1|Vehicle_2",
  "struck_vehicle":   "Vehicle_1|Vehicle_2",
  "impact_type":      "head-on|rear-end|angle|sideswipe|T-bone|single-vehicle",
  "point_of_impact":  "<short phrase>",
  "description":      "<one sentence>"
}
```

It makes one extra multimodal call (summary + crash sketch) and writes a raw `<id>_conflict.txt`
alongside the existing Stage-3 artifacts. JSON is parsed with the existing tolerant
`_extract_json_block`.

**Why:** The BEV synthesizer (bolt-on, `ADS_Testing/BEV_Synthesis/bev_from_dsl.py`) previously had
to *infer* the collision from the lane-keeping rollout, so angle / T-bone / merge crashes never
converged and the vehicles passed without colliding. The explicit `Conflict.at_fault_vehicle` lets
the BEV place the collision **deterministically** (steer the at-fault vehicle to intercept the
victim) instead of guessing.

**Compatibility:** Purely **additive** — no existing DSL field, prompt, or Stage-1/2/4 behavior is
changed. `Conflict` is a new optional key; consumers that ignore it are unaffected. If the model
returns unparseable JSON, the field is `{}`.

> Note: the original Stage-3 file had already been modified once before this work, by the
> free-VLM addition, to add the tolerant `_extract_json_block` JSON extractor (see `SETUP.md`).
> No other original SAFE file (`Meta_Message_Extraction.py`, `Prompts_Generation.py`, the BeamNG /
> MetaDrive generators) was touched.
