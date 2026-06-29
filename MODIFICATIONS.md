# Modifications to the original SAFE framework

This file records every change made to the **original SAFE code** (the files uploaded in
`Upload SAFE codes` / `Update SAFE Framework`), as distinct from the bolt-on tooling added later
(the free-VLM serving stack, BEV synthesis, CIREN tools, and the debug dashboard — see
[`SETUP.md`](SETUP.md)). The bolt-on tooling is *not* part of original SAFE and is not listed here.

## Original SAFE files
`Framework/Meta_Message_Extraction.py`, `Framework/Prompts_Generation.py`,
`Framework/Scenario_Representation_Extraction.py`, and `Framework/ADS_Testing/{BeamNG,MetaDrive}/`.

## Current status: **no changes to original SAFE structure.**

The deterministic-crash work (the per-case `Conflict` descriptor) is implemented entirely in
**bolt-on tooling**, so the original SAFE pipeline is byte-for-byte unchanged. Verify with:

```bash
git diff <last-SAFE-commit> -- Framework/Scenario_Representation_Extraction.py \
                                Framework/Meta_Message_Extraction.py \
                                Framework/Prompts_Generation.py   # → empty
```

### How `Conflict` is produced (non-destructive, bolt-on)
`Framework/ADS_Testing/BEV_Synthesis/conflict_augment.py` reads an existing
`DSL_extraction_results.pkl`, makes one extra multimodal call per case to extract a `Conflict`
block `{at_fault_vehicle, struck_vehicle, impact_type, point_of_impact, description}`, and writes a
**new** pickle with every original DSL field copied verbatim. It never re-runs Stage 3, so vehicle
directions/actions are never re-rolled.

```bash
# from Framework/
python ADS_Testing/BEV_Synthesis/conflict_augment.py \
    --dsl Experiment_results/DSL_results_<ts>/DSL_extraction_results.pkl --gpt "$SAFE_MODEL"
```

The BEV synthesizer (`bev_from_dsl.py`) consumes `Conflict.at_fault_vehicle` to place the
collision deterministically (with a heuristic fallback when the field is absent).

### History (why this file exists)
An earlier iteration added an `extract_conflict()` pass *inside*
`Scenario_Representation_Extraction.py` and re-ran Stage 3 to populate it. That re-ran the
stochastic DSL extraction (`temperature=1`) and **drifted a vehicle direction** (case 119489 V1
flipped `E2W`→`W2E`, contradicting the "facing west" narrative). It was reverted in favor of the
standalone non-destructive augmenter above, restoring the original SAFE file to pristine.
