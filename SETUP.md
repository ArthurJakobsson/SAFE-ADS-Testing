# SAFE — free-LLM setup, nuPlan-style BEVs, CIREN dataset, and debug dashboard

This document covers the additions made to run SAFE without a paid OpenAI key, to generate
nuPlan-style BEVs from the extracted DSL, to grow the dataset from CIREN, and to debug it all
from a single HTML page. The original GPT-4o pipeline logic is unchanged — we only swapped the
model endpoint and added new modules.

```
serving/                         # vLLM (free Qwen2.5-VL) serving — replaces the OpenAI API
  install_vllm_env.sh            # one-time: build `safe-vllm` conda env (py3.11) + install vLLM
  serve_local.sh                 # serve 3B-AWQ on this 12GB GPU (smoke test)
  serve_server.sh                # serve 72B on the 7xL40S server (full quality)
  env.sh                         # `source` it: points the repo's OpenAI() client at vLLM
Framework/ADS_Testing/BEV_Synthesis/
  bev_from_dsl.py                # DSL -> 18-channel nuPlan-style BEV (.npz + .png + .json)
  map_visualizer_nuplan.py       # nuPlan-correct 18-channel renderer
  output/                        # generated BEVs + manifest.json
Framework/Crash_dataset_tools/   # Track B — CIREN legacy scraper (see its own README)
dashboard/
  build_dashboard.py             # regenerate dashboard.html from current outputs
  dashboard.html                 # open in a browser to debug both tracks
requirements.txt                 # client-side deps (NOT the vLLM server deps)
```

## Why a free model needed almost no code change

Both LLM stages call the stock OpenAI SDK with `image_url`/base64 multimodal messages
(`Meta_Message_Extraction.py`, `Scenario_Representation_Extraction.py`). That is exactly the
wire format vLLM's OpenAI-compatible server speaks for **Qwen2.5-VL**. `OpenAI()` auto-reads
`OPENAI_BASE_URL` + `OPENAI_API_KEY`, so swapping GPT-4o → self-hosted Qwen2.5-VL is just env
vars + the existing `--gpt` flag. The only source edit is a more tolerant JSON extractor in
Stage 3 (`_extract_json_block`), since open VLMs are less consistent than GPT-4o about ```` ```json ```` fences.

---

## Track A — run the pipeline on a free model + make BEVs

### 1. One-time install (already done on this machine)
```bash
bash serving/install_vllm_env.sh     # creates conda env `safe-vllm` with vLLM + torch(cuda)
```

### 2. Serve the model
Local smoke test (this RTX 3080 Ti, 12 GB — uses the 4-bit 3B model):
```bash
bash serving/serve_local.sh          # serves Qwen2.5-VL-3B-Instruct-AWQ on :8000
```
Full quality on the L40S server:
```bash
TP=4 bash serving/serve_server.sh    # serves Qwen2.5-VL-72B-Instruct across 4 GPUs
```
Verify: `curl http://localhost:8000/v1/models`

### 3. Point the repo at the server and run the existing 4 stages
```bash
source serving/env.sh                # exports OPENAI_BASE_URL + SAFE_MODEL
cd Framework
python Meta_Message_Extraction.py            --gpt "$SAFE_MODEL"
python Prompts_Generation.py                 --data Experiment_results/Meta_Message_results_*/meta_data_results.pkl
python Scenario_Representation_Extraction.py --gpt "$SAFE_MODEL" \
       --prompts Experiment_results/Prompts_generation_results_*/ \
       --meta_message Experiment_results/Meta_Message_results_*/meta_data_results.pkl
```
This produces the repo's intended output: `Experiment_results/DSL_results_*/DSL_extraction_results.pkl`.

### 4. Generate nuPlan-style BEVs from the DSL
```bash
cd Framework/ADS_Testing/BEV_Synthesis
# demo (no LLM): one BEV per road type
python bev_from_dsl.py --demo --out ./output
# from a real run:
python bev_from_dsl.py --out ./output \
  --dsl  ../../Experiment_results/DSL_results_*/DSL_extraction_results.pkl \
  --meta ../../Experiment_results/Meta_Message_results_*/meta_data_results.pkl
```
Each case writes `<id>.npz` (`gt_bev_masks` = (18,200,200) int8, the UniScene-v2 nuPlan key),
`<id>.png` (viewable), and `<id>.json` (scene sidecar). Channel layout is documented in
`map_visualizer_nuplan.py` (static 0–7 / dynamic 8–14 / dividers 15–17).

### 5. Debug from one page
```bash
python dashboard/build_dashboard.py     # rescans outputs, rebuilds dashboard.html
xdg-open dashboard/dashboard.html        # or just open the file in a browser
```

---

## Track B — CIREN legacy scraper
See `Framework/Crash_dataset_tools/README.md`. It writes new `<case_id>/Summary.txt` +
`Sketch.jpg` pairs (the same format Track A consumes) plus a `manifest.json` the dashboard reads.

---

## What YOU need to do manually (not automatable here)

- **BeamNG (Stage 4, optional):** BeamNG.tech requires a **free academic license** you must
  request from BeamNG, plus a Windows/`beamngpy` setup. It is only needed for the BeamNG
  simulation path; the LLM extraction, BEV synthesis, and the **MetaDrive** simulation path do
  not need it. If you only want scenarios + BEVs, you can skip BeamNG entirely.
- **MetaDrive (Stage 4, optional):** `pip install metadrive-simulator` in a client env if you
  want to run the simulation/GIF outputs. Not required for DSL extraction or BEVs.
- **Server transfer:** to use the 72B model, copy this repo + the `safe-vllm` env (or rerun
  `install_vllm_env.sh`) onto the L40S box and run `serve_server.sh`. The client code is
  identical; only `OPENAI_BASE_URL` changes (point it at the server host). For the full
  7×L40S runbook — picking a **stronger** free VLM, FP8/tensor-parallel sizing, networking, and
  every gotcha carried over from this workstation — see
  [`serving/SERVER_SETUP.md`](serving/SERVER_SETUP.md).
- **Hugging Face access:** the Qwen2.5-VL weights are open and ungated, so no token is needed.
  If you later try a *gated* model, run `huggingface-cli login` first. Setting `HF_TOKEN` also
  raises download rate limits.
- **CIREN scraping verification:** confirm the live LegacyCIREN case/image endpoints and review
  NHTSA's robots.txt / Terms of Use before a bulk sweep (the scraper is rate-limited and
  idempotent, but the endpoints must be validated against a live case first — see Track B README).
- **GPU note:** the local 3080 Ti (12 GB, shared with the desktop) only fits the 3B-AWQ model
  for smoke tests. Treat local runs as correctness checks; do quality runs on the L40S server.
```
