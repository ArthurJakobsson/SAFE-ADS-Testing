# SAFE on the 7×L40S server — serving a stronger free VLM

A runbook for standing up the LLM half of SAFE on the **7×L40S** box and pointing the
existing pipeline at a **bigger, free, downloadable** model than the workstation's 3B/72B.
It folds in every gotcha we hit getting vLLM running on the desktop so you don't re-hit them.

> The client code does **not** change between machines. The whole "swap" is three env vars
> (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `SAFE_MODEL`). Everything below is server-side serving
> plus model selection — see [`../SETUP.md`](../SETUP.md) for what the pipeline does with it.

---

## 0. The one hard constraint that decides the model

Both LLM stages (`Meta_Message_Extraction.py`, `Scenario_Representation_Extraction.py`) send
**images** — the crash sketch and BEV are passed as `image_url`/base64 multimodal messages, up
to **3 images per prompt**. So the model **must be a vision-language model (VLM)**. A text-only
LLM (Llama-3.1-70B, Qwen2.5-72B-*text*, DeepSeek, etc.) will not work — it has no image tower.

A second, softer constraint made the original GPT-4o → open swap painless: Qwen2.5-VL speaks the
**exact** OpenAI multimodal wire format vLLM exposes. Staying inside the **Qwen-VL family** keeps
that property, so the integration risk of a bigger model is ~zero. Switching families
(InternVL, Llama-4-VL, Pixtral) also works through vLLM's OpenAI server, but the chat template /
prompt formatting differs and is worth a smoke test — the tolerant JSON extractor we already
added in Stage 3 (`_extract_json_block`) absorbs most of that variance.

---

## 1. Hardware budget for 7×L40S

- Each **L40S = 48 GB**. Total **= 336 GB** VRAM.
- L40S is **Ada Lovelace → native FP8 (E4M3)** and BF16. No FP4. **FP8 is your size lever** to
  fit models that don't fit in bf16.
- Rough weight memory = `params × bytes` (bf16 = 2 B/param, fp8 = 1 B/param). Add ~15–30% on
  top for the KV cache + activations at your `--max-model-len`.

| Model size | bf16 weights | fits in bf16? | suggested TP |
|-----------:|-------------:|---------------|--------------|
| 32B        | ~64 GB       | yes, easily   | TP=2 (2 GPUs)|
| 72–78B     | ~145–156 GB  | yes           | TP=4 (4 GPUs)|
| 235B (MoE) | ~470 GB      | **no** → FP8 ~235 GB | TP=6, FP8 |

**TP gotcha — you cannot just say `TP=7`.** Tensor-parallel size must *evenly divide* the
model's attention-head count. 7 is prime and almost never divides a head count, so vLLM will
refuse `--tensor-parallel-size 7`. Use **TP ∈ {2, 4}** (or **6** for the MoE), pin the GPUs with
`CUDA_VISIBLE_DEVICES`, and either leave the spare cards idle or run a **second replica** of a
smaller model on the remaining GPUs for throughput.

---

## 2. Model recommendation (pick one tier)

All are open-weights and downloadable from Hugging Face. **Confirm the exact repo id on the
[Qwen](https://huggingface.co/Qwen) / [OpenGVLab](https://huggingface.co/OpenGVLab) HF orgs
before you `vllm serve`** — ids below are the expected names, not copy-paste guarantees.

**Tier A — recommended workhorse (best effort/risk ratio): `Qwen3-VL-32B-Instruct`**
- Newest Qwen generation; a clear step up over Qwen2.5-VL on multimodal reasoning/OCR/spatial,
  and competitive with the old 72B at a fraction of the footprint.
- Same Qwen-VL wire format → **drop-in**, no client changes.
- Fits **TP=2**, leaving 5 GPUs free (run a 2nd replica, or give it more KV cache for long DSL prompts).

**Tier B — maximum quality, uses most of the box: `Qwen3-VL-235B-A22B-Instruct` (FP8)**
- Strongest open VLM as of early 2026 (rivals closed frontier models on multimodal benchmarks).
- MoE (235B total, ~22B active) → **fast** despite the size. bf16 won't fit; serve the **FP8**
  checkpoint (or `--quantization fp8`), **TP=6**, ~235 GB weights + KV across 6 L40S.
- Still Qwen-VL family → drop-in client.

**Tier C — proven fallback / different lineage**
- `Qwen2.5-VL-72B-Instruct` — exactly what `serve_server.sh` already targets; lowest risk if
  Qwen3-VL serving hits a vLLM-version snag. TP=4.
- `InternVL3-78B` (OpenGVLab) — SOTA MMMU among open models; TP=4. Different family → smoke-test
  the prompt format first.

> **Recommendation:** start with **Tier A (Qwen3-VL-32B)** to validate the end-to-end run on the
> server quickly, then move to **Tier B (Qwen3-VL-235B FP8)** for the final-quality dataset pass.

---

## 3. Setup, step by step (server)

### 3.1 Build the serving env
Base conda on these boxes is usually Python 3.13, which vLLM does **not** support — make a fresh
**py3.11** env (same lesson as the workstation):

```bash
# from the repo root on the server
ENV_NAME=safe-vllm bash serving/install_vllm_env.sh
```

> ⚠️ **Qwen3-VL needs a newer vLLM/transformers than the workstation's pinned `vllm>=0.7.2`.**
> For any Qwen3-VL or Llama-4-VL model, install the **latest** vLLM release instead:
> ```bash
> CONDA_BASE="$(conda info --base)"
> "$CONDA_BASE/envs/safe-vllm/bin/pip" install -U "vllm" "transformers" "huggingface_hub[hf_transfer]"
> ```
> Qwen2.5-VL / InternVL3 are fine on the pinned version.

### 3.2 Download the weights
Qwen and InternVL VLM weights are **open / ungated → no token needed**. Use `hf_transfer` for a
fast multithreaded pull (already installed by the env script):

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct   # or your chosen repo id
```
- If you ever pick a **gated** model, run `huggingface-cli login` first.
- Set `HF_HOME` to a big disk if `~/.cache` is small — a 235B FP8 checkpoint is ~235 GB.

### 3.3 Serve it
`serve_server.sh` is already parameterized by `MODEL`/`TP`/`PORT`/`MAX_LEN`, so for the standard
tiers you only change env vars. It carries the workstation's hard-won flags:
`VLLM_USE_FLASHINFER_SAMPLER=0` (avoid the flashinfer JIT sampler that needs `ninja`),
`VLLM_ATTENTION_BACKEND=FLASH_ATTN`, and `--limit-mm-per-prompt '{"image": 3}'` (the pipeline
sends up to 3 images).

```bash
# Tier A — Qwen3-VL-32B on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 MODEL=Qwen/Qwen3-VL-32B-Instruct TP=2 \
  bash serving/serve_server.sh

# Tier C — the script's default 72B on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 bash serving/serve_server.sh

# Tier B — Qwen3-VL-235B MoE in FP8 on 6 GPUs (add --quantization fp8 if the repo is bf16)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 TP=6 MAX_LEN=32768 \
  bash serving/serve_server.sh
```

Two server-specific additions over the workstation script:

1. **Bind to the network** if the SAFE client runs on a *different* host. Either pass
   `--host 0.0.0.0` (edit `serve_server.sh`, or export and add it) and open the port in the
   firewall, **or** keep it on `localhost` and SSH-tunnel from the client:
   `ssh -N -L 8000:localhost:8000 user@<server>`.
2. **Run it detached** — a 72B/235B load takes minutes and you don't want it tied to your SSH
   session. Use `tmux`/`screen`, `nohup ... &`, or a systemd unit. Watch the log for
   `Application startup complete` before sending requests.

### 3.4 Point the pipeline at the server
On whichever machine runs the SAFE stages, set the three vars (this is the *entire* client swap):

```bash
export OPENAI_API_KEY=EMPTY                                  # vLLM ignores it but the SDK requires a value
export OPENAI_BASE_URL=http://<server-ip-or-localhost>:8000/v1
export SAFE_MODEL=Qwen/Qwen3-VL-32B-Instruct                # MUST equal the server's --served-model-name
```
`serving/env.sh` does exactly this — override its defaults inline:
```bash
OPENAI_BASE_URL=http://10.0.0.42:8000/v1 SAFE_MODEL=Qwen/Qwen3-VL-32B-Instruct source serving/env.sh
```

### 3.5 Run the existing stages (unchanged)
```bash
cd Framework
python Meta_Message_Extraction.py            --gpt "$SAFE_MODEL"
python Prompts_Generation.py                 --data Experiment_results/Meta_Message_results_*/meta_data_results.pkl
python Scenario_Representation_Extraction.py --gpt "$SAFE_MODEL" \
       --prompts Experiment_results/Prompts_generation_results_*/ \
       --meta_message Experiment_results/Meta_Message_results_*/meta_data_results.pkl
```

---

## 4. Insight carried over from the workstation setup

These cost time the first time — they're already baked into the scripts, listed here so you know
*why* and can debug fast:

- **Python 3.11 env, not base.** Base conda (3.13) can't install vLLM. `install_vllm_env.sh`
  creates a clean `safe-vllm` py3.11 env.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`.** Otherwise vLLM JIT-compiles a flashinfer sampler at
  runtime and needs `ninja`/`nvcc` on PATH; disabling it uses the native PyTorch sampler. The
  serve scripts also prepend the env's `bin/` to PATH so any JIT helpers resolve.
- **`VLLM_ATTENTION_BACKEND=FLASH_ATTN`** for the stable attention path.
- **`--limit-mm-per-prompt '{"image": 3}'`** — vLLM rejects multi-image prompts above its default
  unless you raise this. The pipeline sends up to 3.
- **`--served-model-name` must equal `SAFE_MODEL`/`--gpt`.** The OpenAI client posts
  `model=<name>` and vLLM 404s on a mismatch. This is the #1 silent failure.
- **`OPENAI_API_KEY=EMPTY`.** The OpenAI SDK refuses to start without *some* key string; vLLM
  doesn't check it.
- **Open VLMs are looser with `​```json` fences than GPT-4o.** Stage 3's `_extract_json_block`
  tolerates that. If you adopt a non-Qwen family and see DSL parse failures, that extractor is
  where to loosen further.
- **Stopping the server:** use `bash serving/stop.sh`. Do **not** `pkill -f vllm` from a shell
  whose own command line contains "vllm" — it matches and kills itself; `stop.sh` uses `pgrep`
  (which excludes its own PID) against `bin/vllm serve`.

---

## 5. Verify & smoke-test

```bash
# server is up and advertising the model id you'll pass as --gpt
curl http://<server>:8000/v1/models

# one-shot multimodal round-trip (text-only here; swap in an image_url to test the vision tower)
curl http://<server>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"'"$SAFE_MODEL"'",
  "messages":[{"role":"user","content":"Reply with the single word OK."}]
}'
```
Then run **one** case through `Meta_Message_Extraction.py` before launching a full sweep.

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `404 model not found` from the client | `SAFE_MODEL` ≠ server `--served-model-name`. Make them identical. |
| `--tensor-parallel-size 7` errors | TP must divide the head count; 7 doesn't. Use TP=2/4/6, idle the spare GPUs. |
| OOM at load | Drop a tier, lower `--max-model-len`, lower `--gpu-memory-utilization`, or serve the **FP8** checkpoint. |
| `ninja`/flashinfer JIT error | Confirm `VLLM_USE_FLASHINFER_SAMPLER=0` is exported (it is, in the serve scripts). |
| Qwen3-VL: "unknown model type" / processor error | vLLM/transformers too old — `pip install -U vllm transformers` (§3.1). |
| Client connects but hangs forever | Model still loading; wait for `Application startup complete`. 72B/235B loads take minutes. |
| Client on another host can't reach :8000 | Serve with `--host 0.0.0.0` + open firewall, or SSH-tunnel the port. |
| DSL JSON won't parse on a non-Qwen model | Different chat template; loosen `_extract_json_block` in `Scenario_Representation_Extraction.py`. |
