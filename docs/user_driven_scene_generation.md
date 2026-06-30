# User-driven scene generation — design brainstorm

Goal: let a user **describe a scene in free text** (e.g. *"crowded intersection with an ambulance
coming quickly trying to make it through and cars moving to the side"*) and get an animated BEV,
then convert it through the UniScene-style **OccDiT** to occupancy and eventually to **LiDAR** and
(optionally) **camera** outputs.

Requirements captured from the brief:
- free-text driven, **no sketch** required (unlike the SAFE crash pipeline);
- **varying levels of detail** (terse prompts auto-filled; detailed prompts honored);
- **varying number of items** in the scene;
- no need to style car colors / textures (an ambulance doesn't need to be textured), **but objects
  should have approximately correct dimensions in BEV and occ**.

---

## Pipeline at a glance

```
free text ──▶ scene spec (DSL+) ──▶ multi-agent BEV clip (ego-centric, 18×400×400)
   (LLM)         (expander)              (general simulator)
                                              │
                                    ┌─────────┴─── OccDiT ──▶ occ (400×400×32) ──▶ LiDAR
                                    │                                         └──▶ camera (optional)
                                 preview (dashboard) ◀── cheap; iterate here before the expensive tail
```

**The encouraging part: the middle bridge already exists.** `Framework/ADS_Testing/BEV_Synthesis/
uniscene_export.py` already turns a SAFE scene into the per-token `(18,400,400)` ego-centric BEV the
UniScene loader expects, with box records. And from the compatibility audit, **OccDiT only reads a
coarse conditioning** — merged-static drivable (ch 1) + dynamic agents (ch 8–13); it ignores
lanes/dividers (15–17). That lowers the bar on the front half a lot: you don't need HD-map geometry,
you need a **correct drivable-area mask + correctly-sized agent footprints in the right channels.**

So the genuinely *new* work is concentrated in two places: **(1) text → scene** and **(2) a general
multi-agent simulator** to replace SAFE's 2-car crash logic.

---

## What's reusable today

- `bev_from_dsl.build_scene / simulate / _stage_collision / _latch_impact` — scene construction +
  kinematic rollout (currently crash-shaped).
- `uniscene_bev._rasterize_ego` — re-rasterize the world in any chosen vehicle's frame → `(18,H,W)`.
- `uniscene_export.py` — the BEV→occ *input* bridge: `(18,400,400)` int8 ego-centric tokens at
  0.25 m/px, divider channels remapped to UniScene generator-truth, plus `gt_boxes` records.
- The dashboard — the natural **preview surface** for the describe→inspect→refine loop.

## New components to build

1. **A richer scene schema.** SAFE's DSL is crash-shaped (2 actors, `W2E/E2W`, a `Conflict` block).
   A curated scene needs a **variable-length agent list** (class, dims, lane/pose, behavior, speed,
   *relationships*), a road archetype with params, an env block, and an explicit **ego choice**
   (which agent the sensors ride on). It should be **hierarchical / optional** so a terse prompt is
   valid (rest sampled) *and* a fully-specified scene is valid — that is the "levels of detail" knob.

2. **Text → scene (LLM, no sketch).** The same Qwen3-VL we serve, text-only, emitting the schema.
   It resolves intent: *"coming quickly"* → high speed; *"trying to make it through"* → straight
   path, ignores the signal; *"cars moving to the side"* → a `yield_pull_aside` behavior **keyed to
   the ambulance**; *"crowded"* → a density level the expander samples into a count. Prior art worth
   borrowing from: **Scenic** (probabilistic scene DSL), **LCTGen / language-conditioned traffic
   generation** for the spec + sampling.

3. **A general multi-agent simulator.** The real engineering. SAFE's `_step` knows ~5 canned
   maneuvers + the crash-intercept. A crowd needs: a small **behavior library** (cruise /
   follow-lane / yield-pull-aside / stop-at-line / turn / run-through), a **placement solver**
   (N agents in lanes, both directions, no overlaps, on drivable area), and **per-class
   rasterization into the correct dynamic channel** (pedestrian→ch 10, cone→ch 11, barrier→ch 12 —
   today SAFE only draws vehicles into ch 8). Collisions become optional, not the point.

---

## Concerns (the parts to weigh before committing)

1. **Dimension fidelity is split — you control footprint, not height.** OccDiT is 2D→3D: it lifts
   the BEV footprint to occ with a *learned, per-class height*. So an ambulance's L×W comes from the
   box we draw (controllable), but its **height in occ is a generic vehicle's** (~1.5 m), not
   ~2.4 m. (`uniscene_export.VEH_HEIGHT` already notes height "feeds aux/height only, not the occ
   model.") So "approximately correct dimensions" is honest for **footprint**, soft for **height**.

2. **Generative vs procedural occ — a real fork.**
   - *Generative (the stated goal): BEV → OccDiT → occ.* Realistic road/background occ and
     scene-completion for free, but dims/heights are **model-approximate** and it can ignore
     conditioning on rare layouts.
   - *Procedural: voxelize the 3D scene → occ directly.* **Exact** dimensions, fully controllable,
     but you build all the occ yourself (road + agents + background) and it looks less "nuPlan-real"
     to the learned lidar/camera renderers.
   - *Hybrid:* OccDiT for road/background, then **stamp exact agent voxels**. Best of both, fiddlier.
   - Instinct: start generative (the stated goal), accept approximate dims, keep procedural-occ in
     reserve for scenarios that need exactness.

3. **Out-of-distribution is the deepest risk.** OccDiT, the lidar renderer, and the camera model are
   all **nuPlan-trained**. *"Ambulance blasting through while cars pull onto the shoulder"* is rare
   to absent in nuPlan. The BEV layout places everything correctly (just rasterization), but the
   **generative stages may smooth away or hallucinate** around the unusual configuration. The more
   dramatic/curated the scene, the more this bites — inherent to leaning on pretrained generators.

4. **Taxonomy gap.** nuPlan classes are `vehicle / bicycle / pedestrian / traffic_cone / barrier /
   czone_sign`. **Ambulance = "vehicle"** to every downstream model — no emergency semantics, no
   lightbar. Acceptable per the brief (dims + position only), just flagged so there are no surprises.

5. **The ego must be a *plausible* ego.** Everything is ego-centric; models expect ego on the road,
   heading along a lane, at center. If the user makes the ego a car *pulled onto the shoulder
   yielding*, that's a weird sensor pose → OOD occ/lidar. Allow any agent as ego, but add a
   "sane ego pose?" guard or default the ego to a normally-driving agent (or the ambulance).

6. **No GT occ at inference — confirm the entry point.** The audited loader
   (`Nuplan_Occ_bev_HR_mini`) pairs BEV **with a ground-truth occ** — that's the *training/eval*
   path. A user-invented scene has no GT occ, so OccDiT must run in **pure inference/sampling** mode
   (condition on BEV, generate occ). Trace the actual OccDiT sampling script and confirm it accepts a
   bare BEV condition. (Related: `build_ciren_uniscene_index.py` smells like *dataset-building*,
   which is a different goal than *generation* — worth clarifying which is intended.)

7. **Temporal stride / clip length.** The loader groups **5 tokens per clip**. *"Coming quickly"*
   over 5 frames at 0.25 m/px needs the per-frame displacement to match the **dt the models were
   trained at** — too fast and frames tunnel; too slow and "quickly" isn't conveyed. Pin the model's
   expected token Hz and design the sim dt to it.

8. **LiDAR has a cheap, exact option.** occ→lidar can be a **geometric raycast** of the voxel grid
   from the ego origin — deterministic, no model, and **preserves dimensions exactly** (occ already
   encodes shape). That sidesteps OOD for lidar entirely. A learned occ→lidar buys realism
   (intensity, dropout) you may not need. Camera is the opposite — either the learned video model
   (photoreal-ish, OOD-risky, needs the nuPlan cam rig extrinsics) or a non-photoreal geometric voxel
   render (no textures, exact geometry — which matches the "no textures" constraint). Given the
   constraints, **geometric raycast lidar + geometric occ-render camera** may be the most controllable
   path, with the learned renderers as the "make it pretty" upgrade.

9. **Validation / plausibility loop.** Crowds need overlap checks, lane-snapping, and drivable-area
   containment — SAFE barely enforces this with 2 agents. And with no sketch to disambiguate, a
   **describe → preview the BEV animation → tweak counts/positions → then commit to the expensive
   occ/lidar** loop is close to mandatory. The dashboard is the natural preview surface and keeps the
   diffusion cost off the critical path until the layout is liked.

---

## Recommendation & open questions

Net: the BEV-input bridge is solved; the front (text→scene→general sim) is a well-bounded build; the
tail (occ→lidar/camera) is mostly "wire up UniScene's renderers + decide generative-vs-geometric."
The dimension promise is honest for **footprint**, soft for **height** (OccDiT is 2D→3D).

Forks to decide:
- **Generation vs dataset-building?** Render scenes once (inference) or build labeled BEV+occ pairs
  (training/eval)? Changes whether procedural occ is needed at all.
- **Occ: generative (OccDiT, approx dims) / procedural (exact dims) / hybrid?**
- **Behavior realism:** scripted primitives (fast, controllable, OOD-ish) vs a light traffic model
  (IDM-ish car-following + yielding)?
- **LiDAR/camera: geometric (exact, untextured, no OOD) or learned (realistic, OOD-risky)?** The
  "no textures" leaning nudges toward geometric.
- **Ego policy:** always a sane driving agent, or truly any agent (with a plausibility guard)?

Useful de-risking next step (read-only): trace the **OccDiT sampling entry** and the
**occ→lidar / occ→camera** paths in `uniscene_stack`, and produce a "here's exactly where each stage
plugs in + the OOD/looks-real risk per stage" map before any code is written.
