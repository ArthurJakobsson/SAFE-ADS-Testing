# Conveying more through BEV → occ: enriching the OccDiT conditioning

We keep BEV → occ as the core conversion — BEV is a universal, ego-centric raster every consumer
understands — but the *current* conditioning is impoverished, so a lot of information is hallucinated
by OccDiT's learned prior. This doc collects ideas to convey more: richer conditioning, DiT design
changes to consume it, how to train it from labels we already have, and an optional richer BEV
format that stays backward-compatible.

## What OccDiT sees today, and what it loses

Today OccDiT reads only **merged-static drivable (ch 0–7 → ch 1) + dynamic agent channels 8–13** —
seven binary 2D planes at 0.25 m/px. Everything else in our `(18,400,400)` BEV (the per-class static
detail, the divider/line layers 15–17) is **discarded**, and the model is asked to invent 3D from a
flat mask. So these are hallucinated rather than conveyed:

| Lost dimension | Why it matters | Label we already have to supply it |
|---|---|---|
| **Object height / elevation** | occ is 3D; an ambulance (~2.4 m) vs a sedan (~1.5 m) is invisible to the model | `gt_boxes[:, h]`, box z-center |
| **Precise orientation** | a near-square footprint is ±180° ambiguous; head-on vs side-impact geometry is the whole point | `gt_boxes[:, yaw]` |
| **Instance identity / separation** | all vehicles OR into one ch-8 blob; two touching cars (every crash impact frame) merge | per-agent corners in `_rasterize_ego` |
| **Velocity / motion** | needed for temporal coherence across the 5-frame clip | `gt_velocity_3d` |
| **Fine class / size** | `nuscenes_name()` squashes everything to `car`; size lost | DSL `Model` + nuPlan box size |
| **Map structure** | curbs, crosswalks, lanes, intersection extent collapsed into one drivable blob | the un-merged ch 0–7 + dividers 15–17 (already rasterized!) |

Crucially, **all of this is recoverable from the same labels that already produce the GT occ** — so a
richer condition costs no new annotation; it's self-supervised by construction.

---

## 1. Enrich the conditioning (still a HxW raster)

Additive channels that keep the BEV a raster but carry more signal. All are stamped per-agent with the
**same corner polygons already used for ch 8**, so they're pixel-aligned by construction.

- **Continuous height + z-elevation planes.** Per pixel under a footprint, paint the agent's box
  height (m) and z-center. The single biggest fix for the 2D→3D gap — gives the model the dimension it
  currently invents. *(Caveat: a 2D height plane encodes only a top surface, not full 3D — no
  overpasses/multi-level.)*
- **2.5D layered occupancy stack.** Instead of (or alongside) one flat mask, splat each footprint into
  4–8 **height bands** (e.g. 0–0.5, 0.5–1.5, 1.5–3, >3 m) from its z and h → a `(Z_lo,400,400)` coarse
  3D condition. A tall truck lights up more bands than a sedan; the strongest "make height honest"
  encoding short of full voxels.
- **Orientation as sin/cos.** Two float channels `sin(yaw)`, `cos(yaw)` stamped in each footprint
  (avoids angle wraparound). Disambiguates heading the binary blob can't.
- **Velocity `vx, vy`.** Per-footprint flow channels — anchors temporal consistency for the clip.
- **Instance separation.** Either an instance-id channel (1..N, ego = 1) or an *offset-to-center*
  `(Δx,Δy)` channel (panoptic-deeplab style). Keeps two adjacent/overlapping vehicles as two instances.
- **Fine class + size.** One-hot base class + a scalar size bucket stamped in the footprint, so the
  model can render a "vehicle, large" differently from "vehicle, small".
- **Un-merged per-class static map + SDF dividers.** Keep ch 0–7 separate (intersections, crosswalks,
  walkways, carpark, lanes, road segments) + dividers 15–17, and add a **signed-distance-to-divider**
  plane so 1–2 px lines carry usable gradient. Recovers curbs/markings/lane structure the merge
  destroys.

**Design principle:** every new channel is *optional and additive*; the canonical 18-ch mask must remain
recoverable from the rich tensor (see §4).

---

## 2. DiT design changes to consume it

Ordered roughly low-risk → high-capability.

- **Zero-init channel-expanded patch-embed (recommended first).** Widen the conditioning conv from 7 to
  `7+K` planes; copy pretrained weights into `[:, :7]`, **zero-init the new channels**. At fine-tune
  step 0 the model is bit-identical to the checkpoint, then the extra planes learn an additive
  correction. Smallest change; preserves the prior exactly.
- **ControlNet side-branch over the rich BEV (recommended for capacity).** A trainable copy of the
  first N OccDiT blocks consumes the *full* rich condition; the **original trunk stays frozen** and
  keeps eating the canonical 7-plane coarse condition. The branch injects features back through
  zero-init connectors. Maximally backward-compatible (frozen trunk = the universal pathway is
  untouched; missing/corrupt rich condition degrades gracefully), and higher capacity than a widened
  stem. Cost: ~30–50% more compute.
- **Per-instance attribute tokens via gated cross-attention (GLIGEN / LayoutDiffusion style).** Encode
  each agent as a *token* — `MLP([Fourier(x,y,z), Fourier(w,l,h), sin/cos yaw, vel, onehot(class)])` —
  and let patch tokens cross-attend to the box-token set, with a **zero-init output gate** per block.
  This is the natural carrier for everything that *doesn't rasterize well*: exact size/height,
  continuous yaw, instance identity, velocity, fine class. Most invasive but most powerful; pair with a
  box-occupancy loss so token `(x,y)` grounds to the footprint.
- **Global scene / text embedding cross-attention.** Encode a clip-level string from the upstream
  artifacts we already store (SAFE DSL actions/speeds, the `Conflict` record `{at_fault, impact_type,
  point_of_impact}`, `road_type` — or the user's free-text description) with a frozen text encoder into
  a few global tokens; cross-attend the denoiser. Steers the prior toward the right *global*
  configuration (crash type, urban vs highway) cheaply; coarse, so pair with instance tokens for
  placement.
- **2.5D / coarse-voxel condition** (consume the layered stack from §1) instead of pure 2D — gives the
  denoiser real 3D structure to anchor on.
- **Temporal conditioning** — feed the velocity channels and/or the BEV *sequence* so occ is coherent
  across the 5 clip frames.
- **Output-head options** — extend the occ class set (finer taxonomy / map-ground classes), or add an
  **instance/panoptic occ head** and a **flow/velocity occ** head, depending on how much downstream
  cares.

---

## 3. Training it from the labels/semantics we already have

- **Free supervised pairs.** Build every rich channel/token from the *same* labels that produce the GT
  occ — no new annotation. The exporter already emits `gt_boxes`, velocities, names, and all 18 raster
  channels; the rich condition is a re-stamp of those.
- **Conditioning dropout → varying levels of detail + controllability.** Define a **nested LOD stack**
  (L0 = canonical 18-ch; L1 = +height; L2 = +instance; L3 = +yaw; L4 = +fine class) and randomly drop
  whole levels per sample to a learned null token. Delivers the project's "varying levels of detail"
  goal directly: a caller with only the universal BEV still gets valid occ; a caller with full labels
  gets faithful dims. Enables **classifier-free guidance** to dial how hard the model obeys the rich
  condition. Keep the schedule nested (no L3 without L1).
- **Preserve the prior.** Finetune with zero-init added channels, or train *only* the ControlNet
  branch + connectors — both start as a no-op on the pretrained checkpoint.
- **Auxiliary consistency losses that force the model to *use* the condition** (the fix for "OccDiT
  ignores the condition / dims only approximate"):
  - **Footprint-IoU** (cheapest, zero new labels): `1 − softIoU(maxpool_z(softmax(occ)[vehicle]),
    bev[8])` + the same for drivable. Makes the 2D→3D map provably honor the input mask.
  - **Height-consistency**: occ column height under each footprint matches the conditioned height.
  - **Orientation-consistency**: angle of the predicted occ footprint's major axis ≈ conditioned yaw
    (weight by aspect ratio; skip near-square/overlapping boxes).
  - **Instance/panoptic separation**: per-voxel embedding pull-same-id / push-different-id, Hungarian-
    matched to GT footprints (permutation-invariant).
- **Fix the *target*, not just the condition.** The consistency losses are only as good as the GT occ
  they regress toward. Today's GT extrusion is a flat road slab + constant `veh_z` for every vehicle.
  **Upgrade `build_ciren_uniscene_index.extrude_occ` to oriented-box voxelization** at each box's true
  `xyz/size/yaw`, so the supervision itself is dimension-faithful (otherwise no model can learn correct
  height).
- **Taxonomy from existing labels.** Supervise a coarse base-class + size-bin head from nuPlan classes;
  decompose out-of-taxonomy strings — **`ambulance` → `vehicle` + `size=large`** + a free attribute
  embedding — which matches the "dimensions matter, textures don't" constraint.
- **Map fidelity.** Supervise a few **ground classes** (road / crosswalk / walkway / boundary) in the
  lowest occ z-bins from the un-merged static planes, recovering map detail the merge discards.
- **Evaluation.** Track occ IoU + the new fidelity metrics: footprint IoU, height error, instance
  recall, orientation error, class accuracy. These are what tell us the richer condition actually moved
  the needle.

---

## 4. An optional richer BEV format (additive, backward-compatible)

Keep `gt_bev_masks (18,400,400) int8` as the universal interface; layer extras *additively* so a
base-only loader is unaffected and the enriched DiT uses them when present.

- **`gt_bev_rich (C',400,400) float16`** — a second key in the *same* `<token>.npz` with the continuous
  per-pixel quantities (height, z, sin/cos yaw, vx, vy, fine-class, instance-id). Mostly-zero → compresses
  well. Stamped with the same corner polygons as ch 8 (pixel-perfect alignment).
- **Per-instance sidecar `<token>_inst.npz`** — the `(N,11)` table the raster can't express
  `[x,y,z,w,l,h,yaw,vx,vy,fine_class,instance_id]`, with **instance_id stable across the clip's 5
  frames**. Feeds the cross-attention tokens; per-token locality suits sharded/streaming loaders.
- **Schema/version envelope** — a tiny `schema` key in every npz
  (`{format, version, base_channels:18, rich_channels:[...], px_per_m:4.0, ego_centric:true, has_rich,
  level_of_detail}`) so base and enriched consumers negotiate gracefully; bump `version` on any layout
  change.
- **Strict-superset discipline** — the rich tensor is the single source of truth; define one canonical
  `to_base(rich) → (18,400,400) int8` reducer and **assert `to_base(rich) == gt_bev_masks` in the
  self-test**. Keeps "BEV is universal" literally true — the canonical mask can never silently diverge.
- **Real dims, not placeholders** — thread true box `(w,l,h)` + z from labels (or a class→size prior
  table) into the height/z planes and the sidecar, replacing the hard-coded `VEH_HEIGHT=1.6` /
  `GROUND_Z=0`. Add a `dims_source` field (`gt` | `class_prior`).
- **Optional scene-text key** — a clip-level `text` string (+ optional precomputed embedding) for the
  global cross-attention; stored raw so the exporter stays torch-free.
- **One ergonomic knob** — a `uniscene_export.py --lod {0,1,2,3}` flag behind a single `RichBevWriter`
  (the one writer of truth, so all keys stay consistent). **LOD 0 is bit-identical to today.** Sharded
  dir layout (`<bev_dir>/<token>.npz`, `<bev_dir>/inst/<token>_inst.npz`) so a base loader globbing the
  top level never sees the extras.

---

## Recommended phasing

1. **Phase 1 — cheap, high value.** Add the **continuous height + un-merged static** channels; widen the
   patch-embed with **zero-init**; finetune with a **footprint-IoU + height-consistency** loss and
   **conditioning dropout**. Save them via `--lod 1` (`gt_bev_rich` + `schema`, strict-superset). This
   alone fixes the two loudest gaps (height, map detail) with minimal risk.
2. **Phase 2 — instance & semantics.** Add the **per-instance sidecar** + **gated cross-attention
   tokens** (exact size/yaw/instance/velocity/fine-class) via a **ControlNet branch** (frozen trunk).
   Add instance/orientation losses. Upgrade the **GT occ extrusion to oriented-box voxelization** so the
   target is faithful.
3. **Phase 3 — global & temporal.** Global **scene/text** cross-attention (from the LLM description),
   **2.5D layered** condition, and **temporal** conditioning for clip coherence.

## Concerns / trade-offs

- **You can't feed a richer BEV to a frozen model** — every idea here needs finetuning/branch-training.
  The zero-init / ControlNet paths keep that low-risk (start as a no-op on the checkpoint).
- **The model may still ignore extra channels** without the consistency losses + dropout; those losses
  are what *force* usage.
- **Channels vs tokens.** Dense rasters carry spatial layout well but explode in count and waste capacity
  on mostly-zero planes; per-instance/text info belongs in **tokens** (cross-attn). Use both for what
  each is good at.
- **2D→3D is still a bottleneck** even with a height plane (single top surface, no multi-level); the 2.5D
  layered condition mitigates but doesn't fully close it.
- **GT-target fidelity caps everything** — consistency losses are only as good as the occ they regress
  toward; the oriented-box voxelization upgrade is a prerequisite for "correct dimensions".
- **OOD risk cuts both ways** — more conditioning can improve faithfulness *or* conflict with the learned
  prior on rare layouts; CFG strength is the knob to trade obedience vs realism.
- **Backward-compat discipline** — the strict-superset reducer + schema/version are what keep "BEV
  universal" honest as the format grows; they need enforcing in the self-test.
