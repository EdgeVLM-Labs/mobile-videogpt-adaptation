# Tier 2 — Exercise-Relevance Gate (Design)

> **Status: design only — not yet implemented.** Tier 1 (the motion gate)
> ships in code today; this document specifies the next gate so it can be
> built and calibrated when there's time before/after the demo.
>
> Companion to the motion gate. Read the [JOURNEY § Input Gating](./JETSON_OPTIMIZATION_JOURNEY.md)
> section first for the problem framing.

---

## 1. Problem this solves

The motion gate (Tier 1) answers *"is anything happening?"* — it suppresses
inference on an empty/idle scene. It does **not** answer *"is the person doing
an exercise the model was trained on?"*. A reviewer doing jumping jacks (an
untrained movement) is *moving*, so the motion gate passes and the VLM still
produces confident-but-wrong coaching.

**Tier 2 gate goal:** before running the VLM, decide whether the current frames
depict a **known / trained exercise**. If not, show *"No supported exercise
detected"* and skip generation — the same suppress-and-skip pattern as the
motion gate, one rung higher in selectivity.

| Failure case | Motion gate (Tier 1) | + Exercise gate (Tier 2) |
|---|---|---|
| Empty room / idle | ✅ skips | ✅ skips |
| Moving, but untrained exercise | ❌ still runs VLM | ✅ skips |
| Correct trained exercise | ✅ runs | ✅ runs |
| **Isometric hold (plank, wall-sit)** | ❌ **wrongly gated as "idle"** (no motion) | ✅ runs (posture matches) |

> **Why Tier 2 also fixes static holds:** the motion gate keys on *movement*, so
> a held plank reads as "no activity" and is wrongly skipped. Prototype matching
> keys on the *posture* embedding, which is distinctive for a plank regardless of
> motion — so it admits static-hold exercises correctly. This is a second
> independent reason to build Tier 2, beyond rejecting untrained movements.

---

## 2. Key technical constraint (read before designing)

The deployed CLIP encoder does **not** produce CLIP's joint text-image
embedding. The ONNX/TensorRT engine bakes in `layer=-2` and returns
**penultimate-layer patch features** (`select_feature="patch"`), consumed by the
VLM in [`arch.py`](../../mobilevideogpt/model/arch.py) at
`encode_videos_by_seletive_frames()` (≈ line 171):

```python
context_image_features = self.get_model().get_image_vision_tower()(
    context_images, select_feature="patch")   # (b·t, l, d)  penultimate patches
```

Two consequences:

1. **You cannot do classic CLIP zero-shot text↔image cosine** against these
   features — they're not in the contrastive projection space. A text-prompt
   gate ("a person doing squats") would require a *separate* standard CLIP
   forward (final layer + visual projection) **plus** the CLIP text tower
   (not currently exported). That is extra cost and extra assets.
2. **What you already have** is a strong per-frame visual embedding. Mean-pool
   the patch tokens → one vector per frame → average over the window → one
   embedding per poll. This is reusable and free, and it's what the gate
   should be built on.

**Decision: use image-embedding prototypes, not text zero-shot.** It reuses the
exact features the VLM computes, needs no new model, and is more accurate for a
small fixed exercise set than text prompts.

---

## 3. Approach — prototype (centroid) matching

**Offline (once):**
1. Collect a handful of short reference clips per trained exercise
   (e.g. 5–10 clips each of squat, push-up, lunge, …) plus a **negative**
   set (standing, sitting, walking, empty room, untrained movements).
2. Run each through the *same* image vision tower the VLM uses, mean-pool to a
   single L2-normalized vector, and average per class → one **centroid** per
   known exercise. Save to `models/gates/exercise_centroids.npz`.

**Online (per poll, before the VLM):**
1. Encode the current context frames with the image tower → pooled,
   L2-normalized embedding `e`.
2. `sim = max_k cosine(e, centroid_k)` over known-exercise centroids.
3. **Gate:** if `sim < tau_known` **or** `(sim - best_negative_sim) < margin`
   → "no supported exercise" → skip VLM. Otherwise run normally.

Two thresholds (`tau_known`, `margin`) are calibrated on a held-out set
(§5). The margin term rejects frames that are vaguely close to an exercise but
closer to a negative (e.g. standing still between reps).

---

## 4. Two implementation options (latency trade-off)

The gate must run **before** the VLM, but the embedding is normally computed
*inside* the VLM forward. How you handle that decides the added latency
(numbers consistent with the [JOURNEY latency analysis](./JETSON_OPTIMIZATION_JOURNEY.md)):

### Option A — standalone gate encode (simple, +~0.2 s on TRT)
- In the polling loop, run **one** image-tower forward on the context frames
  purely for the gate. If it passes, call the VLM normally (which re-encodes).
- Cost: one extra CLIP encode per *passing* poll ≈ **+0.2 s** (TRT GPU) /
  **+3–5 s** (ORT CPU — avoid on the CPU path). On a *gated* poll you pay only
  this and skip the ~5–10 s VLM.
- **No VLM surgery.** Recommended for first implementation and the demo.

### Option B — reuse the embedding (~free, more invasive)
- Factor context-image encoding out of `encode_videos_by_seletive_frames()`,
  compute it once, gate on it, then pass the cached features into the VLM
  forward so it is **not** recomputed.
- Cost on a passing poll ≈ **tens of ms** (just the cosine math); keeps TTFT at
  ~2.3 s. Requires threading a `cached_context_features` kwarg through
  `prepare_inputs_labels_for_multimodal` / the generate path.
- Recommended as the optimization once Option A is validated.

| | Gate passes (real exercise) | Gate blocks | TTFT impact |
|---|---|---|---|
| Option A (TRT) | +~0.2 s | pay ~0.2 s, skip ~5–10 s | 2.3 → ~2.5 s |
| Option B (reuse) | +tens of ms | pay ~0 extra, skip ~5–10 s | unchanged |

---

## 5. Calibration & validation (do not ship un-tuned)

Thresholds are camera/lighting/exercise dependent — pick them from data, not by
guessing:

1. Build a labeled eval set: positives (each trained exercise) + negatives
   (idle, untrained movements, empty room).
2. Sweep `tau_known` / `margin`; plot **false-accept rate** (untrained passed as
   known) vs **false-reject rate** (real exercise wrongly skipped).
3. Pick the operating point that prioritizes **low false-reject** (never drop a
   real rep mid-demo) while cutting most false-accepts. Log `sim` on every poll
   so the threshold can be re-tuned from real session logs.

**Demo-safety:** like the motion gate, ship it **off by default** behind a flag
(suggested `EXERCISE_GATE=1`, `EXERCISE_TAU`, `EXERCISE_MARGIN`) and stack it on
top of the motion gate. With it off, the pipeline behaves exactly as today.

---

## 6. Proposed wiring (mirrors the motion gate)

- **Config:** add `enable_exercise_gate`, `exercise_tau`, `exercise_margin`,
  `exercise_centroids_path` to [`PollingConfig`](../../polling/config.py),
  env-driven and off by default (same pattern as `enable_motion_gate`).
- **Helper:** an `ExerciseGate` class (e.g.
  `mobilevideogpt/model/multimodal_encoder/exercise_gate.py`) that loads the
  centroids and exposes `score(pooled_embedding) -> (is_known, sim)`.
- **Loop hook:** in both [`gradio_app.py`](../../polling/gradio_app.py) and
  [`inference_engine.py`](../../polling/inference_engine.py), place the check
  **right after the motion gate**, before `metrics.start_inference`. On a block,
  yield/log a *"No supported exercise detected"* state and `continue` — the
  exact same shape as the motion-gate skip.
- **Offline tool:** `scripts/build_exercise_centroids.py` to produce the
  `.npz` from reference clips.

---

## 7. Effort estimate

| Piece | Effort |
|---|---|
| Centroid builder + reference clip collection | ~half day (depends on clip gathering) |
| `ExerciseGate` + config + loop wiring (Option A) | ~half day |
| Threshold calibration on eval set | ~half day |
| Option B reuse refactor (optional, later) | ~1 day |

**Total for a working, calibrated Option-A gate: ~1–2 days.** Option B is a
follow-up optimization, not a prerequisite.
