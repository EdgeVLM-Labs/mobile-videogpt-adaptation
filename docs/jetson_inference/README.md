# Jetson Inference — Documentation Index

Documentation for **Mobile-VideoGPT** deployed on **NVIDIA Jetson Orin Nano Super**
(8 GB unified memory) for real-time exercise feedback.

---

## 🚀 Start here

- **[JETSON_OPTIMIZATION_JOURNEY.md](./JETSON_OPTIMIZATION_JOURNEY.md)** —
  the main story. Eight stages from "doesn't run / OOM crash" to
  TTFT 2.3 s with reliable real-time feedback. This is the canonical
  reference for what was done, why, and the measured outcomes.

---

## 🛠️ Setting up & operating

- **[INSTALL_NOTES.md](./INSTALL_NOTES.md)** — why each step in
  `setup_jetson.sh` exists (cuSPARSELt, NVIDIA PyTorch wheel,
  torchvision / decord / causal-conv1d / mamba-ssm built from source,
  the `pypi.jetson-ai-lab.dev` DNS quirk, etc.). Read this if a future
  install fails or you need to explain the install in writing.

- **[HEADLESS_DEMO_SETUP.md](./HEADLESS_DEMO_SETUP.md)** — set up SSH,
  switch the Jetson to headless mode, launch the Gradio inference
  server, revert to GUI if needed. Read this before the first demo.

- **[POWER_MEASUREMENT.md](./POWER_MEASUREMENT.md)** — capture
  `tegrastats` during inference and compute average / peak board
  power. Self-contained step-by-step (designed to read on operator
  laptop while SSH'd in).

---

## 🚦 Input gating (don't run the VLM when nobody's exercising)

- **[TIER2_EXERCISE_GATE_DESIGN.md](./TIER2_EXERCISE_GATE_DESIGN.md)** — design
  for the Tier 2 exercise-relevance gate (rejects untrained movements). The
  Tier 1 motion gate (skip on an empty/idle scene) is already shipped — see
  JOURNEY § Stage 9.

---

## 📝 Paper / reviewer support

- **[REVIEWER_RESPONSE.md](./REVIEWER_RESPONSE.md)** — targeted
  answers to the IEEE AIIoT reviewers' Jetson-related feedback:
  inference latency, peak memory, power consumption, quantization
  stance, and per-watt / per-parameter efficiency.

---

## 📦 Historical / supporting

- **[jetson_inference_fixes.md](./jetson_inference_fixes.md)** — the
  original Phase-1 fix-by-fix notes. **Superseded** by the Journey
  doc above; kept for archive.

- **[stage_logs/](./stage_logs/)** — raw inference logs captured at each
  optimization stage (`logs_stage_1.txt` through `logs_stage_4.txt`)
  plus a sample failure log (`error_inf.txt`).

- **[screenshots/](./screenshots/)** — demo screenshots.

---

## 🎯 Quick answers

| Question | Where |
|---|---|
| What is the current TTFT? | JOURNEY § Summary Table — **2.3 s** |
| How do I launch a demo? | `./run_demo.sh` (preflight + launch) — or HEADLESS_DEMO_SETUP.md → "Running the Demo" |
| How do I just verify the rig is demo-ready? | `./run_demo.sh --check` |
| How does the demo UI work? | HEADLESS_DEMO_SETUP.md → "Demo UI Walkthrough" |
| How does voice feedback work? | HEADLESS_DEMO_SETUP.md → "Voice feedback" — runs **client-side** in the browser |
| How do I measure power? | POWER_MEASUREMENT.md |
| Why does install need so many manual steps? | INSTALL_NOTES.md — explains each |
| What's the recommended invocation? | JOURNEY § How to Run → `USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py` |
| Did we use quantization? | REVIEWER_RESPONSE.md § 4 — **No, by design** |
| Why does each poll cover only the last 4 seconds? | JOURNEY § Stage 8 (Real-Time Frame Buffering) |
| How do I stop it reacting to an empty room? | JOURNEY § Stage 9 — launch with `MOTION_GATE=1` |
| How do I make it ignore untrained exercises? | TIER2_EXERCISE_GATE_DESIGN.md (designed, not yet built) |
| What were the problems we hit and fixed? | JOURNEY § Stages 1–9 |
