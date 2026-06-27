# Mobile-VideoGPT Coach — Web App

Patient-first web UI for the real-time exercise coach. Reuses the existing Python
inference engine **unchanged** — this is only a presentation layer.

- **Backend:** `server.py` (FastAPI) wraps `PollingInferenceEngine`.
- **Frontend:** `static/` (vanilla HTML/CSS/JS, no build step).
- **Camera** stays on the Jetson; the **preview is streamed** to the viewing device
  (tablet/laptop) as MJPEG, and **feedback** is pushed via Server-Sent Events.
- UI framework adds milliseconds only — all latency is the model, unchanged.

## Run (on the Jetson)

```bash
# from the repo root, in the model env (the one that runs the Gradio app)
pip install "fastapi" "uvicorn[standard]"     # usually already present via Gradio
bash polling/webapp/run.sh                     # serves on 0.0.0.0:8000
```

`run.sh` sets the GPU fast-path env vars (`USE_FULL_GPU=1 USE_TRT_CLIP=1`). **Do not
launch with a bare `python -m polling.webapp.server`** — without those vars CLIP falls
back to ONNX-Runtime on CPU (~3.8s/poll) and TTFT jumps from ~2.5s to ~6s. (Equivalent
manual launch: `USE_FULL_GPU=1 USE_TRT_CLIP=1 python -m polling.webapp.server`.)

Requires the TensorRT CLIP engine at `models/tensorrt/clip_vit_base_fp16.engine`
(or `_fp32`). If missing, build it once on the Jetson: `bash scripts/build_clip_tensorrt.sh`.

Then open **`http://<jetson-ip>:8000`** from any device on the same network.

## Usage
- **Patient view (default):** big live video + one feedback card + Start/Stop + Voice.
- **Advanced (gear icon):** camera/source, polling interval, sample fps, max tokens,
  base model, **LoRA weights**, prompt, warmup, naturalizer.

## Endpoints
| Route | Purpose |
|---|---|
| `GET /` | patient UI |
| `GET /api/preview.mjpg` | live camera preview (MJPEG) |
| `GET /api/stream` | feedback stream (SSE) |
| `POST /api/start` / `POST /api/stop` | session control |
| `GET /api/cameras` · `/api/sample_videos` · `/api/config` · `/api/status` | options/state |

## Notes / TODO
- **LoRA weights** default to the V2 adapter (`mobile-videogpt-finetune-v2-mixed`) in
  `polling/config.py` (`lora_weights_path`); override per-session in the Advanced drawer.
  The repo is **private** — the Jetson must be authenticated (`hf auth login` / `HF_TOKEN`)
  or the load fails (the engine now refuses to fall back to the base captioner).
- **Preview is decoupled from the model input.** Capture runs at `config.fps` (15 →
  smooth preview); each inference uniformly samples `num_frames` across the last
  `inference_window_seconds` (4 s) of buffer, so the model still sees 16 frames over a
  4 s window (the 4 fps training density) regardless of capture rate. Raise/lower
  `fps` purely for preview smoothness.
- "No person in frame" is still handled by the model (occasionally wrong); the optional
  person-detection gate discussed separately would make it deterministic.
