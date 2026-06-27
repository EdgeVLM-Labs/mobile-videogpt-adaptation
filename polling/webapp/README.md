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
python -m polling.webapp.server               # serves on 0.0.0.0:8000
```

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
- **Set the V2 LoRA weights** in `polling/config.py` (`lora_weights_path`) or in the
  Advanced drawer — it still defaults to the old `mobile-videogpt-finetune-2000`.
- **Preview is decoupled from the model input.** Capture runs at `config.fps` (15 →
  smooth preview); each inference uniformly samples `num_frames` across the last
  `inference_window_seconds` (4 s) of buffer, so the model still sees 16 frames over a
  4 s window (the 4 fps training density) regardless of capture rate. Raise/lower
  `fps` purely for preview smoothness.
- "No person in frame" is still handled by the model (occasionally wrong); the optional
  person-detection gate discussed separately would make it deterministic.
