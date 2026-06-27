#!/usr/bin/env python3
"""
FastAPI web app for Mobile-VideoGPT real-time exercise coaching.

Why this exists: the Gradio app exposes every hyperparameter on the main screen,
which is wrong for a patient-facing product. This server reuses the existing
Python inference engine UNCHANGED and adds:

  * a patient-first UI (big live video + one feedback card)            -> static/index.html
  * an Advanced drawer for clinicians/devs (all the knobs)
  * MJPEG camera preview streamed from the Jetson to a LAN device      -> /api/preview.mjpg
  * Server-Sent-Events feedback stream (one event per poll)            -> /api/stream

The model runs on the Jetson; the browser only renders. UI framework adds
milliseconds — all latency is the model, so this changes UX with no speed cost.

Run:
    pip install "fastapi" "uvicorn[standard]"     # already present via Gradio
    python -m polling.webapp.server               # serves on 0.0.0.0:8000
"""
import os
import sys
import time
import json
import queue
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# repo root on path so `polling.*` imports resolve when run as a module or script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
try:
    from utils.naturalizer.feedback_naturalizer import FeedbackNaturalizer
except Exception:
    FeedbackNaturalizer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("webapp")

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_DIR = Path(__file__).parent.parent.parent / "sample_videos"

PRAISE_MARKERS = ("good form", "great job", "no obvious issue", "good arm placement", "keep going")
NEG_LABEL = "No recognized exercise"


def classify(response: str) -> Dict:
    """Parse a raw model utterance into {state, exercise, feedback} for the UI."""
    raw = (response or "").strip()
    if not raw:
        return {"state": "none", "exercise": None, "feedback": "", "raw": raw}
    if raw.lower().startswith("no recognized exercise") or raw.lower() == NEG_LABEL.lower():
        return {"state": "no_exercise", "exercise": None, "feedback": NEG_LABEL, "raw": raw}
    if " - " in raw:
        exercise, feedback = raw.split(" - ", 1)
        exercise, feedback = exercise.strip(), feedback.strip()
    else:
        exercise, feedback = raw, ""
    state = "good" if any(m in feedback.lower() for m in PRAISE_MARKERS) else "correction"
    return {"state": state, "exercise": exercise, "feedback": feedback, "raw": raw}


def list_cameras() -> List[Dict]:
    """Enumerate cameras on the host (Jetson). Linux /dev/v4l/by-id first, else 0-5."""
    cams: List[Dict] = []
    try:
        by_id = Path("/dev/v4l/by-id/")
        if by_id.exists():
            for link in by_id.iterdir():
                if link.is_symlink() and "video-index0" in link.name:
                    num = int(link.resolve().name.replace("video", ""))
                    name = " ".join(link.name.replace("usb-", "").replace("_", " ")
                                    .split("-video-index")[0].split()).title()
                    cams.append({"name": f"{name} (video{num})", "index": num})
    except Exception as e:
        logger.warning(f"camera enumeration: {e}")
    if not cams:
        cams = [{"name": f"Camera {i}", "index": i} for i in range(4)]
    return cams


class StartRequest(BaseModel):
    is_file: bool = False
    source: str = "0"                       # camera index (webcam) or filename under sample_videos/
    polling_interval: Optional[float] = None
    fps: Optional[int] = None
    max_new_tokens: Optional[int] = None
    prompt: Optional[str] = None
    base_model_path: Optional[str] = None
    lora_weights_path: Optional[str] = None
    use_naturalizer: bool = False
    warmup_runs: int = 1


class Session:
    """Owns one engine + one polling thread; broadcasts feedback to SSE subscribers."""

    def __init__(self):
        self.engine: Optional[PollingInferenceEngine] = None
        self.naturalizer = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        self.subscribers: List[queue.Queue] = []
        self.sub_lock = threading.Lock()
        self.status: Dict = {"state": "idle", "message": "Idle", "polls": 0}
        self.last_feedback: Optional[Dict] = None

    # ---- pub/sub ----
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self.sub_lock:
            self.subscribers.append(q)
        q.put({"type": "status", **self.status})
        if self.last_feedback:
            q.put({"type": "feedback", **self.last_feedback})
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, event: Dict):
        if event.get("type") == "status":
            self.status = {k: v for k, v in event.items() if k != "type"}
        elif event.get("type") == "feedback":
            self.last_feedback = {k: v for k, v in event.items() if k != "type"}
        with self.sub_lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def _status(self, state, message, **extra):
        self.broadcast({"type": "status", "state": state, "message": message,
                        "polls": extra.get("polls", self.status.get("polls", 0)), **extra})

    # ---- preview ----
    def latest_frame_bgr(self) -> Optional[np.ndarray]:
        eng = self.engine
        if eng and eng.stream_handler and len(eng.stream_handler.frame_buffer) > 0:
            rgb = eng.stream_handler.frame_buffer[-1].frame
            try:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                return rgb
        return None

    # ---- lifecycle ----
    def start(self, req: StartRequest) -> Dict:
        with self.lock:
            if self.running:
                return {"ok": False, "message": "A session is already running."}
            self.running = True
        self.thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        self.thread.start()
        return {"ok": True}

    def stop(self) -> Dict:
        self.running = False
        return {"ok": True}

    def _build_config(self, req: StartRequest) -> PollingConfig:
        kw = {}
        if req.base_model_path:   kw["base_model_path"] = req.base_model_path
        if req.lora_weights_path: kw["lora_weights_path"] = req.lora_weights_path
        if req.polling_interval is not None: kw["polling_interval"] = req.polling_interval
        if req.fps is not None:              kw["fps"] = req.fps
        if req.max_new_tokens is not None:   kw["max_new_tokens"] = req.max_new_tokens
        if req.prompt:                       kw["prompt"] = req.prompt
        return PollingConfig(**kw)

    def _run(self, req: StartRequest):
        try:
            cfg = self._build_config(req)

            # (re)create engine only if the model identity changed
            if not (self.engine and getattr(self.engine, "_is_loaded", False)
                    and self.engine.config.base_model_path == cfg.base_model_path
                    and self.engine.config.lora_weights_path == cfg.lora_weights_path):
                if self.engine:
                    try: self.engine.cleanup()
                    except Exception: pass
                self.engine = PollingInferenceEngine(cfg)
            else:
                # reuse loaded model; just refresh runtime knobs
                self.engine.config.polling_interval = cfg.polling_interval
                self.engine.config.fps = cfg.fps
                self.engine.config.max_new_tokens = cfg.max_new_tokens
                self.engine.config.prompt = cfg.prompt

            engine = self.engine
            prompt = cfg.prompt

            # naturalizer (optional, off by default)
            self.naturalizer = None
            if req.use_naturalizer and FeedbackNaturalizer is not None:
                try: self.naturalizer = FeedbackNaturalizer()
                except Exception as e: logger.warning(f"naturalizer disabled: {e}")

            # 1) start the camera FIRST so the preview is live while the model loads
            is_webcam = not req.is_file
            if is_webcam:
                self._status("connecting", "Connecting to camera…")
                engine.stream_handler.start_stream_capture(str(req.source))
                time.sleep(0.8)  # let the buffer fill a few frames for preview
            else:
                path = str(SAMPLE_DIR / req.source)
                if not os.path.exists(path):
                    self._status("error", f"Video not found: {req.source}"); self.running = False; return

            # 2) load the model (slow on first run) — preview keeps streaming meanwhile
            if not getattr(engine, "_is_loaded", False):
                self._status("loading", "Loading model… (first run can take a while)")
                if not engine.load_model():
                    self._status("error", "Failed to load model."); self.running = False; return

            if not is_webcam:
                if not engine.stream_handler.open_video_file(str(SAMPLE_DIR / req.source)):
                    self._status("error", "Failed to open video file."); self.running = False; return

            # 3) warmup
            if req.warmup_runs > 0:
                self._status("warmup", "Warming up…")
                try: engine.warmup(req.warmup_runs)
                except Exception as e: logger.warning(f"warmup: {e}")

            # 4) polling loop
            self._status("running", "Coaching started.")
            poll = 0
            total = engine.stream_handler.total_duration if not is_webcam else 0.0
            while self.running:
                if (not is_webcam) and engine.stream_handler.current_position >= total > 0:
                    break
                t0 = time.time()
                try:
                    vframes, cframes, slice_len = engine.stream_handler.get_frames_for_inference(
                        engine.image_processor, engine.video_processor,
                        num_video_frames=cfg.num_frames,
                        num_context_images=cfg.num_context_images,
                        polling_interval=cfg.polling_interval,
                    )
                    if slice_len == 0:
                        time.sleep(cfg.polling_interval); continue

                    poll += 1
                    # Show 'Analyzing…' while the single (non-streaming) inference runs.
                    # We use the SAME blocking call as the Gradio app — generate() in
                    # this thread — to avoid the background-thread + per-token streamer
                    # overhead that adds latency on the Jetson's CPU.
                    self.broadcast({"type": "thinking", "poll": poll, "message": "Analyzing…"})
                    response, _ttft, _in_tok, _out_tok = engine.run_single_inference(
                        vframes, cframes, prompt, slice_len)

                    parsed = classify(response)
                    display = parsed["feedback"]
                    if self.naturalizer:
                        try: display = self.naturalizer.process(response).get("display", display)
                        except Exception: pass

                    self.broadcast({"type": "feedback",
                                    "state": parsed["state"],
                                    "exercise": parsed["exercise"],
                                    "feedback": parsed["feedback"],
                                    "display": display,
                                    "raw": parsed["raw"],
                                    "poll": poll,
                                    "latency_ms": round((time.time() - t0) * 1000, 1)})
                    self.status["polls"] = poll
                except Exception as e:
                    logger.error(f"poll {poll+1}: {e}", exc_info=True)
                    self._status("error", f"Poll error: {e}")

                time.sleep(cfg.polling_interval)

            self._status("complete", f"Session complete — {poll} polls.", polls=poll)
        except Exception as e:
            logger.error(f"session fatal: {e}", exc_info=True)
            self._status("error", f"Fatal error: {e}")
        finally:
            self.running = False
            if self.engine:
                try: self.engine.stream_handler.close()
                except Exception: pass


SESSION = Session()
app = FastAPI(title="Mobile-VideoGPT Coach")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/cameras")
def api_cameras():
    return {"cameras": list_cameras()}


@app.get("/api/sample_videos")
def api_sample_videos():
    vids = []
    if SAMPLE_DIR.exists():
        for ext in ("*.mp4", "*.avi", "*.mov"):
            vids += [p.name for p in SAMPLE_DIR.glob(ext)]
    return {"videos": sorted(vids)}


@app.get("/api/config")
def api_config():
    c = PollingConfig()
    return {"base_model_path": c.base_model_path, "lora_weights_path": c.lora_weights_path,
            "polling_interval": c.polling_interval, "fps": c.fps, "num_frames": c.num_frames,
            "max_new_tokens": c.max_new_tokens, "prompt": c.prompt}


@app.get("/api/status")
def api_status():
    return {"running": SESSION.running, **SESSION.status}


@app.post("/api/start")
def api_start(req: StartRequest):
    return JSONResponse(SESSION.start(req))


@app.post("/api/stop")
def api_stop():
    return JSONResponse(SESSION.stop())


@app.get("/api/stream")
def api_stream():
    def gen():
        q = SESSION.subscribe()
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    ev = q.get(timeout=10)
                    yield f"data: {json.dumps(ev)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            SESSION.unsubscribe(q)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/preview.mjpg")
def api_preview():
    def gen():
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        while True:
            frame = SESSION.latest_frame_bgr()
            if frame is None:
                frame = blank
            # Downscale for the PREVIEW only. latest_frame_bgr() already returns a
            # copy and cv2.resize makes a new array, so the model's full-res buffer
            # frames are never touched — zero effect on inference/accuracy.
            h, w = frame.shape[:2]
            if w > 480:
                frame = cv2.resize(frame, (480, int(h * 480 / w)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            time.sleep(1 / 10)  # preview output cap (decoupled from model/accuracy)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
