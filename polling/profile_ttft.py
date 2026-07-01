#!/usr/bin/env python3
"""
TTFT profiler — find out exactly where prefill time goes.

Loads the model the SAME way the web app does, then:
  1. Prints device + dtype of every major component (LLM, VideoMamba video
     tower, CLIP image tower, mm projectors). A tower on CPU or in fp32 is the
     usual cause of a multi-second TTFT.
  2. CUDA-times each vision tower's forward + the whole generate, averaged over
     a few dummy runs (zeros, same shapes the live pipeline feeds).

Run on the Jetson, in the model's conda env:
    cd <repo>/mobile-videogpt-adaptation
    python -m polling.profile_ttft
"""
import os, sys, time, collections
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine


def summarize_module(name, mod):
    if mod is None:
        print(f"  {name:24s}: <None>")
        return
    devs, dtypes, n = collections.Counter(), collections.Counter(), 0
    for p in mod.parameters():
        devs[str(p.device)] += p.numel()
        dtypes[str(p.dtype)] += p.numel()
        n += p.numel()
    if n == 0:
        print(f"  {name:24s}: no params")
        return
    dev = ", ".join(f"{k}:{v/1e6:.1f}M" for k, v in devs.items())
    dt = ", ".join(f"{k}:{v/1e6:.1f}M" for k, v in dtypes.items())
    print(f"  {name:24s}: {n/1e6:6.1f}M params | device[{dev}] | dtype[{dt}]")


def main():
    cfg = PollingConfig()
    print("=" * 70)
    print(f"base={cfg.base_model_path}  lora={cfg.lora_weights_path}")
    print(f"num_frames={cfg.num_frames}  num_context_images={cfg.num_context_images}")
    print("=" * 70)

    engine = PollingInferenceEngine(cfg)
    if not engine.load_model():
        print("FAILED to load model"); return
    m = engine.model

    print("\n--- COMPONENT PLACEMENT (look for cpu / float32) ---")
    summarize_module("LLM (full model)", m)
    try: summarize_module("video tower (Mamba)", m.get_vision_tower())
    except Exception as e: print("  video tower:", e)
    try: summarize_module("image tower (CLIP)", m.get_image_vision_tower())
    except Exception as e: print("  image tower:", e)
    for attr in ("mm_projector", "image_mm_projector"):
        sub = getattr(getattr(m, "model", m), attr, None)
        summarize_module(attr, sub)

    # ---- CUDA-timed split via forward hooks ----
    times = collections.defaultdict(list)

    def mk_hooks(tag):
        ev = {}
        def pre(mod, inp):
            s = torch.cuda.Event(enable_timing=True); s.record(); ev["s"] = s
        def post(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev["e"] = e
            torch.cuda.synchronize(); times[tag].append(ev["s"].elapsed_time(ev["e"]))
        return pre, post

    handles = []
    try:
        vt = m.get_vision_tower(); p, q = mk_hooks("video_tower")
        handles += [vt.register_forward_pre_hook(p), vt.register_forward_hook(q)]
    except Exception as e: print("hook video tower:", e)
    try:
        it = m.get_image_vision_tower(); p, q = mk_hooks("image_tower")
        handles += [it.register_forward_pre_hook(p), it.register_forward_hook(q)]
    except Exception as e: print("hook image tower:", e)

    dummy_v = [torch.zeros((3, cfg.image_resolution, cfg.image_resolution),
                           dtype=torch.float16, device=cfg.device) for _ in range(cfg.num_frames)]
    dummy_c = [torch.zeros((3, cfg.image_resolution, cfg.image_resolution),
                           dtype=torch.float16, device=cfg.device) for _ in range(cfg.num_context_images)]

    print("\n--- TIMED RUNS (1 warmup + 4 measured, dummy zeros) ---")
    N = 5
    for i in range(N):
        torch.cuda.synchronize(); t0 = time.time()
        engine.run_single_inference(dummy_v, dummy_c, "Analyze this exercise.", cfg.num_frames)
        torch.cuda.synchronize(); dt = (time.time() - t0) * 1000
        tag = "warmup" if i == 0 else f"run{i}"
        vt_ms = times["video_tower"][-1] if times["video_tower"] else 0
        it_ms = times["image_tower"][-1] if times["image_tower"] else 0
        print(f"  {tag:7s}: total={dt:7.0f}ms  video_tower={vt_ms:6.0f}ms  image_tower={it_ms:6.0f}ms")

    for h in handles: h.remove()

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0
    print("\n--- AVERAGE over measured runs (excludes warmup) ---")
    print(f"  video_tower (VideoMamba): {avg(times['video_tower'][1:]):7.0f} ms")
    print(f"  image_tower (CLIP)      : {avg(times['image_tower'][1:]):7.0f} ms")
    print("  (LLM prefill+decode = total - the two towers above)")


if __name__ == "__main__":
    main()
