# Jetson Headless Demo Setup Guide

> Step-by-step guide to operate the inference pipeline from a remote laptop
> via SSH while the Jetson runs in headless mode (no GNOME desktop).
>
> Going headless frees ~1 GB of RAM, which is critical for hitting the
> fastest TTFT (~2.3s) reliably.

---

## 🎯 Why Headless?

| Mode | Free CUDA | Demo speed | Reliability |
|---|---|---|---|
| GUI desktop running | ~3.5–4.5 GB | TTFT ~10s (safe mode) | Variable |
| **Headless** | **~5.5–6.5 GB** | **TTFT ~2.3s (fast mode)** | **Bulletproof** |

In headless mode, you operate the Jetson via SSH from your laptop. The Jetson runs only the Gradio inference server — no desktop wasting memory.

---

## 📋 One-Time Setup

### Step 1 — Enable SSH on the Jetson

Open a terminal on the Jetson (using GUI for now):

```bash
# Install + enable SSH server (usually already installed on JetPack)
sudo apt install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Step 2 — Find the Jetson's IP address

```bash
hostname -I | awk '{print $1}'
```

Note this IP. Example: `192.168.1.126`

> ⚠️ DHCP IPs can change after reboot. For a stable demo, reserve the IP
> in your router or set a static IP. For testing, just check the IP again
> after reboots with the same command.

### Step 3 — Test SSH from your operator laptop

On your laptop:

```bash
ssh edgevlm@192.168.1.126
# (enter Jetson's password)
```

If it logs in → SSH works ✅

### Step 4 — Set up passwordless login (recommended)

On your laptop:

```bash
# Generate a key (skip if you already have ~/.ssh/id_ed25519)
ssh-keygen -t ed25519

# Copy public key to Jetson
ssh-copy-id edgevlm@192.168.1.126
# (enter password one last time)

# Test — should log in without prompting:
ssh edgevlm@192.168.1.126
```

### Step 5 — Verify SSH works for everything you need

Before going headless, confirm via SSH:

```bash
ssh edgevlm@192.168.1.126 "echo 'SSH ok' && free -h && nvpmodel -q"
```

You should see: `SSH ok`, memory info, and `NV Power Mode: MAXN_SUPER` (or similar).

---

## 🚀 Switch to Headless Mode

> ⚠️ **Don't skip Step 5 above.** Make sure SSH from laptop works first,
> otherwise you'll lose access if anything goes wrong.

### Switch (one-time per device):

SSH'd into the Jetson:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

The Jetson reboots into a black terminal-only screen (no GNOME). Wait ~30 seconds.

### Reconnect via SSH from your laptop:

```bash
ssh edgevlm@192.168.1.126
```

### Verify the memory savings:

```bash
free -h
nvpmodel -q
```

You should now see substantially more free memory (~6 GB vs ~3-4 GB before).

---

## 🎬 Running the Demo

### Once per Jetson power-on, set max performance:

```bash
ssh edgevlm@192.168.1.126
sudo nvpmodel -m 2          # MAXN_SUPER
sudo jetson_clocks          # lock max clocks
```

`nvpmodel -m 2` persists across reboots. `jetson_clocks` resets on reboot.

### Launch the inference server (every demo session):

```bash
ssh edgevlm@192.168.1.126
cd ~/Documents/mobile-videogpt-adaptation
conda activate mvgpt

# Clear any stale memory
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Launch with full performance flags
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py
```

The terminal will print a URL like:

```
Running on local URL:  http://0.0.0.0:7860
```

### Open the URL on your operator laptop:

In any browser on a device on the same network:

```
http://192.168.1.126:7860
```

(Replace with your Jetson's IP.)

You'll see the Gradio UI — click Start, the patient stands in front of the
USB camera, feedback streams onto the page.

---

## 🖥️ Demo UI Walkthrough

The interface is split into a single workspace with advanced settings tucked
into a tab navbar at the bottom. Everything the panel needs to see lives at
the top; engineer-facing controls stay one click away.

### Top half (always visible)

| Element | What it shows |
|---|---|
| Header banner | App name, model spec pills (`0.5B PARAMS`, `VIDEOMAMBA + QWEN2`, `FP16`) |
| Status badge | `⏸ Idle` → `🟢 Live` / `🟡 Analyzing` → `✅ Complete` / `❌ Error` with current poll # and timestamp |
| Input source | Radio chips: **Video File** / **Browser Webcam** / **Direct Webcam (Linux only)**. Selecting a mode swaps the preview and reveals the matching dropdown in the *Source* tab. |
| Preview | Single slot — shows a `gr.Video` for file mode, a webcam capture for browser mode, or a live frame-buffer preview for direct-V4L2 mode. Only one is visible at a time. |
| Coaching feedback | Large response card. Streams tokens during generation, then locks the final coaching text. |
| 🔊 Voice toggle | On by default. Speaks each completed poll's response via the browser's Web Speech API. |
| Start / Stop | Primary indigo gradient + secondary slate. |

### Voice feedback

Voice is rendered **client-side in the browser**, not on the Jetson. Implications:

- Zero extra Jetson CPU/GPU cost — the synth runs on the operator's laptop.
- Works in Chrome / Edge / Safari / Firefox out of the box. (Chrome has the smoothest English voices.)
- Toggle off if the panel asks for silence during a screenshot.
- Streaming-token partials are filtered out — only the final response per poll
  is spoken, deduped by poll ID so back-to-back identical answers still get
  voiced.
- Includes a Chrome keep-alive (pause/resume every 10 s) so longer
  naturalizer-rephrased responses don't cut off at ~15 s.

### Bottom half — Advanced tabs

| Tab | Contents |
|---|---|
| **Source** | Sample video dropdown, camera device picker, mode hints |
| **Model** | Base model (0.5B / 1.5B), LoRA adapter |
| **Inference** | Polling interval, frames per poll, sample FPS, max new tokens, warmup runs, prompt |
| **Naturalizer** | Toggle + similarity threshold for repeat-detection rephrasing |
| **Metrics** | Current-poll latency / TTFT / tok-per-sec, full session response history |
| **Logs** | Live tail of the inference logger |

For a panel demo, leave the tabs collapsed on **Source**. Switch to **Metrics**
if a reviewer asks "what was the latency on that one?".

---

### Stopping the demo:

Press `Ctrl+C` in the SSH terminal where Gradio is running.

---

## 🔄 Reverting to GUI Mode (if needed)

If you ever want the GNOME desktop back:

```bash
ssh edgevlm@192.168.1.126
sudo systemctl set-default graphical.target
sudo reboot
```

After reboot, the Jetson boots into the normal desktop with monitor again.

### Want GUI temporarily without permanent change?

```bash
sudo systemctl start gdm
```

This starts GNOME for the current session only. Next reboot still goes headless.

---

## 🛟 Recovery (If Something Breaks)

If SSH stops working after going headless and you can't get in:

1. **Plug in a monitor + keyboard** to the Jetson directly
2. The black terminal will show. Log in with your username/password.
3. Run the revert command:
   ```bash
   sudo systemctl set-default graphical.target
   sudo reboot
   ```
4. After reboot, GNOME desktop is back.

---

## 📋 Pre-Demo Checklist

Before the panel arrives, run through this:

```bash
# 1. SSH in and verify everything
ssh edgevlm@192.168.1.126

# 2. Inside the SSH session:
nvpmodel -q                                # → MAXN_SUPER
sudo jetson_clocks --show | head -3        # → max clocks locked
free -h                                    # → 6+ GB available
ls models/tensorrt/clip_vit_base_fp32.engine  # → engine file exists
ls /dev/video*                             # → /dev/video0 (webcam)

# 3. Launch the demo
cd ~/Documents/mobile-videogpt-adaptation
conda activate mvgpt
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py

# 4. On laptop browser, open http://192.168.1.126:7860

# 5. Verify health log lines:
#    ✓ Loading TensorRT CLIP engine: .../clip_vit_base_fp32.engine
#    ✓ TensorRT CLIP preloaded on GPU
#    ✓ Qwen2 layer placement: {'cuda:0': 24}
#    ✓ Model loaded successfully
```

If any step fails, see the **Recovery** section above.

---

## 📚 Related Docs

- [`JETSON_OPTIMIZATION_JOURNEY.md`](./JETSON_OPTIMIZATION_JOURNEY.md) — full optimization story
- [`jetson_inference_fixes.md`](./jetson_inference_fixes.md) — initial fixes from Phase 1

---

## 🆘 Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `ssh: connection refused` | SSH not running | `sudo systemctl start ssh` (need monitor) |
| `Connection refused` from laptop | Wrong IP / network mismatch | Re-check `hostname -I` on Jetson, ensure same WiFi |
| Black screen after reboot | Headless mode active (this is expected!) | SSH in from laptop |
| Want GUI back temporarily | — | `sudo systemctl start gdm` |
| Permanent GUI revert | — | `sudo systemctl set-default graphical.target && sudo reboot` |
| URL `0.0.0.0:7860` not reachable from laptop | Firewall or Gradio binding | Run `ufw status`, add `--server-name 0.0.0.0` to gradio_app.py if needed |
