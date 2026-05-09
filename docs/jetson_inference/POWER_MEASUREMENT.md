# Power Measurement — Step-by-Step

> Self-contained guide to measure Jetson Orin Nano Super power consumption
> during real inference. Designed for headless operation (read this on
> laptop, execute on Jetson via SSH).
>
> All file paths are inside `~/Documents/mobile-videogpt-adaptation/`.

---

## 🛠️ One-time prep

Make sure power mode is MAXN_SUPER (only needs to be done once per boot):

```bash
ssh edgevlm@<jetson-ip>
sudo nvpmodel -m 2
sudo jetson_clocks
```

Verify:
```bash
nvpmodel -q
# should print: NV Power Mode: MAXN_SUPER
```

---

## 🏃 Full Procedure (two SSH sessions to Jetson)

### Session 1 — Inference

```bash
ssh edgevlm@<jetson-ip>
cd ~/Documents/mobile-videogpt-adaptation
conda activate mvgpt
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py
```

Wait until logs print:

```
Running on local URL:  http://0.0.0.0:7860
```

**Don't click Start in the browser yet.**

### Session 2 — Power logger

Open another SSH session in a separate terminal on your laptop:

```bash
ssh edgevlm@<jetson-ip>
cd ~/Documents/mobile-videogpt-adaptation
mkdir -p diagnostic_frames

# Start logging — runs in foreground, leave it running
sudo tegrastats --interval 1000 --logfile diagnostic_frames/power_log.txt
```

You won't see output (it's writing to file). That's fine.

### Run the inference

Now go back to your laptop browser → open `http://<jetson-ip>:7860` →
pick **Direct Webcam** → **Start**.

**Let it complete at least 5 polls.** Watch the response area — wait until
you've seen 5 different responses stream in. Roughly 60-90 seconds.

### Stop everything

1. In **Session 2**: press **Ctrl+C** to stop tegrastats
2. In **Session 1**: press **Ctrl+C** to stop Gradio
3. In the browser: close the tab

### Compute the numbers

Still in Session 2:

```bash
cd ~/Documents/mobile-videogpt-adaptation
bash scripts/analyze_power.sh
```

Or run inline:

```bash
echo "==========================================="
echo "VDD_IN (total board power)"
echo "==========================================="
grep -oP 'VDD_IN \K[0-9]+' diagnostic_frames/power_log.txt | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  printf "  Min   = %.2f W\n", min/1000
  printf "  Avg   = %.2f W\n", sum/n/1000
  printf "  Peak  = %.2f W\n", max/1000
  printf "  Samples: %d\n", n
}'

echo ""
echo "==========================================="
echo "VDD_CPU_GPU_CV (compute-only power)"
echo "==========================================="
grep -oP 'VDD_CPU_GPU_CV \K[0-9]+' diagnostic_frames/power_log.txt | awk '{
  if (NR==1 || $1<min) min=$1
  if ($1>max) max=$1
  sum+=$1; n++
} END {
  printf "  Min   = %.2f W\n", min/1000
  printf "  Avg   = %.2f W\n", sum/n/1000
  printf "  Peak  = %.2f W\n", max/1000
}'
```

## 📊 What you'll see — example output

```
==========================================
VDD_IN (total board power)
==========================================
  Min   = 5.42 W
  Avg   = 13.18 W
  Peak  = 18.74 W
  Samples: 78

==========================================
VDD_CPU_GPU_CV (compute-only power)
==========================================
  Min   = 0.84 W
  Avg   = 5.92 W
  Peak  = 9.31 W
```

Use these numbers in `REVIEWER_RESPONSE.md` Section 3.

---

## 🎯 What goes in the paper

| Metric | Source |
|---|---|
| Idle (model loaded, no inference) | The Min from VDD_IN |
| Average during inference | The Avg from VDD_IN |
| Peak during inference | The Peak from VDD_IN |
| Power envelope (hardware ceiling) | 25 W (MAXN_SUPER spec) |

---

## 💡 If something goes wrong

| Problem | Fix |
|---|---|
| `tegrastats: command not found` | It ships with JetPack — use full path: `/usr/bin/tegrastats` |
| `power_log.txt` is empty | Did you press Ctrl+C before any inference? Re-run, wait at least 60 s with active inference |
| `No matches found` from grep | Wrong field name — check log first line: `head -1 diagnostic_frames/power_log.txt` |
| Numbers all zero | tegrastats was running but inference never triggered. Verify Gradio Start was clicked and polls happened |

---

## 📋 Capture this output

Copy-paste the **example output** style into `REVIEWER_RESPONSE.md`.
Save the raw `power_log.txt` for your records — useful if reviewers ask
for raw data.
