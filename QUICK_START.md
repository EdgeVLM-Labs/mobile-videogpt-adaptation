# Quick Reference: Streaming Mobile-VideoGPT

## 🚀 Getting Started (30 seconds)

```bash
# 1. Run webcam demo
python demo_streaming.py

# 2. Process video file
python demo_streaming.py --video your_video.mp4

# 3. Run tests
python -m pytest tests/test_streaming.py -v
```

## 📁 Files Created

```
streaming/          # Core module (~1,580 lines)
├── buffer.py       # Frame buffering
├── context.py      # Temporal memory
├── predictor.py    # Action prediction
├── engine.py       # Main engine
└── utils.py        # Helpers

demo_streaming.py   # Webcam demo (350 lines)
streaming_config.yaml  # Configuration
tests/test_streaming.py  # Unit tests (400 lines)

PHASE1_ANALYSIS_REPORT.md    # Architecture analysis
STREAMING_README.md          # Full documentation
IMPLEMENTATION_SUMMARY.md    # This summary
```

## 🎯 Key Concepts

### 1. Sliding Window Processing

- Video buffered in 8-frame chunks
- 4-frame overlap between chunks
- Processes continuously, not frame-by-frame

### 2. Action Tokens

- `<next>`: Keep observing, don't speak
- `<feedback>`: Generate correction feedback
- `<correct>`: Positive reinforcement

### 3. Temporal Context

- Remembers last 3 chunks
- Maintains awareness across ~24 frames
- Helps detect patterns over time

## 🔧 Common Tasks

### Adjust Feedback Frequency

```yaml
# streaming_config.yaml
action_prediction:
  rules:
    time_based_interval: 5 # Every N chunks (default: 5)
    min_feedback_interval: 3.0 # Min seconds (default: 3.0)
```

### Change Video Settings

```yaml
video:
  chunk_size: 8 # Don't change (VideoMamba requirement)
  overlap: 4 # 0-7, higher = more context
  num_context_images: 16 # 8-16, lower = faster
```

### Optimize for Speed

```yaml
video:
  num_context_images: 8 # Reduce from 16

generation:
  max_new_tokens: 128 # Reduce from 256

temporal:
  max_history: 2 # Reduce from 3
```

### Optimize for Quality

```yaml
video:
  overlap: 6 # Increase from 4

temporal:
  max_history: 5 # Increase from 3

generation:
  max_new_tokens: 512 # Increase from 256
  temperature: 0.7 # Lower for consistency
```

## 🐛 Quick Troubleshooting

| Problem           | Solution                                                     |
| ----------------- | ------------------------------------------------------------ |
| Too slow          | Reduce `num_context_images`, lower `max_new_tokens`          |
| No feedback       | Lower `confidence_threshold`, reduce `min_feedback_interval` |
| Too much feedback | Increase `time_based_interval`, raise `confidence_threshold` |
| Out of memory     | Reduce `max_history`, use CPU mode                           |
| Webcam issues     | Try `--video 1` or `--video 2`                               |

## 📊 Performance Expectations

| Hardware | FPS   | Notes                        |
| -------- | ----- | ---------------------------- |
| RTX 3090 | 4-5   | Current, optimizable to 8-10 |
| RTX 4090 | 6-8   | Estimated                    |
| RTX 3060 | 3-4   | Estimated                    |
| CPU only | 0.5-1 | Very slow                    |

## 💡 Pro Tips

1. **Start Simple**: Use default config first
2. **Monitor FPS**: Press 's' during demo for stats
3. **Test with Video**: Easier debugging than webcam
4. **Check Logs**: Enable DEBUG logging for issues
5. **Profile**: Use `--max-frames 100` for quick tests

## 🔗 Important Links

- **Full Docs**: [STREAMING_README.md](STREAMING_README.md)
- **Analysis**: [PHASE1_ANALYSIS_REPORT.md](PHASE1_ANALYSIS_REPORT.md)
- **Tests**: `tests/test_streaming.py`
- **Config**: `streaming_config.yaml`

## 📝 Example Code Snippets

### Minimal Usage

```python
from streaming import StreamingMobileVideoGPT
import cv2

engine = StreamingMobileVideoGPT(
    model_path="Amshaker/Mobile-VideoGPT-0.5B"
)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    result = engine.process_frame(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    if result:
        print(result["feedback_text"])

cap.release()
```

### With Custom Config

```python
config = {
    "video": {"overlap": 6},
    "action_prediction": {
        "rules": {"time_based_interval": 3}
    }
}

engine = StreamingMobileVideoGPT(
    model_path="Amshaker/Mobile-VideoGPT-0.5B",
    config_dict=config
)
```

### Batch Processing

```python
import cv2
from pathlib import Path

engine = StreamingMobileVideoGPT(...)
results = []

for video_file in Path("videos/").glob("*.mp4"):
    engine.reset()  # Clear state between videos
    cap = cv2.VideoCapture(str(video_file))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        result = engine.process_frame(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        if result:
            results.append({
                "video": video_file.name,
                **result
            })

    cap.release()

# Save results
import json
with open("feedback_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## ✅ Checklist for First Run

- [ ] Python 3.8+ installed
- [ ] PyTorch with CUDA installed
- [ ] Mobile-VideoGPT dependencies installed
- [ ] `pip install opencv-python pyyaml einops`
- [ ] Model downloaded (auto on first run)
- [ ] Webcam connected (or video file ready)
- [ ] Run: `python demo_streaming.py`

## 🎓 Next Steps

1. **Test**: Run demo with sample video
2. **Tune**: Adjust config for your use case
3. **Optimize**: Implement async capture
4. **Train**: Collect data for action predictor
5. **Deploy**: Containerize for production

---

**Ready to start?** Run: `python demo_streaming.py`

For detailed information, see [STREAMING_README.md](STREAMING_README.md)
