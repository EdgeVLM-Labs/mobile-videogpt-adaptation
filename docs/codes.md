# Helper codes

#### 1) List total number of video files in dataset

```bash
cd dataset && for d in */; do echo "$d" $(find "$d" -name "*.mp4" | wc -l); done
```

#### 2) It loops through your videos and launches a tiny Python process for each one. If a video causes a crash (segfault), this script will catch it.

```bash
echo "Starting corruption check using Python/Decord..."
find dataset -name "*.mp4" | while read video; do
    python -c "
import sys
try:
    # Try loading with decord (standard for VideoGPT/VideoMamba)
    from decord import VideoReader, cpu
    vr = VideoReader('$video', ctx=cpu(0))
    len(vr) # Trigger header read
except ImportError:
    # Fallback to OpenCV if decord isn't installed
    import cv2
    cap = cv2.VideoCapture('$video')
    if not cap.isOpened(): sys.exit(1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1

    # Check exit code. 139 means Segfault (Crash), 1 means Error
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ CRASH/CORRUPT detected ($EXIT_CODE): $video"
    else
        echo "✅ OK: $video"
    fi
done
echo "Check complete."
```

