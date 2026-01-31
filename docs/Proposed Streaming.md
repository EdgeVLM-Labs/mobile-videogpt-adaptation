# **Batch vs. Streaming**

## **ORIGINAL DESIGN: Why It Doesn't Work with Webcam**

### **The Code:**
```python
# Original: mobilevideogpt/model/dataloader.py
def _get_rawvideo_dec(video_path, ...):
    # 1. Open COMPLETE video file
    vreader = VideoReader(video_path)  # ❌ Needs a complete file!
    
    # 2. Get TOTAL length
    total_frames = len(vreader)  # ❌ Camera has infinite frames!
    
    # 3. Sample frames from ENTIRE video
    sample_pos = [0, 30, 60, 90, ...]  # ❌ Needs to know all positions upfront
    frames = vreader.get_batch(sample_pos)  # ❌ Gets frames all at once
    
    return frames  # Returns fixed set of frames
```

### **Why This Fails for Webcam:**
```
Webcam → Live stream (never ends) → ❌ Cannot load "complete video"
                                    ❌ Cannot calculate "total frames"
                                    ❌ Cannot sample from "entire video"
```

**Problem:** It's like trying to read a book that's still being written!

---

## **NEW DESIGN: How Streaming Works**

### **The Solution in 3 Parts:**

### **Part 1: Continuous Capture (Background)**
```python
# polling/stream_handler.py - Line 155
def _capture_loop(self):
    """Runs in background thread forever"""
    while True:
        ret, frame = self._cap.read()  # ✅ Get ONE frame from camera
        self.frame_buffer.append(frame)  # ✅ Add to circular buffer
        # Buffer keeps last 64 frames, old ones auto-removed
```

**What this does:**
```
Camera → [Frame 1] → Buffer
      → [Frame 2] → Buffer  (Frame 1 still there)
      → [Frame 3] → Buffer  (Frame 1, 2 still there)
      ... continuously ...
      → [Frame 64] → Buffer (Frames 1-63 still there)
      → [Frame 65] → Buffer (Frame 1 removed, keep 2-65)
```

### **Part 2: Extract Frames When Needed**
```python
# polling/stream_handler.py - Line 357
def get_frames_for_inference(self):
    """Called every N seconds by inference engine"""
    # Get last 16 frames from buffer
    buffer_frames = list(self.frame_buffer)  # ✅ Get current buffer
    
    # Sample uniformly (same as original, but from buffer)
    if len(buffer_frames) >= 16:
        step = len(buffer_frames) // 16
        selected = [buffer_frames[i * step] for i in range(16)]
    
    return selected  # Returns 16 frames (just like original!)
```

**What this does:**
```
Buffer [64 frames] → Sample 16 frames → [Frame 4, Frame 8, Frame 12, ...]
                     (every 4th frame)
```

### **Part 3: Polling Loop**
```python
# polling/inference_engine.py - Line 450
def start_polling(self):
    """Main loop"""
    while True:
        # 1. Get frames from buffer
        frames, context = self.stream_handler.get_frames_for_inference()
        
        # 2. Run SAME inference as original
        result = self.model.generate(frames, context, prompt)
        
        # 3. Show feedback
        print(f"Feedback: {result}")
        
        # 4. Wait for next poll
        time.sleep(3)  # Wait 3 seconds, then repeat
```

**What this does:**
```
Time 0s:  Buffer has frames 1-64    → Extract frames 4,8,12...64  → Inference → "Good form!"
Time 3s:  Buffer has frames 91-154  → Extract frames 94,98...154  → Inference → "Lower your back"
Time 6s:  Buffer has frames 181-244 → Extract frames 184,188...244 → Inference → "Great!"
... continues forever ...
```

---

## **Visual Comparison**

### **Original (Batch):**
```
┌──────────────┐
│ Complete     │
│ Video File   │  → Load ALL frames → Sample 16 → Inference → Done ✓
│ (Fixed size) │     (happens once)
└──────────────┘
```

**Limitation:** Needs complete video upfront

---

### **Streaming (Polling):**
```
┌──────────────┐
│   Webcam     │ ─┐
│ (Infinite)   │  │  
└──────────────┘  │
                  ↓
         ┌────────────────┐
         │  Circular      │  ←─── Background thread
         │  Buffer (64)   │        keeps filling
         └────────────────┘
                  │
         Every 3 seconds...
                  ↓
         ┌────────────────┐
         │ Sample 16      │
         │ from buffer    │  → Inference → Feedback
         └────────────────┘
                  ↓
              Wait 3s
                  ↓
         ┌────────────────┐
         │ Sample 16      │
         │ (new frames)   │  → Inference → Feedback
         └────────────────┘
                  │
              Repeat forever...
```

**Advantage:** Works with never-ending streams

---

## **The Key Trick**

**Original thinks:** "Give me the whole video, I'll process it once"

**Streaming thinks:** "Give me frames continuously, I'll process the last 16 whenever asked"

### **It's Like:**

**Original:** 
- Reading a complete book, then writing a summary
- ❌ Can't summarize a book that's still being written

**Streaming:**
- Reading the last chapter every few minutes as book is written
- ✅ Can give updates on a book being written in real-time

---

## **The Bottom Line**

| Feature | Original | Streaming |
|---------|----------|-----------|
| **Input** | `video.mp4` (file path) | `0` (camera index) |
| **Frame source** | `VideoReader(file)` | `cv2.VideoCapture(camera)` |
| **Frames available** | All at once | Continuously added to buffer |
| **Processing** | Once | Every N seconds |
| **Works with webcam?** | ❌ No | ✅ Yes |

**The streaming implementation wraps the original's "process 16 frames" logic in a loop that feeds it frames from a live buffer instead of a static file.**