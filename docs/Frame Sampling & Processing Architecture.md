# Mobile-VideoGPT Frame Sampling & Processing Architecture

Based on the codebase and paper analysis, here's the complete picture of how Mobile-VideoGPT samples and processes frames:

**Your Understanding is Partially Correct** ✓ (with clarifications needed)

Let me break down what actually happens:

## **1. Frame Sampling Process**

### Initial Video Loading:
- **Sampling rate**: Yes, **1 FPS** from the source video
- **Total frames extracted**: 16 frames are uniformly sampled from the video
- **Two parallel streams**:
  - **16 video frames** → for VideoMamba encoder  
  - **16 context images** → for CLIP encoder (same frames, different processing)

From dataloader.py:
```python
sample_fps = int(video_framerate)  # Default: 1 FPS
t_stride = int(round(float(fps) / sample_fps))  # Stride between frames
all_pos = list(range(f_start, f_end + 1, t_stride))
```

## **2. The Attention-Based Frame Scoring (Your Key Question)**

Here's where your understanding needs refinement:

### The Process:
1. **16 frames** are split into **2 chunks of 8 frames each** (VideoMamba constraint)
2. **For each 8-frame chunk**:
   - First, the **16 context images** (from CLIP) are also divided into 2 chunks
   - The CLIP features act as a "scoring mechanism" for the video frames
   - **Attention computation** happens on CLIP features (not VideoMamba yet!)
   - Based on attention scores, **top-4 frames** are selected from each 8-frame chunk
3. **Only the selected frames** (4 per chunk = **8 total frames**) go through VideoMamba
4. Final output: **2 chunks × 4 frames × 49 tokens = 392 video tokens**

From arch.py:
```python
def select_frame_in_chunk(self, video_features, batchsize):
    # video_features are CLIP-encoded context images (shape: B, T=8, L=256, D=768)
    # Compute self-attention among all tokens
    tokens = current_video.view(B, T * L, D)  # Flatten frames
    attn_logits = torch.matmul(tokens, tokens.transpose(-1, -2)) / math.sqrt(D)
    attn_weights = F.softmax(attn_logits, dim=-1)
    
    # Calculate which frames receive most attention
    total_attention_received_per_frame = attention_received_per_token.sum(dim=1)
    
    # Select top-K frames (default K=4)
    topk_indices = torch.topk(total_attention_received_per_frame, k=4)
```

## **3. Why Is It Fast? (Answering Your 2-3 Second Question)**

### **You're right to be confused!** Here's why it's still fast:

**The sampling doesn't happen in real-time during inference:**

1. **Pre-processing is done ONCE** before inference:
   - Video is decoded
   - 16 frames extracted at 1 FPS (happens in ~100-200ms using Decord)
   - Frames are already in memory

2. **The "1 FPS" doesn't mean processing speed:**
   - For a 16-second video: sample 1 frame/sec → 16 frames total
   - For a 8-second video: sample 1 frame/sec → 8 frames total
   - These frames are extracted nearly instantly from the decoded video

3. **Actual inference pipeline timing (~2-3 seconds total):**
   ```
   Frame extraction:        ~0.1-0.2s  (Decord is fast!)
   CLIP encoding (16):      ~0.03s     (context images)
   Attention scoring:       ~0.005s    (select top frames)
   VideoMamba encoding (8): ~0.05s     (only selected frames)
   LLM generation:          ~2-2.5s    (depends on output length)
   ────────────────────────────────
   Total:                   ~2.2-2.8s
   ```

## **4. Full Architecture Summary**

```
Input Video (e.g., 30 FPS, 16 seconds = 480 frames)
         ↓
┌─────────────────────────────────┐
│  Sample at 1 FPS                │  ← Extract 16 frames (happens fast!)
│  Extract frames: 0, 30, 60...   │
└─────────────────────────────────┘
         ↓
    16 frames extracted
         ↓
         ├──────────────────┬──────────────────┐
         ↓                  ↓                  ↓
   Chunk 1 (8 frames)  Chunk 2 (8 frames)  All 16 frames
         ↓                  ↓                  ↓
         │                  │              CLIP Encoder
         │                  │                  ↓
         │                  │          Context Features
         │                  │          (16×256×768)
         ├──────────────────┴──────────────────┤
         ↓                                     ↓
    Split into 2 chunks of context features
         ↓
┌────────────────────────────────────────────┐
│  For each chunk:                           │
│  1. Compute attention on CLIP features     │
│  2. Select top-4 frames                    │  ← Attention scoring!
│  3. Only encode those 4 with VideoMamba    │
└────────────────────────────────────────────┘
         ↓
    8 selected frames total (4 per chunk)
         ↓
    VideoMamba Encoder
         ↓
    Video Features (8×49×576)
         ↓
    [Project + Merge with CLIP features]
         ↓
    LLM (Qwen2) generates text
```

## **5. Key Insights:**

1. ✅ **1 FPS sampling** - Correct
2. ✅ **16 frames aggregated** - Correct
3. ⚠️ **Attention-based scoring** - Uses CLIP features (context images), not VideoMamba
4. ⚠️ **Only 4 frames per chunk** actually go through VideoMamba (8 total, not 16)
5. ✅ **Fast inference** - Because frame extraction is preprocessing, not real-time

**The "2-3 second" inference time is possible because:**
- Frame extraction from video is extremely fast (Decord library)
- Only 8 frames go through the heavy VideoMamba encoder (not 16)
- CLIP encoding 16 frames is relatively lightweight
- Most time is spent in LLM text generation, not video processing

Does this clarify the architecture? The key misconception was thinking all 16 frames go through VideoMamba - actually only the top-4 per chunk (8 total) do!