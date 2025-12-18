# Video Augmentation Guide

The video augmentation system creates additional training samples by applying various transformations to existing videos.

- Videos are loaded frame-by-frame using OpenCV
- Frames are converted to PIL Images for augmentation
- Augmentations from vidaug library are applied
- Output maintains original video FPS
- Output codec: mp4v (widely compatible)

## Usage

### During Dataset Initialization

The augmentation step is integrated into the `initialize_dataset.sh` script and will be offered as an option after cleaning (if enabled):

```bash
bash scripts/initialize_dataset.sh
```

The script will:

1. Download videos
2. Filter ground truth
3. Clean dataset (optional)
4. **Augment videos (optional)** ← New step!
5. Generate train/val/test splits

### Standalone Usage

You can also run the augmentation script independently:

```bash
python utils/augment_videos.py
```

## Available Augmentation Techniques

The script provides 14 different augmentation techniques:

| Index | Technique                  | Description                                         |
| ----- | -------------------------- | --------------------------------------------------- |
| 1     | Horizontal Flip            | Mirrors the video horizontally                      |
| 2     | Vertical Flip              | Mirrors the video vertically                        |
| 3     | Random Rotate (±10°)       | Rotates video by random angle within ±10°           |
| 4     | Random Resize (±20%)       | Scales video up or down by up to 20%                |
| 5     | Gaussian Blur              | Applies blur effect to simulate camera focus issues |
| 6     | Add Brightness (+30)       | Increases brightness to simulate different lighting |
| 7     | Multiply Brightness (1.2x) | Multiplies pixel values for brightness adjustment   |
| 8     | Random Translate (±15px)   | Shifts video position by up to 15 pixels            |
| 9     | Random Shear               | Applies shearing transformation                     |
| 10    | Invert Color               | Inverts all colors (creates negative effect)        |
| 11    | Salt Noise                 | Adds random white pixels (salt noise)               |
| 12    | Pepper Noise               | Adds random black pixels (pepper noise)             |
| 13    | Temporal Downsample (0.8x) | Reduces frame rate by 20%                           |
| 14    | Elastic Transformation     | Applies elastic distortion to frames                |

## Workflow

1. **Display Video Counts**: Shows number of videos in each exercise folder

   ```
   Exercise folders and video counts:
   1. knee_circles                          (5 videos)
   2. squats                                (5 videos)
   3. pushups_on_knees                      (5 videos)
   ```

2. **Select Folders**: Choose which exercise folders to augment

   ```
   Enter the indices of folders you want to augment (comma-separated)
   Example: 1,3,5 or just press Enter to augment all
   Folder indices: 1,2
   ```

3. **Select Augmentations**: For each selected folder, choose augmentation techniques

   ```
   Enter augmentation techniques to apply (comma-separated indices)
   Example: 1,3,5 for Horizontal Flip, Random Rotate, Gaussian Blur
   Augmentation indices: 1,3,5
   ```

4. **Processing**: The script processes each video and creates augmented versions

5. **JSON Update**: Automatically updates all JSON files with new video paths

## Output

### File Naming Convention

Augmented videos are saved with the following naming pattern:

```
original_video_name_<augmentation_index>.mp4
```

Example:

- Original: `knee_circles_001.mp4`
- Augmented with technique #1: `knee_circles_001_1.mp4`
- Augmented with technique #3: `knee_circles_001_3.mp4`

### Updated JSON Files

The script automatically updates three JSON files:

1. **dataset/fine_grained_labels.json**: Copies labels from original videos
2. **dataset/manifest.json**: Adds augmented video metadata
3. **dataset/ground_truth.json**: Includes augmented videos in ground truth

Each augmented video entry references its original video for traceability.

## Best Practices

### Recommended Augmentations for Exercise Videos

1. **Horizontal Flip (1)**: Essential for exercises that can be performed on either side
2. **Random Rotate (3)**: Helps with camera angle variations
3. **Gaussian Blur (5)**: Simulates different camera qualities
4. **Add Brightness (6)**: Handles different lighting conditions
5. **Random Translate (8)**: Helps with framing variations

### Augmentations to Use Carefully

- **Vertical Flip (2)**: May not make sense for most exercises
- **Invert Color (10)**: Creates unrealistic videos, use sparingly
- **Salt/Pepper Noise (11, 12)**: Only for robustness to low-quality videos
- **Temporal Downsample (13)**: May lose important motion information
