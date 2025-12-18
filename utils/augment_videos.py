#!/usr/bin/env python3
"""
Video Augmentation Script for QVED Dataset
Uses vidaug library to apply various augmentation techniques to exercise videos.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import vidaug.augmentors as va
from PIL import Image, ImageFilter
import sys

# Base directory
BASE_DIR = Path(__file__).parent.parent / "dataset"
GROUND_TRUTH_FILE = BASE_DIR / "fine_grained_labels.json"
MANIFEST_FILE = BASE_DIR / "manifest.json"
OUTPUT_GROUND_TRUTH_FILE = BASE_DIR / "ground_truth.json"

# Define available augmentations with numbers
AUGMENTATION_OPTIONS = {
    1: ("Horizontal Flip", va.HorizontalFlip()),
    2: ("Vertical Flip", va.VerticalFlip()),
    3: ("Random Rotate (±10°)", va.RandomRotate(degrees=10)),
    4: ("Random Resize (±20%)", va.RandomResize(rate=0.2)),
    5: ("Gaussian Blur", va.GaussianBlur(sigma=1.5)),
    6: ("Add Brightness (+30)", va.Add(value=30)),
    7: ("Multiply Brightness (1.2x)", va.Multiply(value=1.2)),
    8: ("Random Translate (±15px)", va.RandomTranslate(x=15, y=15)),
    9: ("Random Shear", va.RandomShear(x=0.1, y=0.1)),
    10: ("Invert Color", va.InvertColor()),
    11: ("Salt Noise", va.Salt(ratio=100)),
    12: ("Pepper Noise", va.Pepper(ratio=100)),
    13: ("Temporal Downsample (0.8x)", va.Downsample(ratio=0.8)),
    14: ("Elastic Transformation", va.ElasticTransformation(alpha=10, sigma=3)),
}


def display_augmentation_options():
    """Display all available augmentation options."""
    print("\n" + "="*60)
    print("Available Video Augmentation Techniques:")
    print("="*60)
    for idx, (name, _) in AUGMENTATION_OPTIONS.items():
        print(f"  {idx:2d}. {name}")
    print("="*60 + "\n")


def load_video_frames(video_path):
    """Load video frames as a list of PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    cap.release()
    return frames


def save_video_frames(frames, output_path, fps=30):
    """Save frames as a video file."""
    if not frames:
        print(f"Warning: No frames to save for {output_path}")
        return False

    # Convert first frame to get dimensions
    first_frame = np.array(frames[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        # Convert PIL Image to numpy array
        frame_np = np.array(frame)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return True


def get_video_fps(video_path):
    """Get the FPS of a video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30


def augment_video(video_path, augmentors, output_path):
    """Apply augmentation to a video."""
    print(f"  Processing: {video_path.name}...", end=" ", flush=True)

    # Load video frames
    frames = load_video_frames(video_path)

    if not frames:
        print("❌ Failed to load frames")
        return False

    # Apply augmentation sequence
    try:
        augmented_frames = augmentors(frames)
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        return False

    # Get original FPS
    fps = get_video_fps(video_path)

    # Save augmented video
    success = save_video_frames(augmented_frames, output_path, fps)

    if success:
        print("✓")
        return True
    else:
        print("❌ Failed to save")
        return False


def update_json_files(augmented_videos_info):
    """Update JSON files with augmented video paths."""
    print("\n" + "="*60)
    print("Updating JSON files with augmented videos...")
    print("="*60)

    # Load existing JSON files
    with open(GROUND_TRUTH_FILE, 'r') as f:
        fine_grained_labels = json.load(f)

    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)

    # Load ground_truth.json if it exists
    ground_truth = {}
    if OUTPUT_GROUND_TRUTH_FILE.exists():
        with open(OUTPUT_GROUND_TRUTH_FILE, 'r') as f:
            ground_truth = json.load(f)

    # Add augmented videos to JSON files
    for aug_info in augmented_videos_info:
        original_path = aug_info['original_path']
        augmented_path = aug_info['augmented_path']

        # Find original entry in fine_grained_labels
        if original_path in fine_grained_labels:
            # Copy the entry for augmented video
            fine_grained_labels[augmented_path] = fine_grained_labels[original_path].copy()

        # Add to manifest (copy from original if exists)
        # Use string format to maintain consistency with original entries
        if original_path in manifest:
            # If original is a string, use it directly; if dict, extract the exercise name
            original_value = manifest[original_path]
            if isinstance(original_value, str):
                manifest[augmented_path] = original_value
            elif isinstance(original_value, dict) and 'path' in original_value:
                # Get exercise name from original entry's key
                manifest[augmented_path] = original_value.get('exercise', original_value.get('path', augmented_path).split('/')[0].replace('_', ' '))
            else:
                manifest[augmented_path] = augmented_path.split('/')[0].replace('_', ' ')
        else:
            # Extract exercise name from path (folder name)
            exercise_name = augmented_path.split('/')[0]
            manifest[augmented_path] = exercise_name

        # Add to ground_truth.json if original exists there
        if original_path in ground_truth:
            ground_truth[augmented_path] = ground_truth[original_path].copy()

    # Save updated JSON files
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(fine_grained_labels, f, indent=2)
    print(f"✓ Updated {GROUND_TRUTH_FILE}")

    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Updated {MANIFEST_FILE}")

    if ground_truth:
        with open(OUTPUT_GROUND_TRUTH_FILE, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print(f"✓ Updated {OUTPUT_GROUND_TRUTH_FILE}")

    print(f"\n✓ Added {len(augmented_videos_info)} augmented videos to JSON files")


def main():
    print("\n" + "="*60)
    print("Video Augmentation Tool for QVED Dataset")
    print("="*60)

    # Get list of exercise folders
    exercise_folders = sorted([d for d in BASE_DIR.iterdir() if d.is_dir() and (d / "*.mp4" or list(d.glob("*.mp4")))])

    if not exercise_folders:
        print("❌ No exercise folders found in dataset directory!")
        return

    # Display video counts
    print("\nExercise folders and video counts:")
    print("-" * 60)
    for idx, folder in enumerate(exercise_folders, 1):
        video_count = len(list(folder.glob("*.mp4")))
        print(f"  {idx}. {folder.name:<40} ({video_count} videos)")
    print("-" * 60)

    # Let user choose folders to augment
    print("\nEnter the indices of folders you want to augment (comma-separated)")
    print("Example: 1,3,5 or just press Enter to augment all")
    folder_input = input("Folder indices: ").strip()

    if folder_input:
        try:
            selected_indices = [int(x.strip()) for x in folder_input.split(',')]
            selected_folders = [exercise_folders[i-1] for i in selected_indices if 1 <= i <= len(exercise_folders)]
        except (ValueError, IndexError):
            print("❌ Invalid input! Please enter valid comma-separated numbers.")
            return
    else:
        selected_folders = exercise_folders

    print(f"\n✓ Selected {len(selected_folders)} folder(s) for augmentation")

    # Display augmentation options
    display_augmentation_options()

    # Track all augmented videos for JSON update
    all_augmented_videos = []

    # For each selected folder, ask for augmentation techniques
    for folder in selected_folders:
        print("\n" + "="*60)
        print(f"Folder: {folder.name}")
        print("="*60)

        videos = sorted(list(folder.glob("*.mp4")))
        if not videos:
            print("No videos found, skipping...")
            continue

        print(f"Found {len(videos)} video(s)")
        print("\nEnter augmentation techniques to apply (comma-separated indices)")
        print("Example: 1,3,5 for Horizontal Flip, Random Rotate, Gaussian Blur")
        aug_input = input("Augmentation indices: ").strip()

        if not aug_input:
            print("No augmentations selected, skipping folder...")
            continue

        try:
            selected_aug_indices = [int(x.strip()) for x in aug_input.split(',')]
            selected_augmentors = []
            aug_names = []

            for idx in selected_aug_indices:
                if idx in AUGMENTATION_OPTIONS:
                    name, augmentor = AUGMENTATION_OPTIONS[idx]
                    selected_augmentors.append(augmentor)
                    aug_names.append(name)
                else:
                    print(f"Warning: Invalid augmentation index {idx}, skipping...")

            if not selected_augmentors:
                print("No valid augmentations selected, skipping folder...")
                continue

            print(f"\n✓ Will apply: {', '.join(aug_names)}")

            # Create augmentation sequence
            seq = va.Sequential(selected_augmentors)

            # Process each video in the folder
            for video_path in videos:
                # Generate output filename with augmentation index
                for aug_idx in selected_aug_indices:
                    if aug_idx not in AUGMENTATION_OPTIONS:
                        continue

                    # Create individual augmentor for this technique
                    _, augmentor = AUGMENTATION_OPTIONS[aug_idx]
                    single_aug = va.Sequential([augmentor])

                    video_stem = video_path.stem
                    output_filename = f"{video_stem}_{aug_idx}.mp4"
                    output_path = folder / output_filename

                    # Apply augmentation
                    success = augment_video(video_path, single_aug, output_path)

                    if success:
                        # Track for JSON update
                        relative_original = str(Path(folder.name) / video_path.name)
                        relative_augmented = str(Path(folder.name) / output_filename)
                        all_augmented_videos.append({
                            'original_path': relative_original,
                            'augmented_path': relative_augmented
                        })

        except ValueError:
            print("❌ Invalid input! Please enter valid comma-separated numbers.")
            continue

    # Update JSON files
    if all_augmented_videos:
        update_json_files(all_augmented_videos)
        print("\n" + "="*60)
        print(f"✓ Augmentation Complete! Created {len(all_augmented_videos)} augmented videos")
        print("="*60)
    else:
        print("\n❌ No videos were augmented")


if __name__ == "__main__":
    main()
