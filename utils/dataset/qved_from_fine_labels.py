import json
import os
import random
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("dataset")
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"
MANIFEST_JSON = BASE_DIR / "manifest.json"
OUTPUT_TRAIN_JSON = BASE_DIR / "qved_train.json"
OUTPUT_VAL_JSON = BASE_DIR / "qved_val.json"
OUTPUT_TEST_JSON = BASE_DIR / "qved_test.json"
USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

# Dataset split ratios (adjustable)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42  # For reproducibility

def main():
    # Load manifest to map video filenames to full paths
    with open(MANIFEST_JSON, 'r') as f:
        manifest = json.load(f)

    # Create reverse lookup: filename -> full_path
    filename_to_path = {}
    filename_to_exercise = {}
    for full_path, exercise in manifest.items():
        filename = os.path.basename(full_path)
        filename_to_path[filename] = full_path
        # Handle both string and dict values in manifest
        if isinstance(exercise, str):
            filename_to_exercise[filename] = exercise.replace('_', ' ')
        elif isinstance(exercise, dict):
            # For dict entries (augmented videos), extract exercise from path
            exercise_name = full_path.split('/')[0] if '/' in full_path else 'unknown'
            filename_to_exercise[filename] = exercise_name.replace('_', ' ')
        else:
            filename_to_exercise[filename] = str(exercise).replace('_', ' ')

    # Load fine-grained labels
    with open(FINE_LABELS_JSON, 'r') as f:
        fine_labels = json.load(f)

    # Convert to Mobile-VideoGPT format
    output_data = []

    for record in fine_labels:
        video_path = record.get('video_path', '')
        # Extract filename from path (handles ./ prefix)
        filename = os.path.basename(video_path)

        # Look up full path in manifest
        if filename not in filename_to_path:
            print(f"Warning: {filename} not found in manifest, skipping")
            continue

        full_video_path = filename_to_path[filename]
        exercise = filename_to_exercise[filename]

        # Remove 'dataset/' prefix if present to make path relative to data_path
        # data_path is 'dataset', so videos should be 'exercise_name/video.mp4'
        if full_video_path.startswith('dataset/'):
            relative_video_path = full_video_path[len('dataset/'):]
        else:
            relative_video_path = full_video_path

        # Get assistant answer from most descriptive label
        if 'labels_descriptive' in record and record['labels_descriptive']:
            assistant_answer = record['labels_descriptive']
        elif 'labels' in record and record['labels']:
            assistant_answer = record['labels'][0] if isinstance(record['labels'], list) else record['labels']
        else:
            assistant_answer = "No feedback available."

        # Ensure assistant answer is a single string
        if isinstance(assistant_answer, list):
            assistant_answer = '\n'.join(str(item) for item in assistant_answer)
        else:
            assistant_answer = str(assistant_answer)

        user_prompt = USER_PROMPT_TEMPLATE  # No longer using exercise name in prompt

        output_data.append({
            "video": relative_video_path,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train"  # Will be updated during split
        })

    # Shuffle data for random split
    random.seed(RANDOM_SEED)
    random.shuffle(output_data)

    # Calculate split indices
    total_count = len(output_data)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)

    # Split the data
    train_data = output_data[:train_end]
    val_data = output_data[train_end:val_end]
    test_data = output_data[val_end:]

    # Update split labels
    for item in train_data:
        item["split"] = "train"
    for item in val_data:
        item["split"] = "val"
    for item in test_data:
        item["split"] = "test"

    # Write output JSONs
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TRAIN_JSON, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(OUTPUT_VAL_JSON, 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(OUTPUT_TEST_JSON, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset Split Summary")
    print(f"{'='*60}")
    print(f"Total videos: {total_count}")
    print(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
    print(f"\nSplit Distribution:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total_count*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/total_count*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/total_count*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"  Train: {OUTPUT_TRAIN_JSON}")
    print(f"  Val:   {OUTPUT_VAL_JSON}")
    print(f"  Test:  {OUTPUT_TEST_JSON}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

