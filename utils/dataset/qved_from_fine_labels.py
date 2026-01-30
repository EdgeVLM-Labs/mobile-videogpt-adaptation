import json
import os
import random
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("dataset")
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"
FEEDBACK_JSON = BASE_DIR / "feedbacks.json"
MANIFEST_JSON = BASE_DIR / "manifest.json"
OUTPUT_TRAIN_JSON = BASE_DIR / "qved_train.json"
OUTPUT_VAL_JSON = BASE_DIR / "qved_val.json"
OUTPUT_TEST_JSON = BASE_DIR / "qved_test.json"
FEEDBACK_TRAIN_JSON = BASE_DIR / "feedback_train.json"
FEEDBACK_VAL_JSON = BASE_DIR / "feedback_val.json"
FEEDBACK_TEST_JSON = BASE_DIR / "feedback_test.json"
USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

# Dataset split ratios (adjustable)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42  # For reproducibility

def process_labels(records, filename_to_path, label_key):
    """Convert label records to output format."""
    output_data = []

    for record in records:
        video_path = record.get('video_path', '')
        filename = os.path.basename(video_path)

        if filename not in filename_to_path:
            print(f"Warning: {filename} not found in manifest, skipping")
            continue

        full_video_path = filename_to_path[filename]

        # Remove 'dataset/' prefix if present
        if full_video_path.startswith('dataset/'):
            relative_video_path = full_video_path[len('dataset/'):]
        else:
            relative_video_path = full_video_path

        # Get assistant answer based on label_key
        if label_key in record and record[label_key]:
            assistant_answer = record[label_key]
        else:
            assistant_answer = "No feedback available."

        # Handle list-type answers
        if isinstance(assistant_answer, list):
            assistant_answer = '\n'.join(str(item) for item in assistant_answer)
        else:
            assistant_answer = str(assistant_answer)

        output_data.append({
            "video": relative_video_path,
            "conversations": [
                {"from": "human", "value": USER_PROMPT_TEMPLATE},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train"  # Will be updated during split
        })

    return output_data


def split_and_assign(output_data, video_to_split=None):
    """Split data into train/val/test or assign splits from mapping."""
    if video_to_split is None:
        # Create new splits
        random.seed(RANDOM_SEED)
        random.shuffle(output_data)

        total_count = len(output_data)
        train_end = int(total_count * TRAIN_RATIO)
        val_end = train_end + int(total_count * VAL_RATIO)

        train_data = output_data[:train_end]
        val_data = output_data[train_end:val_end]
        test_data = output_data[val_end:]

        for item in train_data:
            item["split"] = "train"
        for item in val_data:
            item["split"] = "val"
        for item in test_data:
            item["split"] = "test"
    else:
        # Use existing split mapping
        train_data, val_data, test_data = [], [], []

        for item in output_data:
            filename = os.path.basename(item["video"])
            item["split"] = video_to_split.get(filename, "train")

            if item["split"] == "train":
                train_data.append(item)
            elif item["split"] == "val":
                val_data.append(item)
            else:
                test_data.append(item)

    return train_data, val_data, test_data


def save_splits(train_data, val_data, test_data, train_file, val_file, test_file):
    """Save train/val/test splits to JSON files."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)

    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)


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

    # Process ground truth labels
    with open(FINE_LABELS_JSON, 'r') as f:
        fine_labels = json.load(f)

    output_data = process_labels(fine_labels, filename_to_path, 'labels_descriptive')
    total_count = len(output_data)
    train_data, val_data, test_data = split_and_assign(output_data)

    # Create video-to-split mapping for feedback processing
    video_to_split = {os.path.basename(item["video"]): item["split"]
                      for item in train_data + val_data + test_data}

    # Process feedback JSON with same splits
    feedback_train = feedback_val = feedback_test = []
    if FEEDBACK_JSON.exists():
        with open(FEEDBACK_JSON, 'r') as f:
            feedback_labels = json.load(f)

        feedback_data = process_labels(feedback_labels, filename_to_path, 'feedbacks')
        feedback_train, feedback_val, feedback_test = split_and_assign(feedback_data, video_to_split)

    # Save all splits
    save_splits(train_data, val_data, test_data, OUTPUT_TRAIN_JSON, OUTPUT_VAL_JSON, OUTPUT_TEST_JSON)

    if FEEDBACK_JSON.exists():
        save_splits(feedback_train, feedback_val, feedback_test,
                   FEEDBACK_TRAIN_JSON, FEEDBACK_VAL_JSON, FEEDBACK_TEST_JSON)

    print(f"\n{'='*60}")
    print(f"Dataset Split Summary")
    print(f"{'='*60}")
    print(f"Total videos: {total_count}")
    print(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
    print(f"\nGround Truth Split Distribution:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total_count*100:.1f}%)")
    print(f"  Val:   {len(val_data)} samples ({len(val_data)/total_count*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/total_count*100:.1f}%)")
    print(f"\nGround Truth Output files:")
    print(f"  Train: {OUTPUT_TRAIN_JSON}")
    print(f"  Val:   {OUTPUT_VAL_JSON}")
    print(f"  Test:  {OUTPUT_TEST_JSON}")

    if FEEDBACK_JSON.exists():
        feedback_total = len(feedback_train) + len(feedback_val) + len(feedback_test)
        print(f"\nFeedback Split Distribution:")
        print(f"  Train: {len(feedback_train)} samples")
        print(f"  Val:   {len(feedback_val)} samples")
        print(f"  Test:  {len(feedback_test)} samples")
        print(f"\nFeedback Output files:")
        print(f"  Train: {FEEDBACK_TRAIN_JSON}")
        print(f"  Val:   {FEEDBACK_VAL_JSON}")
        print(f"  Test:  {FEEDBACK_TEST_JSON}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()

