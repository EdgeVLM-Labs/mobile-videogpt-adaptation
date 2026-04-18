import json
import os
import random
from pathlib import Path

# Paths
BASE_DIR = Path("dataset")
FINE_GRAINED_JSON = BASE_DIR / "fine_grained_labels.json"
FEEDBACKS_JSON = BASE_DIR / "feedbacks_short_clips.json"
MANIFEST_JSON = BASE_DIR / "manifest.json"
OUTPUT_TRAIN_JSON = BASE_DIR / "qved_train.json"
OUTPUT_VAL_JSON = BASE_DIR / "qved_val.json"
OUTPUT_TEST_JSON = BASE_DIR / "qved_test.json"
OUTPUT_FEEDBACKS_TRAIN_JSON = BASE_DIR / "qved_feedbacks_train.json"
OUTPUT_FEEDBACKS_VAL_JSON = BASE_DIR / "qved_feedbacks_val.json"
OUTPUT_FEEDBACKS_TEST_JSON = BASE_DIR / "qved_feedbacks_test.json"
# USER_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
USER_PROMPT_TEMPLATE = "Watch the exercise being performed and provide short corrective feedback to help improve the form."
FEEDBACK_PROMPT_TEMPLATE = "Please evaluate the exercise form shown. What feedback would you provide to improve the performance?"

# The JSON attribute to use as the ground truth answer (e.g., "feedback", "labels_descriptive", "coach")
GROUND_TRUTH_ATTRIBUTE = "labels_descriptive"

# Dataset split ratios (adjustable)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42  # For reproducibility

def load_manifest():
    """Load manifest and create lookup dictionaries"""
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

    return filename_to_path, filename_to_exercise

def process_fine_grained_labels(fine_labels_path, filename_to_path, filename_to_exercise):
    """Process fine-grained labels JSON file"""
    if not fine_labels_path.exists():
        print(f"Warning: {fine_labels_path} not found, skipping fine-grained labels processing")
        return []

    with open(fine_labels_path, 'r') as f:
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

        # Get assistant answer in "exercise - feedback" format
        ground_truth = record.get(GROUND_TRUTH_ATTRIBUTE, '')
        if ground_truth:
            # Strip exercise prefix from each item and collect unique descriptions
            prefix = f"{exercise} - ".lower()
            if isinstance(ground_truth, list):
                descriptions = []
                for item in ground_truth:
                    item_str = str(item).strip()
                    if item_str.lower().startswith(prefix):
                        item_str = item_str[len(prefix):].strip()
                    descriptions.append(item_str)
                assistant_answer = f"{exercise} - {', '.join(descriptions)}"
            else:
                ground_truth = str(ground_truth).strip()
                if ground_truth.lower().startswith(prefix):
                    ground_truth = ground_truth[len(prefix):].strip()
                assistant_answer = f"{exercise} - {ground_truth}"
        else:
            assistant_answer = f"{exercise} - No feedback available."

        user_prompt = USER_PROMPT_TEMPLATE  # No longer using exercise name in prompt

        output_data.append({
            "video": relative_video_path,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train",  # Will be updated during split
            "filename": filename  # Keep filename for cross-reference
        })

    return output_data

def process_feedbacks(feedbacks_path, filename_to_path, filename_to_exercise):
    """Process feedbacks JSON file"""
    if not feedbacks_path.exists():
        print(f"Warning: {feedbacks_path} not found, skipping feedbacks processing")
        return []

    with open(feedbacks_path, 'r') as f:
        feedbacks = json.load(f)

    # Convert to Mobile-VideoGPT format
    output_data = []

    for record in feedbacks:
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
        if full_video_path.startswith('dataset/'):
            relative_video_path = full_video_path[len('dataset/'):]
        else:
            relative_video_path = full_video_path

        # Get feedbacks and join them
        feedbacks_list = record.get('feedbacks', [])
        if isinstance(feedbacks_list, list):
            assistant_answer = '\n'.join(str(item) for item in feedbacks_list)
        else:
            assistant_answer = str(feedbacks_list)

        if not assistant_answer.strip():
            assistant_answer = "No feedback available."

        user_prompt = FEEDBACK_PROMPT_TEMPLATE

        output_data.append({
            "video": relative_video_path,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train",  # Will be updated during split
            "filename": filename  # Keep filename for cross-reference
        })

    return output_data

def split_data_consistently(data1, data2):
    """
    Split two datasets consistently - same videos go to same splits in same order.
    Returns (train1, val1, test1, train2, val2, test2)
    """
    # Create filename to index mappings
    filename_to_idx1 = {item['filename']: idx for idx, item in enumerate(data1)}
    filename_to_idx2 = {item['filename']: idx for idx, item in enumerate(data2)}

    # Find common filenames
    common_filenames = set(filename_to_idx1.keys()) & set(filename_to_idx2.keys())

    if not common_filenames:
        print("Warning: No common videos found between the two datasets")
        return [], [], [], [], [], []

    # Create paired list with common videos
    common_list = list(common_filenames)

    # Shuffle for random split
    random.seed(RANDOM_SEED)
    random.shuffle(common_list)

    # Calculate split indices
    total_count = len(common_list)
    train_end = int(total_count * TRAIN_RATIO)
    val_end = train_end + int(total_count * VAL_RATIO)

    # Split filenames
    train_filenames = common_list[:train_end]
    val_filenames = common_list[train_end:val_end]
    test_filenames = common_list[val_end:]

    # Create split datasets maintaining order
    def create_split(filenames, dataset, filename_to_idx, split_name):
        split_data = []
        for filename in filenames:
            idx = filename_to_idx[filename]
            item = dataset[idx].copy()
            item['split'] = split_name
            # Remove filename from output (it was only for cross-reference)
            item.pop('filename', None)
            split_data.append(item)
        return split_data

    train_data1 = create_split(train_filenames, data1, filename_to_idx1, "train")
    val_data1 = create_split(val_filenames, data1, filename_to_idx1, "val")
    test_data1 = create_split(test_filenames, data1, filename_to_idx1, "test")

    train_data2 = create_split(train_filenames, data2, filename_to_idx2, "train")
    val_data2 = create_split(val_filenames, data2, filename_to_idx2, "val")
    test_data2 = create_split(test_filenames, data2, filename_to_idx2, "test")

    return train_data1, val_data1, test_data1, train_data2, val_data2, test_data2

def main():
    # Load manifest
    filename_to_path, filename_to_exercise = load_manifest()

    # Process both datasets
    fine_data = []
    feedbacks_data = []

    if FINE_GRAINED_JSON.exists():
        fine_data = process_fine_grained_labels(FINE_GRAINED_JSON, filename_to_path, filename_to_exercise)
        print(f"Using fine-grained labels from: {FINE_GRAINED_JSON}")

    if FEEDBACKS_JSON.exists():
        feedbacks_data = process_feedbacks(FEEDBACKS_JSON, filename_to_path, filename_to_exercise)
        print(f"Using feedbacks from: {FEEDBACKS_JSON}")

    # Check if we have data to process
    if not fine_data and not feedbacks_data:
        print("Error: No input data files found. Need at least one of:")
        print(f"  - {FINE_GRAINED_JSON}")
        print(f"  - {FEEDBACKS_JSON}")
        return

    # If we have both datasets, split them consistently
    if fine_data and feedbacks_data:
        train_fine, val_fine, test_fine, train_fb, val_fb, test_fb = split_data_consistently(fine_data, feedbacks_data)

        # Write fine-grained labels splits
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_TRAIN_JSON, 'w') as f:
            json.dump(train_fine, f, indent=2)
        with open(OUTPUT_VAL_JSON, 'w') as f:
            json.dump(val_fine, f, indent=2)
        with open(OUTPUT_TEST_JSON, 'w') as f:
            json.dump(test_fine, f, indent=2)

        # Write feedbacks splits
        with open(OUTPUT_FEEDBACKS_TRAIN_JSON, 'w') as f:
            json.dump(train_fb, f, indent=2)
        with open(OUTPUT_FEEDBACKS_VAL_JSON, 'w') as f:
            json.dump(val_fb, f, indent=2)
        with open(OUTPUT_FEEDBACKS_TEST_JSON, 'w') as f:
            json.dump(test_fb, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Dataset Split Summary (Both Datasets)")
        print(f"{'='*60}")
        print(f"Total common videos: {len(train_fine) + len(val_fine) + len(test_fine)}")
        print(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
        print(f"\nFine-Grained Labels Split:")
        print(f"  Train: {len(train_fine)} samples ({len(train_fine)/(len(train_fine)+len(val_fine)+len(test_fine))*100:.1f}%)")
        print(f"  Val:   {len(val_fine)} samples ({len(val_fine)/(len(train_fine)+len(val_fine)+len(test_fine))*100:.1f}%)")
        print(f"  Test:  {len(test_fine)} samples ({len(test_fine)/(len(train_fine)+len(val_fine)+len(test_fine))*100:.1f}%)")
        print(f"\nFeedbacks Split:")
        print(f"  Train: {len(train_fb)} samples ({len(train_fb)/(len(train_fb)+len(val_fb)+len(test_fb))*100:.1f}%)")
        print(f"  Val:   {len(val_fb)} samples ({len(val_fb)/(len(train_fb)+len(val_fb)+len(test_fb))*100:.1f}%)")
        print(f"  Test:  {len(test_fb)} samples ({len(test_fb)/(len(train_fb)+len(val_fb)+len(test_fb))*100:.1f}%)")
        print(f"\nFine-Grained Labels Output files:")
        print(f"  Train: {OUTPUT_TRAIN_JSON}")
        print(f"  Val:   {OUTPUT_VAL_JSON}")
        print(f"  Test:  {OUTPUT_TEST_JSON}")
        print(f"\nFeedbacks Output files:")
        print(f"  Train: {OUTPUT_FEEDBACKS_TRAIN_JSON}")
        print(f"  Val:   {OUTPUT_FEEDBACKS_VAL_JSON}")
        print(f"  Test:  {OUTPUT_FEEDBACKS_TEST_JSON}")
        print(f"{'='*60}")

    # If we only have one dataset, use the original splitting logic
    elif fine_data:

        # Remove filename field before processing
        for item in fine_data:
            item.pop('filename', None)

        # Shuffle data for random split
        random.seed(RANDOM_SEED)
        random.shuffle(fine_data)

        # Calculate split indices
        total_count = len(fine_data)
        train_end = int(total_count * TRAIN_RATIO)
        val_end = train_end + int(total_count * VAL_RATIO)

        # Split the data
        train_data = fine_data[:train_end]
        val_data = fine_data[train_end:val_end]
        test_data = fine_data[val_end:]

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
        print(f"Dataset Split Summary (Fine-Grained Labels Only)")
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

    elif feedbacks_data:
        # Remove filename field before processing
        for item in feedbacks_data:
            item.pop('filename', None)

        # Shuffle data for random split
        random.seed(RANDOM_SEED)
        random.shuffle(feedbacks_data)

        # Calculate split indices
        total_count = len(feedbacks_data)
        train_end = int(total_count * TRAIN_RATIO)
        val_end = train_end + int(total_count * VAL_RATIO)

        # Split the data
        train_data = feedbacks_data[:train_end]
        val_data = feedbacks_data[train_end:val_end]
        test_data = feedbacks_data[val_end:]

        # Update split labels
        for item in train_data:
            item["split"] = "train"
        for item in val_data:
            item["split"] = "val"
        for item in test_data:
            item["split"] = "test"

        # Write output JSONs
        BASE_DIR.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FEEDBACKS_TRAIN_JSON, 'w') as f:
            json.dump(train_data, f, indent=2)

        with open(OUTPUT_FEEDBACKS_VAL_JSON, 'w') as f:
            json.dump(val_data, f, indent=2)

        with open(OUTPUT_FEEDBACKS_TEST_JSON, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Dataset Split Summary (Feedbacks Only)")
        print(f"{'='*60}")
        print(f"Total videos: {total_count}")
        print(f"Exercise classes: {len(set(filename_to_exercise.values()))}")
        print(f"\nSplit Distribution:")
        print(f"  Train: {len(train_data)} samples ({len(train_data)/total_count*100:.1f}%)")
        print(f"  Val:   {len(val_data)} samples ({len(val_data)/total_count*100:.1f}%)")
        print(f"  Test:  {len(test_data)} samples ({len(test_data)/total_count*100:.1f}%)")
        print(f"\nOutput files:")
        print(f"  Train: {OUTPUT_FEEDBACKS_TRAIN_JSON}")
        print(f"  Val:   {OUTPUT_FEEDBACKS_VAL_JSON}")
        print(f"  Test:  {OUTPUT_FEEDBACKS_TEST_JSON}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

