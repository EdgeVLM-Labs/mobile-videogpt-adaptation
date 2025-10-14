import json
import os
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("dataset")
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"
MANIFEST_JSON = BASE_DIR / "manifest.json"
OUTPUT_JSON = BASE_DIR / "qved_train.json"
USER_PROMPT_TEMPLATE = "Analyze this {exercise} video and provide corrective feedback."

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
        filename_to_exercise[filename] = exercise.replace('_', ' ')

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

        user_prompt = USER_PROMPT_TEMPLATE.format(exercise=exercise)

        output_data.append({
            "video": full_video_path,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": assistant_answer}
            ],
            "split": "train"
        })

    # Write output JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Converted {len(output_data)} videos from {len(set(filename_to_exercise.values()))} exercise classes")
    print(f"Output saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
