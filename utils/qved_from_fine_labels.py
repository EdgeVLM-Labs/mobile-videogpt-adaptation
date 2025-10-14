import json
import os
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path("dataset")
FINE_LABELS_JSON = BASE_DIR / "ground_truth.json"
OUTPUT_JSON = BASE_DIR / "qved_train.json"
VIDEO_ROOT = BASE_DIR  # subfolders per exercise
USER_PROMPT_TEMPLATE = "Analyze this {exercise} video and provide corrective feedback."

def main():
    # Load fine-grained labels
    with open(FINE_LABELS_JSON, 'r') as f:
        fine_labels = json.load(f)

    # Group by exercise class
    exercise_groups = defaultdict(list)

    for record in fine_labels:
        video_path = record.get('video_path', '')
        # Extract exercise name from folder or metadata
        if 'exercise' in record:
            exercise = record['exercise']
        else:
            # Derive from path: e.g., "exercise_name/video.mp4" -> "exercise_name"
            parts = Path(video_path).parts
            if len(parts) > 1:
                exercise = parts[-2]  # Parent folder name
            else:
                exercise = "unknown"

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

        exercise_groups[exercise].append({
            'video_path': video_path,
            'exercise': exercise,
            'assistant': assistant_answer
        })

    # Convert to Mobile-VideoGPT format
    output_data = []
    for exercise, records in exercise_groups.items():
        for item in records:
            # Make video path relative to repo root
            video_rel = item['video_path']
            if os.path.isabs(video_rel):
                # Convert absolute to relative if needed
                video_rel = os.path.relpath(video_rel, os.getcwd())

            user_prompt = USER_PROMPT_TEMPLATE.format(exercise=item['exercise'])

            output_data.append({
                "video": video_rel,
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": item['assistant']}
                ],
                "split": "train"
            })

    # Write output JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Converted {len(output_data)} videos from {len(exercise_groups)} exercise classes")
    print(f"Output saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
