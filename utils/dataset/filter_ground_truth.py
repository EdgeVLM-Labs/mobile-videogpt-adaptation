"""
Filters fine_grained_labels.json to include only
ground truths of videos actually downloaded
by load_dataset.py.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path("dataset")
GROUND_TRUTH_FILE = BASE_DIR / "fine_grained_labels.json"
FEEDBACK_FILE = BASE_DIR / "feedbacks_short_clips.json"
MANIFEST_FILE = BASE_DIR / "manifest.json"
OUTPUT_FILE = BASE_DIR / "ground_truth.json"
FEEDBACK_OUTPUT_FILE = BASE_DIR / "feedbacks.json"

def filter_json_file(input_file, output_file, downloaded_filenames, file_label):
    """Filters a JSON file to include only entries for downloaded videos."""
    if not input_file.exists():
        print(f"⚠️ {input_file.name} not found. Skipping.")
        return

    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"🧠 Filtering {len(data)} {file_label} entries...")
    filtered = [
        item for item in data
        if "video_path" in item and Path(item["video_path"]).name in downloaded_filenames
    ]

    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"✅ Filtered {file_label}: {len(filtered)} entries")
    print(f"📝 Saved to: {output_file}")


def main():

    if not MANIFEST_FILE.exists():
        print("⚠️ Manifest file missing. Please run load_dataset.py first.")
        return

    # Load manifest (downloaded files)
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    downloaded_filenames = {Path(p).name for p in manifest.keys()}

    # Filter both JSON files
    filter_json_file(GROUND_TRUTH_FILE, OUTPUT_FILE, downloaded_filenames, "ground truth")
    filter_json_file(FEEDBACK_FILE, FEEDBACK_OUTPUT_FILE, downloaded_filenames, "feedback")

if __name__ == "__main__":
    main()
