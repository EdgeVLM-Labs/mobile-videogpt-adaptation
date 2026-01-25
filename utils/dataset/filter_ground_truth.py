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
MANIFEST_FILE = BASE_DIR / "manifest.json"
OUTPUT_FILE = BASE_DIR / "ground_truth.json"

def main():

    if not GROUND_TRUTH_FILE.exists() or not MANIFEST_FILE.exists():
        print("⚠️ Required files missing. Please run DownloadDataset.py first.")
        return

    # Load manifest (downloaded files)
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    downloaded_filenames = {Path(p).name for p in manifest.keys()}

    # Load fine_grained_labels.json (complete)
    with open(GROUND_TRUTH_FILE, "r") as f:
        gt_data = json.load(f)

    # Filter ground truths
    print(f"🧠 Filtering {len(gt_data)} ground truth entries...")
    filtered = [
        item for item in gt_data
        if "video_path" in item and Path(item["video_path"]).name in downloaded_filenames
    ]

    # Save filtered file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"✅ Filtered ground truths: {len(filtered)} entries")
    print(f"📝 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
