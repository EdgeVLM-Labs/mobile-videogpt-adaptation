"""
Features:
- Randomly downloads N videos per exercise subfolder
- Preserves folder structure
- Downloads fine_grained_labels.json (complete ground truth)
- Creates manifest.json of downloaded videos
"""

import os
import random
import json
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
import shutil
import time

REPO_ID = "EdgeVLM-Labs/QVED-Test-Dataset"
LOCAL_DIR = Path("dataset")  # local download directory
MAX_PER_CLASS = 5
FILE_EXT = ".mp4"
GROUND_TRUTH_FILE = "fine_grained_labels.json"
RANDOM_SEED = 42


def collect_videos(repo_id):
    """Collects all video files grouped by class (subfolder)."""

    print(f"📂 Listing repo files from: {repo_id}")
    all_files = list_repo_files(repo_id, repo_type="dataset")
    by_class = {}
    for f in all_files:
        if not f.endswith(FILE_EXT):
            continue
        parts = f.split("/")
        if len(parts) < 2:
            continue
        cls = parts[0]
        by_class.setdefault(cls, []).append(f)
    print(f"✅ Found {len(by_class)} classes with video files.")
    return by_class, all_files


def sample_and_download(by_class, repo_id, local_dir, max_per_class):
    """Samples random videos per class and downloads them into <local_dir>/<class>/<file> (no duplicate subfolders)."""

    random.seed(RANDOM_SEED)
    manifest = {}
    total_downloaded = 0

    for cls, vids in by_class.items():
        class_dir = local_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        sample = random.sample(vids, min(len(vids), max_per_class))
        print(f"🎥 {cls}: {len(sample)} sampled of {len(vids)} available")

        for rel_path in sample:
            filename = os.path.basename(rel_path)  # e.g., "00018209.mp4"
            target_path = class_dir / filename

            while True:
                try:
                    cached_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel_path,
                        repo_type="dataset",
                    )

                    shutil.copy2(cached_path, target_path)

                    manifest[str(target_path)] = cls
                    total_downloaded += 1
                    break
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        print(f"⚠️ Rate limit hit (429). Waiting ~ 3 minutes before retrying {rel_path}...")
                        time.sleep(200)
                    else:
                        print(f"⚠️ Failed to download {rel_path}: {e}")
                        break

    print(f"\n✅ Download complete: {total_downloaded} videos.")
    return manifest


def download_ground_truth(repo_id, local_dir, all_files):
    """Downloads fine_grained_labels.json if present."""

    candidates = [f for f in all_files if f.endswith(GROUND_TRUTH_FILE)]
    if not candidates:
        print(f"⚠️ No {GROUND_TRUTH_FILE} found in repo.")
        return None

    gt_path = local_dir / GROUND_TRUTH_FILE
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=candidates[0],
            local_dir=str(local_dir),
            repo_type="dataset",
        )
        print(f"🧠 Ground truth file downloaded to: {gt_path}")
        return gt_path
    except Exception as e:
        print(f"⚠️ Failed to download {GROUND_TRUTH_FILE}: {e}")
        return None


def save_manifest(manifest, local_dir):
    """Saves manifest.json mapping downloaded videos to their class."""

    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest saved to: {manifest_path}")
    return manifest_path


def main():

    max_per_class = MAX_PER_CLASS
    if len(sys.argv) > 1:
        try:
            max_per_class = int(sys.argv[1])
            print(f"📊 Using MAX_PER_CLASS = {max_per_class} (from command line)")
        except ValueError:
            print(f"⚠️ Invalid argument. Using default MAX_PER_CLASS = {MAX_PER_CLASS}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    by_class, all_files = collect_videos(REPO_ID)
    manifest = sample_and_download(by_class, REPO_ID, LOCAL_DIR, max_per_class)
    save_manifest(manifest, LOCAL_DIR)
    download_ground_truth(REPO_ID, LOCAL_DIR, all_files)
    print("🏁 Dataset download completed.")


if __name__ == "__main__":
    main()
