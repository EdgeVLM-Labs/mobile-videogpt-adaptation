"""
Features:
- Randomly downloads N videos per exercise subfolder
- Preserves folder structure
- Downloads fine_grained_labels.json (complete ground truth)
- Creates manifest.json of downloaded videos
- Optional parallel downloads for faster processing
"""

import os
import random
import json
import sys
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

REPO_ID = "EdgeVLM-Labs/QEVD-fine-grained-feedback-cleaned"
LOCAL_DIR = Path("dataset")  # local download directory
MAX_PER_CLASS = 5
FILE_EXT = ".mp4"
GROUND_TRUTH_FILE = "fine_grained_labels.json"
FEEDBACKS_FILE = "feedbacks_short_clips.json"
RANDOM_SEED = 42
MAX_WORKERS = 4  # Number of parallel download threads


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


def download_single_file(repo_id, rel_path, target_path, cls):
    """Download a single file with retry logic. Returns (target_path, cls, success)."""

    while True:
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
            )

            shutil.copy2(cached_path, target_path)
            return (str(target_path), cls, True)
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                print(f"⚠️ Rate limit hit (429). Waiting ~ 3 minutes before retrying {rel_path}...")
                time.sleep(200)
            else:
                print(f"⚠️ Failed to download {rel_path}: {e}")
                return (str(target_path), cls, False)


def sample_and_download(by_class, repo_id, local_dir, max_per_class, parallel=False):
    """Samples random videos per class and downloads them into <local_dir>/<class>/<file> (no duplicate subfolders)."""

    random.seed(RANDOM_SEED)
    manifest = {}
    total_downloaded = 0

    # Prepare all download tasks
    download_tasks = []
    for cls, vids in by_class.items():
        class_dir = local_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        sample = random.sample(vids, min(len(vids), max_per_class))
        print(f"🎥 {cls}: {len(sample)} sampled of {len(vids)} available")

        for rel_path in sample:
            filename = os.path.basename(rel_path)
            target_path = class_dir / filename
            download_tasks.append((repo_id, rel_path, target_path, cls))

    if parallel:
        # Parallel download
        print(f"⚡ Using parallel downloads with {MAX_WORKERS} workers")
        manifest_lock = Lock()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_single_file, repo_id, rel_path, target_path, cls): (rel_path, cls)
                for repo_id, rel_path, target_path, cls in download_tasks
            }

            for future in as_completed(futures):
                target_path, cls, success = future.result()
                if success:
                    with manifest_lock:
                        manifest[target_path] = cls
                        total_downloaded += 1
                        print(f"✓ Downloaded {os.path.basename(target_path)} ({total_downloaded}/{len(download_tasks)})")
    else:
        # Sequential download (default)
        print(f"📥 Using sequential downloads")
        for repo_id, rel_path, target_path, cls in download_tasks:
            target_path_str, cls, success = download_single_file(repo_id, rel_path, target_path, cls)
            if success:
                manifest[target_path_str] = cls
                total_downloaded += 1

    print(f"\n✅ Download complete: {total_downloaded} videos.")
    return manifest


def download_json_file(repo_id, local_dir, all_files, json_filename):
    """Downloads a JSON file if present."""

    candidates = [f for f in all_files if f.endswith(json_filename)]
    if not candidates:
        print(f"⚠️ No {json_filename} found in repo.")
        return None

    json_path = local_dir / json_filename
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=candidates[0],
            local_dir=str(local_dir),
            repo_type="dataset",
        )
        print(f"📄 {json_filename} downloaded to: {json_path}")
        return json_path
    except Exception as e:
        print(f"⚠️ Failed to download {json_filename}: {e}")
        return None


def download_ground_truth(repo_id, local_dir, all_files):
    """Downloads fine_grained_labels.json if present."""
    return download_json_file(repo_id, local_dir, all_files, GROUND_TRUTH_FILE)


def download_feedbacks(repo_id, local_dir, all_files):
    """Downloads feedbacks_short_clips.json if present."""
    return download_json_file(repo_id, local_dir, all_files, FEEDBACKS_FILE)


def save_manifest(manifest, local_dir):
    """Saves manifest.json mapping downloaded videos to their class."""

    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest saved to: {manifest_path}")
    return manifest_path


def main():
    max_per_class = MAX_PER_CLASS
    parallel = False

    # Parse command line arguments
    # Usage: python load_dataset.py [max_per_class] [--parallel]
    args = sys.argv[1:]

    for arg in args:
        if arg == "--parallel":
            parallel = True
            print(f"⚡ Parallel download mode enabled")
        else:
            try:
                max_per_class = int(arg)
                print(f"📊 Using MAX_PER_CLASS = {max_per_class} (from command line)")
            except ValueError:
                print(f"⚠️ Invalid argument '{arg}'. Using default MAX_PER_CLASS = {MAX_PER_CLASS}")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    by_class, all_files = collect_videos(REPO_ID)
    manifest = sample_and_download(by_class, REPO_ID, LOCAL_DIR, max_per_class, parallel=parallel)
    save_manifest(manifest, LOCAL_DIR)

    # Download JSON files
    download_ground_truth(REPO_ID, LOCAL_DIR, all_files)
    download_feedbacks(REPO_ID, LOCAL_DIR, all_files)

    print("🏁 Dataset download completed.")


if __name__ == "__main__":
    main()
