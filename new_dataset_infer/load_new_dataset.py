"""
Download and extract exercise videos from tar file for inference.

Features:
- Downloads exercise_videos.tar from HuggingFace
- Extracts all videos (no sampling)
- Preserves folder structure: dataset/exercise_class/video.mov
- Creates manifest.json mapping video paths to exercise classes
- Handles both .mov and .mp4 video formats
"""

import os
import json
import tarfile
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

REPO_ID = "EdgeVLM-Labs/physio-exercise-videos"
TAR_FILE = "exercise_videos.tar"
LOCAL_DIR = Path("dataset")  # local download directory
VIDEO_EXTENSIONS = [".mov", ".mp4"]


def download_tar_file(repo_id, tar_filename):
    """Downloads the tar file from HuggingFace."""
    
    print(f"📦 Downloading {tar_filename} from {repo_id}...")
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=tar_filename,
            repo_type="dataset",
        )
        print(f"✅ Downloaded to cache: {cached_path}")
        return cached_path
    except Exception as e:
        print(f"❌ Failed to download {tar_filename}: {e}")
        return None


def extract_tar_file(tar_path, extract_to):
    """Extracts tar file to the specified directory."""
    
    print(f"📂 Extracting {tar_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=extract_to)
        print(f"✅ Extraction complete")
        return True
    except Exception as e:
        print(f"❌ Failed to extract tar file: {e}")
        return False


def scan_videos(base_dir):
    """Scans the extracted directory for video files and organizes by class."""
    
    print(f"🔍 Scanning for video files in {base_dir}...")
    manifest = {}
    video_count = 0
    class_counts = {}
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if file is a video
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                full_path = Path(root) / file
                
                # Determine the exercise class (parent folder name)
                relative_path = full_path.relative_to(base_dir)
                
                if len(relative_path.parts) >= 2:
                    # Format: exercise_class/video.mov
                    exercise_class = relative_path.parts[0]
                else:
                    # Video is directly in base_dir (no subfolder)
                    exercise_class = "unknown"
                
                # Add to manifest
                manifest[str(full_path)] = exercise_class
                class_counts[exercise_class] = class_counts.get(exercise_class, 0) + 1
                video_count += 1
    
    print(f"✅ Found {video_count} videos across {len(class_counts)} exercise classes:")
    for cls, count in sorted(class_counts.items()):
        print(f"   📹 {cls}: {count} videos")
    
    return manifest


def save_manifest(manifest, local_dir):
    """Saves manifest.json mapping video paths to their class."""
    
    manifest_path = local_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📝 Manifest saved to: {manifest_path}")
    return manifest_path


def cleanup_temp_files(tar_path):
    """Optional: Remove the downloaded tar file to save space."""
    
    print(f"🗑️  Cleaning up temporary files...")
    try:
        if Path(tar_path).exists():
            Path(tar_path).unlink()
            print(f"✅ Removed {tar_path}")
    except Exception as e:
        print(f"⚠️  Could not remove {tar_path}: {e}")


def main():
    print("=" * 50)
    print("  Exercise Video Dataset Downloader (Inference)")
    print("=" * 50)
    print()
    
    # Parse command line arguments
    # Usage: python load_new_dataset.py [--keep-tar]
    args = sys.argv[1:]
    keep_tar = "--keep-tar" in args
    
    # Step 1: Download tar file
    tar_path = download_tar_file(REPO_ID, TAR_FILE)
    if not tar_path:
        print("❌ Download failed. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 2: Extract tar file
    success = extract_tar_file(tar_path, LOCAL_DIR)
    if not success:
        print("❌ Extraction failed. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 3: Scan and catalog videos
    manifest = scan_videos(LOCAL_DIR)
    
    if not manifest:
        print("⚠️  No videos found in extracted archive!")
        sys.exit(1)
    
    print()
    
    # Step 4: Save manifest
    save_manifest(manifest, LOCAL_DIR)
    
    print()
    
    # Step 5: Optional cleanup
    if not keep_tar:
        cleanup_temp_files(tar_path)
    else:
        print(f"📦 Keeping tar file at: {tar_path}")
    
    print()
    print("=" * 50)
    print("  ✅ Dataset Download Complete!")
    print("=" * 50)
    print()
    print(f"Videos extracted to: {LOCAL_DIR.absolute()}")
    print(f"Manifest saved to: {(LOCAL_DIR / 'manifest.json').absolute()}")
    print()
    print("You can now run inference on these videos!")


if __name__ == "__main__":
    main()
