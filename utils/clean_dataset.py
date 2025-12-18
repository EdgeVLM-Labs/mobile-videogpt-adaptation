"""
Features:
- Checks video quality (resolution, brightness, sharpness, motion)
- Copies only good videos to a cleaned dataset
- Keeps folder structure
- Creates:
    - cleaning_report.csv  → Summary per exercise
    - exercise_analysis_report.csv  → Detailed per-file metrics, motion stats, and rejection reasons
"""

import cv2
import numpy as np
import shutil
import pandas as pd
import os
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import json

DATASET_PATH = Path("dataset")
CLEANED_DATASET_PATH = Path("cleaned_dataset")

MIN_FRAME_WIDTH = 640
MIN_FRAME_HEIGHT = 360
MIN_SHARPNESS_SCORE = 50
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 190
NUM_SAMPLED_FRAMES = 20
FRAME_STRIDE = 15

MOTION_DIFF_THRESHOLD = 18
MOTION_MIN_PIXEL_CHANGE_RATIO = 0.01
MOTION_MIN_ACTIVE_FRAME_PCT = 0.3

MOTION_FLAGS_FILE = Path("utils/exercise_motion_overview.json")
if MOTION_FLAGS_FILE.exists():
    with open(MOTION_FLAGS_FILE, "r") as f:
        MOTION_FLAGS = json.load(f)
else:
    raise FileNotFoundError(f"Motion flags file not found: {MOTION_FLAGS_FILE}")

VIDEO_LOG = []
REJECTED_VIDEOS = {}


def ensure_directory_exists(directory: Path) -> None:
    """Create the directory if it doesn’t exist."""
    directory.mkdir(parents=True, exist_ok=True)


def copy_video_with_structure(source_file: Path, source_root: Path, destination_root: Path) -> None:
    """Copy a video file to the destination while preserving the folder structure."""
    relative_path = source_file.relative_to(source_root)
    destination_file = destination_root / relative_path
    ensure_directory_exists(destination_file.parent)
    shutil.copy2(source_file, destination_file)


def analyze_video_quality(video_path: Path, num_frames: int, frame_stride: int, exercise_name: str):
    """Analyze a video’s quality based on brightness, sharpness, and motion."""
    issues = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}, ["corrupted_file"]

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_brightness = []
    sampled_sharpness = []
    frame_index = 0
    samples_collected = 0

    prev_gray_for_motion = None
    active_count = 0
    motion_pairs = 0
    change_ratios = []

    motion_flag = MOTION_FLAGS.get(exercise_name, False)  # Check if motion analysis is needed for an exercise

    while samples_collected < num_frames and frame_index < max(frame_count, num_frames * frame_stride + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            frame_index += frame_stride
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sampled_brightness.append(float(gray.mean()))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sampled_sharpness.append(float(lap.var()))

        if motion_flag:
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray_for_motion is not None:
                diff = cv2.absdiff(gray_blur, prev_gray_for_motion)
                _, thresh = cv2.threshold(diff, MOTION_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                changed = np.count_nonzero(thresh)
                total = thresh.size
                ratio = changed / float(total)
                change_ratios.append(ratio)

                motion_pairs += 1
                if ratio >= MOTION_MIN_PIXEL_CHANGE_RATIO:
                    active_count += 1

            prev_gray_for_motion = gray_blur

        samples_collected += 1
        frame_index += frame_stride

    cap.release()

    if samples_collected == 0:
        issues.append("corrupted_file")

    motion_detected = False
    motion_active_frame_pct = float("nan")
    motion_mean_change_ratio = float("nan")
    motion_max_change_ratio = float("nan")

    if motion_flag and motion_pairs > 0:
        motion_active_frame_pct = active_count / motion_pairs
        motion_mean_change_ratio = float(np.mean(change_ratios)) if change_ratios else float("nan")
        motion_max_change_ratio = float(np.max(change_ratios)) if change_ratios else float("nan")
        motion_detected = motion_active_frame_pct >= MOTION_MIN_ACTIVE_FRAME_PCT

    metrics = {
        "width": width,
        "height": height,
        "mean_brightness": np.mean(sampled_brightness) if sampled_brightness else float("nan"),
        "sharpness_score": np.mean(sampled_sharpness) if sampled_sharpness else float("nan"),
        "motion_flag": motion_flag,
        "motion_detected": motion_detected,
        "motion_pairs": motion_pairs if motion_flag else 0,
        "motion_active_pairs": active_count if motion_flag else 0,
        "motion_active_frame_pct": motion_active_frame_pct,
        "motion_mean_change_ratio": motion_mean_change_ratio,
        "motion_max_change_ratio": motion_max_change_ratio,
    }
    return metrics, issues


def evaluate_video_acceptance(metrics: dict, issues: list, stats: dict):
    """Check if a video meets the minimum quality thresholds for acceptance and return (accepted, reasons)."""
    width, height = metrics.get("width", 0), metrics.get("height", 0)
    brightness = metrics.get("mean_brightness", float("nan"))
    sharpness = metrics.get("sharpness_score", float("nan"))
    motion_detected = metrics.get("motion_detected", False)
    motion_flag = metrics.get("motion_flag", False)

    reasons = []
    accepted = True

    if ("corrupted_file" in issues) or np.isnan(brightness) or np.isnan(sharpness):
        stats["corrupted_files"] += 1
        reasons.append("corrupted_file")
        return False, reasons

    if width < MIN_FRAME_WIDTH or height < MIN_FRAME_HEIGHT:
        stats["low_resolution"] += 1
        reasons.append("low_resolution")
        accepted = False

    if brightness < MIN_BRIGHTNESS:
        stats["too_dark"] += 1
        reasons.append("too_dark")
        accepted = False
    elif brightness > MAX_BRIGHTNESS:
        stats["too_bright"] += 1
        reasons.append("too_bright")
        accepted = False

    if sharpness < MIN_SHARPNESS_SCORE:
        stats["blurry"] += 1
        reasons.append("blurry")
        accepted = False

    if (not motion_detected) and motion_flag:
        stats["insufficient_motion"] += 1
        reasons.append("insufficient_motion")
        accepted = False

    return accepted, reasons


def default_stats(exercise_name: str) -> dict:
    """Return a default dictionary to track video quality statistics."""
    return {
            "Exercise": exercise_name,
            "total_videos": 0,
            "accepted_videos": 0,
            "rejected_videos": 0,
            "corrupted_files": 0,
            "low_resolution": 0,
            "too_dark": 0,
            "too_bright": 0,
            "blurry": 0,
            "insufficient_motion": 0,
    }


def generate_cleaning_report(overall_exercise_stats, totals, destination_root: Path = None):
    """Generate and optionally save a summary report of the dataset cleaning results."""
    if not overall_exercise_stats:
        print("\nNo exercise stats available.")
        return

    print("\n========== FINAL SUMMARY ==========")
    print(f"Total videos:    {totals['total_videos']}")
    print(f"Accepted videos: {totals['accepted_videos']}")
    print(f"Rejected videos: {totals['rejected_videos']}")
    print("===================================")

    df = pd.DataFrame(overall_exercise_stats)
    df = df[
        [
            "Exercise",
            "total_videos",
            "accepted_videos",
            "rejected_videos",
            "corrupted_files",
            "low_resolution",
            "too_dark",
            "too_bright",
            "blurry",
            "insufficient_motion",
        ]
    ]

    total_row = {
        "Exercise": "TOTAL",
        "total_videos": df["total_videos"].sum(),
        "accepted_videos": df["accepted_videos"].sum(),
        "rejected_videos": df["rejected_videos"].sum(),
        "corrupted_files": df["corrupted_files"].sum(),
        "low_resolution": df["low_resolution"].sum(),
        "too_dark": df["too_dark"].sum(),
        "too_bright": df["too_bright"].sum(),
        "blurry": df["blurry"].sum(),
        "insufficient_motion": df["insufficient_motion"].sum(),
    }

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    print("\n========== Exercise Breakdown ==========")
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    if destination_root is not None:
        report_dir = destination_root
        ensure_directory_exists(report_dir)

        csv_path = report_dir / "cleaning_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Report saved to: {csv_path}")

        if VIDEO_LOG:
            details_path = report_dir / "exercise_analysis_report.csv"
            df_details = pd.DataFrame(VIDEO_LOG)
            df_details.to_csv(details_path, index=False)
            print(f"[INFO] Detailed file analysis saved to: {details_path}")


def save_rejected_videos_json(destination_root: Path):
    """Save rejected video paths organized by exercise to a JSON file."""
    if not REJECTED_VIDEOS:
        print("\n[INFO] No rejected videos to save.")
        return

    ensure_directory_exists(destination_root)
    json_path = destination_root / "rejected_videos.json"

    with open(json_path, "w") as f:
        json.dump(REJECTED_VIDEOS, f, indent=2)

    print(f"[INFO] Rejected videos JSON saved to: {json_path}")


def filter_json_files_for_accepted_videos(destination_root: Path):
    """Filter manifest.json and ground_truth.json to only include videos that were actually copied."""
    # Get list of accepted video files from cleaned dataset
    accepted_videos = set()
    for root, _, files in os.walk(destination_root):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                accepted_videos.add(file)

    # Filter manifest.json
    manifest_file = destination_root / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r") as f:
            original_manifest = json.load(f)

        # Filter to only include accepted videos
        filtered_manifest = {}
        for video_path, exercise in original_manifest.items():
            video_filename = Path(video_path).name
            if video_filename in accepted_videos:
                filtered_manifest[video_path] = exercise

        with open(manifest_file, "w") as f:
            json.dump(filtered_manifest, f, indent=2)

        print(f"[INFO] Filtered manifest.json: {len(filtered_manifest)}/{len(original_manifest)} videos retained")

    # Filter ground_truth.json
    ground_truth_file = destination_root / "ground_truth.json"
    if ground_truth_file.exists():
        with open(ground_truth_file, "r") as f:
            original_ground_truth = json.load(f)

        # Filter to only include accepted videos
        filtered_ground_truth = []
        for entry in original_ground_truth:
            video_filename = Path(entry.get("video_path", "")).name
            if video_filename in accepted_videos:
                filtered_ground_truth.append(entry)

        with open(ground_truth_file, "w") as f:
            json.dump(filtered_ground_truth, f, indent=2)

        print(f"[INFO] Filtered ground_truth.json: {len(filtered_ground_truth)}/{len(original_ground_truth)} entries retained")


def prompt_replace_dataset(source_root: Path, destination_root: Path) -> None:
    """Ask user whether to replace the original dataset with the cleaned dataset."""
    print("\n" + "=" * 50)
    print("Replace original dataset with cleaned dataset?")
    print("=" * 50)
    print(f"  Original dataset:  {source_root}")
    print(f"  Cleaned dataset:   {destination_root}")
    print("")
    print("  y = Remove original 'dataset' folder and rename 'cleaned_dataset' to 'dataset'")
    print("  n = Keep both folders (cleaned dataset saved separately)")
    print("")

    while True:
        response = input("Replace original dataset? (y/N): ").strip().lower()
        if response in ("", "n", "no"):
            print("\n[INFO] Keeping both folders.")
            print(f"  Original dataset preserved at: {source_root}")
            print(f"  Cleaned dataset available at:  {destination_root}")
            break
        elif response in ("y", "yes"):
            try:
                # Preserve important JSON files before removing original dataset
                manifest_file = source_root / "manifest.json"
                fine_labels_file = source_root / "fine_grained_labels.json"
                ground_truth_file = source_root / "ground_truth.json"
                qved_train_file = source_root / "qved_train.json"
                qved_val_file = source_root / "qved_val.json"
                qved_test_file = source_root / "qved_test.json"

                # Copy JSON files to cleaned dataset before swap
                if manifest_file.exists():
                    shutil.copy2(manifest_file, destination_root / "manifest.json")
                    print(f"[INFO] Preserved: manifest.json")
                if fine_labels_file.exists():
                    shutil.copy2(fine_labels_file, destination_root / "fine_grained_labels.json")
                    print(f"[INFO] Preserved: fine_grained_labels.json")
                if ground_truth_file.exists():
                    shutil.copy2(ground_truth_file, destination_root / "ground_truth.json")
                    print(f"[INFO] Preserved: ground_truth.json")
                if qved_train_file.exists():
                    shutil.copy2(qved_train_file, destination_root / "qved_train.json")
                    print(f"[INFO] Preserved: qved_train.json")
                if qved_val_file.exists():
                    shutil.copy2(qved_val_file, destination_root / "qved_val.json")
                    print(f"[INFO] Preserved: qved_val.json")
                if qved_test_file.exists():
                    shutil.copy2(qved_test_file, destination_root / "qved_test.json")
                    print(f"[INFO] Preserved: qved_test.json")

                # Filter JSON files to only include accepted videos
                print(f"\n[INFO] Filtering JSON files to match cleaned dataset...")
                filter_json_files_for_accepted_videos(destination_root)

                print(f"\n[INFO] Removing original dataset: {source_root}")
                shutil.rmtree(source_root)
                print(f"[INFO] Renaming {destination_root} -> {source_root}")
                destination_root.rename(source_root)
                print(f"\n✓ Dataset replaced successfully!")
                print(f"  Cleaned dataset now at: {source_root}")
            except Exception as e:
                print(f"\n[ERROR] Failed to replace dataset: {e}")
                print(f"  Cleaned dataset still available at: {destination_root}")
            break
        else:
            print("  Please enter 'y' or 'n'")


def print_exercise_stats(stats: dict) -> None:
    """Print a summary of the quality stats for a specific exercise."""
    summary = (
        f"[SUMMARY] {stats['Exercise']}\n"
        f"  total_videos:        {stats['total_videos']}\n"
        f"  accepted_videos:     {stats['accepted_videos']}\n"
        f"  rejected_videos:     {stats['rejected_videos']}\n"
        f"  corrupted_files:     {stats['corrupted_files']}\n"
        f"  low_resolution:      {stats['low_resolution']}\n"
        f"  too_dark:            {stats['too_dark']}\n"
        f"  too_bright:          {stats['too_bright']}\n"
        f"  blurry:              {stats['blurry']}\n"
        f"  insufficient_motion: {stats['insufficient_motion']}\n"
    )

    tqdm.write(summary)


def clean_dataset(source_root: Path, destination_root: Path):
    """Clean a dataset by filtering out low-quality or corrupted videos."""
    print("\n[INFO] Starting dataset cleaning...\n")
    ensure_directory_exists(destination_root)

    video_extensions = (".mp4", ".avi", ".mov")

    overall_exercise_stats = []
    totals = default_stats("ALL_EXERCISES")

    for root, _, files in os.walk(source_root):
        if len(files) == 0:
            continue
        root_path = Path(root)
        folder_name = root_path.name

        exercise_stats = default_stats(folder_name)
        any_video = False

        with tqdm(total=len(files), desc=f"Processing {folder_name}", unit="video", ncols=80, leave=False) as pbar:
            for file in files:
                if not file.lower().endswith(video_extensions):
                    continue
                any_video = True

                exercise_stats["total_videos"] += 1
                video_path = root_path / file

                metrics, issues = analyze_video_quality(video_path, NUM_SAMPLED_FRAMES, FRAME_STRIDE, folder_name)
                accepted, reasons = evaluate_video_acceptance(metrics, issues, exercise_stats)

                decision = "accepted" if accepted else "rejected"

                VIDEO_LOG.append({
                    "exercise": folder_name,
                    "file": file,
                    "width": metrics.get("width"),
                    "height": metrics.get("height"),
                    "brightness": None if np.isnan(metrics.get("mean_brightness", float("nan"))) else float(np.round(metrics.get("mean_brightness"), 2)),
                    "sharpness": None if np.isnan(metrics.get("sharpness_score", float("nan"))) else float(np.round(metrics.get("sharpness_score"), 2)),
                    "motion_flag": metrics.get("motion_flag"),
                    "motion_detected": metrics.get("motion_detected"),
                    "motion_pairs": metrics.get("motion_pairs"),
                    "motion_active_pairs": metrics.get("motion_active_pairs"),
                    "motion_active_frame_pct": None if np.isnan(metrics.get("motion_active_frame_pct", float("nan"))) else float(np.round(metrics.get("motion_active_frame_pct"), 4)),
                    "motion_mean_change_ratio": None if np.isnan(metrics.get("motion_mean_change_ratio", float("nan"))) else float(np.round(metrics.get("motion_mean_change_ratio"), 6)),
                    "motion_max_change_ratio": None if np.isnan(metrics.get("motion_max_change_ratio", float("nan"))) else float(np.round(metrics.get("motion_max_change_ratio"), 6)),
                    "decision": decision,
                    "reasons": ", ".join(reasons) if reasons else "passed_all_checks"
                })

                if accepted:
                    copy_video_with_structure(video_path, source_root, destination_root)
                    exercise_stats["accepted_videos"] += 1
                else:
                    exercise_stats["rejected_videos"] += 1
                    # Track rejected videos by exercise
                    if folder_name not in REJECTED_VIDEOS:
                        REJECTED_VIDEOS[folder_name] = []
                    REJECTED_VIDEOS[folder_name].append(str(video_path))
                pbar.update(1)

        if any_video:
            print_exercise_stats(exercise_stats)
            overall_exercise_stats.append(exercise_stats)

            for key in totals:
                if key != "Exercise":
                    totals[key] += exercise_stats[key]

    generate_cleaning_report(overall_exercise_stats, totals, CLEANED_DATASET_PATH)
    save_rejected_videos_json(CLEANED_DATASET_PATH)

    # Return source and destination for the replace prompt
    return source_root, destination_root


if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset path not found:\n  {DATASET_PATH}")
    else:
        source, dest = clean_dataset(DATASET_PATH, CLEANED_DATASET_PATH)
        print("\n[SUCCESS] Dataset cleaning completed.")
        prompt_replace_dataset(source, dest)
