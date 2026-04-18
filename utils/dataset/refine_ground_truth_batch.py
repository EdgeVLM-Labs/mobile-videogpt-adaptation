import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refine_ground_truth_one_entry import refine_entry

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "QVED-CLEANED" / "fine_grained_labels.json"
print(f"Labels path: {LABELS_PATH}")
OUTPUT_JSON = BASE_DIR / "QVED-CLEANED" / "main.json"
OUTPUT_XLSX = BASE_DIR / "QVED-CLEANED" / "main.xlsx"

RETRY_ATTEMPTS = 3
RETRY_DELAY = 5


def load_checkpoint():
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r") as f:
            data = json.load(f)
        done = {entry["video_path"] for entry in data if "feedback" in entry}
        return data, done
    return [], set()


def build_feedback_dict(results):
    """Build dictionary of {exercise: [unique_feedbacks]} from existing results."""
    feedback_dict = {}
    for entry in results:
        exercise = entry.get("exercise")
        feedback = entry.get("feedback", "").strip()
        if exercise and feedback:
            if exercise not in feedback_dict:
                feedback_dict[exercise] = []
            if feedback not in feedback_dict[exercise]:
                feedback_dict[exercise].append(feedback)
    return feedback_dict


def save_checkpoint(data):
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)


def export_xlsx(data):
    rows = []
    for entry in data:
        rows.append({
            "video_path": entry.get("video_path", ""),
            "exercise": entry.get("exercise", ""),
            "labels_descriptive": "\n".join(entry.get("labels_descriptive", [])),
            "coach": "\n".join(entry.get("coach", [])),
            "feedback": entry.get("feedback", ""),
        })
    df = pd.DataFrame(rows, columns=["video_path", "exercise", "labels_descriptive", "coach", "feedback"])
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ground Truths")
        ws = writer.sheets["Ground Truths"]
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 40
    print(f"Saved Excel: {OUTPUT_XLSX}")


def main():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    total = len(labels)
    print(f"Loaded {total} entries from {LABELS_PATH}")

    results, done = load_checkpoint()
    if done:
        print(f"Resuming — {len(done)} entries already processed\n")

    results_map = {e["video_path"]: e for e in results}
    feedback_dict = build_feedback_dict(results)

    for i, entry in enumerate(labels, 1):
        vp = entry["video_path"]

        if vp in done:
            print(f"[{i}/{total}] Skipping (done): {vp}")
            continue

        print(f"[{i}/{total}] Processing: {vp}", end="  ", flush=True)

        exercise = entry["exercise"]
        existing_feedbacks = feedback_dict.get(exercise, [])

        feedback = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                feedback = refine_entry(entry, existing_feedbacks)
                break
            except Exception as e:
                print(f"\n  Attempt {attempt} failed: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY)

        if feedback is None:
            feedback = ""
            print(f"  [FAILED after {RETRY_ATTEMPTS} attempts]")
        else:
            print(f"-> {feedback}")
            # Add new feedback to the dict if it's unique for this exercise
            if feedback and feedback not in existing_feedbacks:
                if exercise not in feedback_dict:
                    feedback_dict[exercise] = []
                feedback_dict[exercise].append(feedback)

        new_entry = {
            "video_path": entry["video_path"],
            "exercise": entry["exercise"],
            "labels_descriptive": entry.get("labels_descriptive", []),
            "coach": entry.get("coach", []),
            "feedback": feedback,
        }
        results_map[vp] = new_entry
        results = list(results_map.values())
        save_checkpoint(results)

    print(f"\nAll entries processed. Saved: {OUTPUT_JSON}")
    export_xlsx(results)


if __name__ == "__main__":
    main()
