"""Parse unique exercise labels from JSON dataset

Supports two JSON formats:
  Format 1 (10 exercises): items have 'exercise', 'labels_descriptive', 'coach', 'feedback'
  Format 2 (14 exercises): items have 'video_path', 'labels', 'labels_descriptive', 'split'

Three label types can be processed via --label_type:
  - labels_descriptive(default): Exercise labels with "exercise - description" format (grouped by exercise)
  - coach: Encouraging phrases/compliments (flat list, no exercise grouping)
  - feedback: Corrective instructions (flat list, no exercise grouping)

Usage:
    python utils/inference/parse_unique_labels.py --json_file_path "new_jsons/fine_grained_labels_10.json"

    python utils/inference/parse_unique_labels.py --json_file_path "new_jsons/fine_grained_labels_10.json" --label_type coach

    python utils/inference/parse_unique_labels.py --json_file_path "new_jsons/fine_grained_labels_10.json" --label_type feedback --enable_feedback_sentiment_analysis

    python utils/inference/parse_unique_labels.py --json_file_path "new_jsons/fine_grained_labels_14.json" --output results/my_output.xlsx
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
import transformers
from transformers import pipeline

# Suppress noisy model-load INFO/WARNING messages (e.g. unexpected key notices)
transformers.logging.set_verbosity_error()

DASH_CONTAIN_EXERCISES = [
    "mountain-climbers",
]

_sentiment_classifier = None

SENTIMENT_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def analyze_feedback_sentiment(feedback: str) -> str:
    """Classify a feedback string as positive, neutral, or negative.

    Uses the cardiffnlp/twitter-roberta-base-sentiment model.
    The classifier is lazily initialised on first call.

    Args:
        feedback: The feedback text to classify.

    Returns:
        One of 'positive', 'neutral', or 'negative'.
        Returns an empty string when *feedback* is empty.
    """
    if not feedback or not feedback.strip():
        return ""

    global _sentiment_classifier
    if _sentiment_classifier is None:
        _sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
        )

    result = _sentiment_classifier(feedback[:512])  # model max-length guard
    label = result[0]["label"]
    return SENTIMENT_LABEL_MAP.get(label, label.lower())


def _split_exercise_feedback(text: str) -> Tuple[str, str]:
    """Split text into (exercise_name, feedback) handling exercises that contain dashes.

    Args:
        text: The text to split into exercise name and feedback.

    Returns:
        A tuple of (exercise_name, feedback) where feedback may be empty if no separator found.
    """
    if not text:
        return ("", "")

    text_lower = text.strip().lower()

    # Check if text starts with a dash-containing exercise name
    for exercise in DASH_CONTAIN_EXERCISES:
        if text_lower.startswith(exercise):
            rest = text.strip()[len(exercise):]
            # Expect " - feedback" after the exercise name
            if rest.startswith(' - '):
                return (exercise, rest[3:].strip())
            elif rest.startswith(' -'):
                return (exercise, rest[2:].strip())
            elif rest.startswith('- '):
                return (exercise, rest[2:].strip())
            elif rest.startswith('-'):
                return (exercise, rest[1:].strip())
            else:
                # Exercise name matched but no dash separator after it
                return (exercise, "")

    # Default: split on first " - " or first "-"
    if ' - ' in text:
        parts = text.split(' - ', 1)
        return (parts[0].strip(), parts[1].strip())
    elif '-' in text:
        parts = text.split('-', 1)
        return (parts[0].strip(), parts[1].strip())

    return (text.strip(), "")


def parse_labels(data: List[Dict[str, Any]], label_type: str = 'labels_descriptive') -> Dict[str, Any]:
    """Parse unique labels from dataset items.

    Args:
        data: List of dataset items.
        label_type: Type of label to process - 'labels_descriptive', 'coach', or 'feedback'.

    Returns:
        Dictionary with:
            'ground_truths'   – dict mapping exercise name -> sorted list of unique label strings
                               (for labels_descriptive), or a flat sorted list (for coach/feedback)
            'total_instances' – total label assignments across all items (counts duplicates)
            'grouped_by_exercise' – True if grouped by exercise, False otherwise
    
    Raises:
        ValueError: If the label_type field is not found in the dataset.
    """
    # Guardrail: Check if the label_type field exists in the dataset
    if not data:
        raise ValueError("Dataset is empty")
    
    field_exists = any(label_type in item for item in data)
    if not field_exists:
        available_fields = set()
        for item in data:
            available_fields.update(item.keys())
        raise ValueError(
            f"Label type '{label_type}' not found in any dataset items. "
            f"Available fields: {sorted(available_fields)}"
        )
    
    if label_type == 'labels_descriptive':
        ground_truths: Dict[str, Set[str]] = defaultdict(set)
        total_instances = 0

        for item in data:
            for label in item.get('labels_descriptive', []):
                exercise, _ = _split_exercise_feedback(label)
                ground_truths[exercise].add(label)
                total_instances += 1

        return {
            "ground_truths": {k: sorted(list(v)) for k, v in sorted(ground_truths.items())},
            "total_instances": total_instances,
            "grouped_by_exercise": True,
        }
    
    elif label_type == 'coach':
        unique_labels: Set[str] = set()
        total_instances = 0

        for item in data:
            coach_list = item.get('coach', [])
            if isinstance(coach_list, list):
                for label in coach_list:
                    unique_labels.add(label)
                    total_instances += 1

        return {
            "ground_truths": sorted(list(unique_labels)),
            "total_instances": total_instances,
            "grouped_by_exercise": False,
        }
    
    elif label_type == 'feedback':
        unique_labels: Set[str] = set()
        total_instances = 0

        for item in data:
            feedback = item.get('feedback', '')
            if feedback and isinstance(feedback, str):
                unique_labels.add(feedback)
                total_instances += 1

        return {
            "ground_truths": sorted(list(unique_labels)),
            "total_instances": total_instances,
            "grouped_by_exercise": False,
        }
    
    else:
        raise ValueError(f"Invalid label_type: {label_type}")


def _write_sentiment_counts(
    writer: pd.ExcelWriter,
    sentiments: List[str],
    startrow: int,
) -> None:
    """Write a positive/neutral/negative count summary table to the Excel sheet.

    Args:
        writer: Active ExcelWriter instance.
        sentiments: List of sentiment strings.
        startrow: Zero-based row at which to start writing.
    """
    labels = ["positive", "neutral", "negative"]
    counts_data = {
        "Sentiment": labels,
        "Count": [sentiments.count(lbl) for lbl in labels],
    }
    counts_df = pd.DataFrame(counts_data)
    counts_df.to_excel(writer, sheet_name="Summary", index=False, startrow=startrow)


def save_to_excel(
    result: Dict[str, Any],
    output_path: Path,
    label_type: str = 'labels_descriptive',
    enable_feedback_sentiment_analysis: bool = False,
) -> None:
    """Save parsed labels to Excel with summary tables.

    Args:
        result: Dictionary from parse_labels() containing 'ground_truths' and 'total_instances'.
        output_path: Path to save the Excel file.
        label_type: Type of label being processed.
        enable_feedback_sentiment_analysis: When True, appends a sentiment column
            next to the label column.
    """
    ground_truths = result["ground_truths"]
    grouped_by_exercise = result.get("grouped_by_exercise", True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # --- Labels sheet ---
        data_rows: List[Dict[str, str]] = []

        if grouped_by_exercise:
            # For labels_descriptive: grouped by exercise with separators
            all_exercises = sorted(ground_truths.keys())
            
            for idx, exercise in enumerate(all_exercises):
                for label in ground_truths[exercise]:
                    _, feedback_part = _split_exercise_feedback(label)
                    row: Dict[str, str] = {'Label': label}
                    if enable_feedback_sentiment_analysis:
                        row['Sentiment'] = analyze_feedback_sentiment(feedback_part)
                    data_rows.append(row)

                # Empty separator row between exercises (except after the last)
                if idx < len(all_exercises) - 1:
                    separator: Dict[str, str] = {'Label': ''}
                    if enable_feedback_sentiment_analysis:
                        separator['Sentiment'] = ''
                    data_rows.append(separator)
        else:
            # For coach/feedback: flat list, no grouping
            for label in ground_truths:
                row: Dict[str, str] = {'Label': label}
                if enable_feedback_sentiment_analysis:
                    row['Sentiment'] = analyze_feedback_sentiment(label)
                data_rows.append(row)

        data_df = pd.DataFrame(data_rows)
        data_df.to_excel(writer, sheet_name='Labels', index=False, startrow=0)

        # --- Summary sheet ---
        if grouped_by_exercise:
            total_unique = sum(len(labels) for labels in ground_truths.values())
        else:
            total_unique = len(ground_truths)
        
        summary_current_row = 0

        summary_data = {
            'Metric': ['Total Label Instances', 'Total Unique Labels'],
            'Count': [result['total_instances'], total_unique],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=summary_current_row)
        summary_current_row += len(summary_df) + 2

        # Per-exercise unique label count table (only for labels_descriptive)
        if grouped_by_exercise:
            all_exercises = sorted(ground_truths.keys())
            exercise_counts = [
                {
                    'Exercise': exercise,
                    'Unique Label Count': len(ground_truths[exercise]),
                }
                for exercise in all_exercises
            ]
            exercise_count_df = pd.DataFrame(exercise_counts)
            exercise_count_df.to_excel(writer, sheet_name='Summary', index=False, startrow=summary_current_row)
            summary_current_row += len(exercise_count_df) + 2

        # Sentiment counts table
        if enable_feedback_sentiment_analysis:
            sentiments = [r.get('Sentiment', '') for r in data_rows]
            _write_sentiment_counts(writer, sentiments, summary_current_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse unique labels from a dataset JSON file"
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="labels_descriptive",
        choices=["labels_descriptive", "coach", "feedback"],
        help=(
            "Type of label to process: 'labels_descriptive' (grouped by exercise), "
            "'coach' (encouraging phrases), or 'feedback' (corrective instructions)"
        ),
    )
    parser.add_argument(
        "--enable_feedback_sentiment_analysis",
        action="store_true",
        help=(
            "Run sentiment analysis (positive/neutral/negative) on the feedback "
            "portion of each label and add the result as a column in the Excel output."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Excel file path (optional; defaults to input filename with _{label_type}_parsed.xlsx suffix)"
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.json_file_path)
    if not input_path.exists():
        print(f"Error: File not found: {args.json_file_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse labels
    try:
        result = parse_labels(data, args.label_type)
    except Exception as e:
        print(f"Error parsing labels: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        if args.enable_feedback_sentiment_analysis:
            output_path = input_path.parent / f"{input_path.stem}_{args.label_type}_parsed_with_sentiment.xlsx"
        else:
            output_path = input_path.parent / f"{input_path.stem}_{args.label_type}_parsed.xlsx"

    # Ensure output has .xlsx extension
    if output_path.suffix.lower() != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')

    # Save to Excel
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_excel(result, output_path, args.label_type, args.enable_feedback_sentiment_analysis)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
