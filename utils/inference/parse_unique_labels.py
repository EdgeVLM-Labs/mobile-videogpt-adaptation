"""Parse unique exercise labels and feedback from prediction results.

Usage:
    # Parse labels with default format (Exercise - Feedback)
    python utils/inference/parse_unique_labels.py --json_file_path "results/QEVD-Fit-300k Only.json"

    # Parse labels with only feedback format
    python utils/inference/parse_unique_labels.py --json_file_path "results/Modified Ground Truth (Feedback Only).json" --enable_only_feedback_label_format

    # Add sentiment analysis column next to each feedback column
    python utils/inference/parse_unique_labels.py --json_file_path "results/QEVD-Fit-300k Only.json" --enable_feedback_sentiment_analysis

    # Both flags can be combined
    python utils/inference/parse_unique_labels.py --json_file_path "results/Modified Ground Truth (Feedback Only).json" --enable_only_feedback_label_format --enable_feedback_sentiment_analysis

"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
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


def parse_unique_labels(
    data: List[Dict[str, str]], 
    enable_only_feedback_label_format: bool = False
) -> Dict[str, Any]:
    """Parse unique labels from prediction data.

    Args:
        data: List of dictionaries containing 'ground_truth' and 'prediction' keys.
        enable_only_feedback_label_format: If True, returns tuples of (exercise, feedback).
            If False, groups feedbacks by exercise name.

    Returns:
        Dictionary with 'ground_truths' and 'predictions' keys containing parsed labels.
        When enable_only_feedback_label_format is True, values are lists of tuples.
        Otherwise, values are dicts mapping exercise names to lists of feedback strings.
    """
    if enable_only_feedback_label_format:
        ground_truths: Set[str] = set()
        predictions: Set[str] = set()

        for item in data:
            gt = item['ground_truth']
            pred = item['prediction']
            ground_truths.add(gt)
            predictions.add(pred)
        
        return {
            "ground_truths": sorted(list(ground_truths)),
            "predictions": sorted(list(predictions))
        }

    else:
        ground_truths: Dict[str, Set[str]] = defaultdict(set)
        predictions: Dict[str, Set[str]] = defaultdict(set)
        
        for item in data:
            gt = item['ground_truth']
            pred = item['prediction']
            gt_exercise, _ = _split_exercise_feedback(gt)
            pred_exercise, _ = _split_exercise_feedback(pred)
            
            ground_truths[gt_exercise].add(gt)
            predictions[pred_exercise].add(pred)
        
        return {
            "ground_truths": {k: sorted(list(v)) for k, v in ground_truths.items()},
            "predictions": {k: sorted(list(v)) for k, v in predictions.items()}
        }

def _write_sentiment_counts(
    writer: pd.ExcelWriter,
    gt_sentiments: List[str],
    pred_sentiments: List[str],
    startrow: int,
) -> None:
    """Write a positive/neutral/negative count summary table to the Excel sheet.

    Args:
        writer: Active ExcelWriter instance.
        gt_sentiments: List of sentiment strings for ground truths.
        pred_sentiments: List of sentiment strings for predictions.
        startrow: Zero-based row at which to start writing.
    """
    labels = ["positive", "neutral", "negative"]
    counts_data = {
        "Sentiment": labels,
        "Ground Truth Count": [gt_sentiments.count(lbl) for lbl in labels],
        "Prediction Count": [pred_sentiments.count(lbl) for lbl in labels],
    }
    counts_df = pd.DataFrame(counts_data)
    counts_df.to_excel(writer, sheet_name="Summary", index=False, startrow=startrow)


def save_to_excel(
    result: Dict[str, Any],
    enable_only_feedback_label_format: bool = False,
    output_path: Optional[Path] = None,
    enable_feedback_sentiment_analysis: bool = False,
) -> None:
    """Save parsed labels to Excel with summary tables.

    Args:
        result: Dictionary containing 'ground_truths' and 'predictions'
        enable_only_feedback_label_format: Format mode for the output
        output_path: Path to save the Excel file
        enable_feedback_sentiment_analysis: When True, appends a sentiment
            column next to each feedback column.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        if enable_only_feedback_label_format:
            # --- Labels sheet: main data only ---
            max_len = max(len(result['ground_truths']), len(result['predictions']))
            gt_list = result['ground_truths'] + [''] * (max_len - len(result['ground_truths']))
            pred_list = result['predictions'] + [''] * (max_len - len(result['predictions']))

            df_data: Dict[str, List[str]] = {'Ground Truths': gt_list}
            if enable_feedback_sentiment_analysis:
                df_data['Ground Truth Sentiment'] = [
                    analyze_feedback_sentiment(fb) for fb in gt_list
                ]
            df_data['Predictions'] = pred_list
            if enable_feedback_sentiment_analysis:
                df_data['Prediction Sentiment'] = [
                    analyze_feedback_sentiment(fb) for fb in pred_list
                ]
            data_df = pd.DataFrame(df_data)
            data_df.to_excel(writer, sheet_name='Labels', index=False, startrow=0)

            # --- Summary sheet ---
            summary_current_row = 0

            summary_data = {
                'Metric': ['Total Ground Truths', 'Total Predictions'],
                'Count': [len(result['ground_truths']), len(result['predictions'])],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=summary_current_row)
            summary_current_row += len(summary_df) + 2

            if enable_feedback_sentiment_analysis:
                gt_sentiments = df_data.get('Ground Truth Sentiment', [])
                pred_sentiments = df_data.get('Prediction Sentiment', [])
                _write_sentiment_counts(
                    writer, gt_sentiments, pred_sentiments, summary_current_row
                )

        else:
            # --- Labels sheet: main data only ---
            data_rows = []

            all_exercises = sorted(set(list(result['ground_truths'].keys()) + list(result['predictions'].keys())))

            for idx, exercise in enumerate(all_exercises):
                gt_feedbacks = sorted(result['ground_truths'].get(exercise, []))
                pred_feedbacks = sorted(result['predictions'].get(exercise, []))

                max_feedbacks = max(len(gt_feedbacks), len(pred_feedbacks))

                for i in range(max_feedbacks):
                    gt_feedback = gt_feedbacks[i] if i < len(gt_feedbacks) else ''
                    pred_feedback = pred_feedbacks[i] if i < len(pred_feedbacks) else ''

                    row: Dict[str, str] = {'Ground Truth': gt_feedback}
                    if enable_feedback_sentiment_analysis:
                        _, gt_fb_only = _split_exercise_feedback(gt_feedback)
                        row['Ground Truth Sentiment'] = analyze_feedback_sentiment(gt_fb_only)
                    row['Prediction'] = pred_feedback
                    if enable_feedback_sentiment_analysis:
                        _, pred_fb_only = _split_exercise_feedback(pred_feedback)
                        row['Prediction Sentiment'] = analyze_feedback_sentiment(pred_fb_only)
                    data_rows.append(row)

                # Add empty row between exercises (except after the last)
                separator: Dict[str, str] = {'Ground Truth': '', 'Prediction': ''}
                if enable_feedback_sentiment_analysis:
                    separator['Ground Truth Sentiment'] = ''
                    separator['Prediction Sentiment'] = ''
                if idx < len(all_exercises) - 1:
                    data_rows.append(separator)

            data_df = pd.DataFrame(data_rows)
            data_df.to_excel(writer, sheet_name='Labels', index=False, startrow=0)

            # --- Summary sheet ---
            summary_current_row = 0

            # Total counts table
            total_gt_count = sum(len(feedbacks) for feedbacks in result['ground_truths'].values())
            total_pred_count = sum(len(feedbacks) for feedbacks in result['predictions'].values())
            summary_data = {
                'Metric': ['Total Ground Truths', 'Total Predictions'],
                'Count': [total_gt_count, total_pred_count],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=summary_current_row)
            summary_current_row += len(summary_df) + 2

            # Per-exercise count table
            exercise_counts = []
            for exercise in all_exercises:
                exercise_counts.append({
                    'Exercise': exercise,
                    'Ground Truth Count': len(result['ground_truths'].get(exercise, [])),
                    'Prediction Count': len(result['predictions'].get(exercise, [])),
                })
            exercise_count_df = pd.DataFrame(exercise_counts)
            exercise_count_df.to_excel(writer, sheet_name='Summary', index=False, startrow=summary_current_row)
            summary_current_row += len(exercise_count_df) + 2

            # Sentiment counts table
            if enable_feedback_sentiment_analysis:
                gt_sentiments = [r.get('Ground Truth Sentiment', '') for r in data_rows]
                pred_sentiments = [r.get('Prediction Sentiment', '') for r in data_rows]
                _write_sentiment_counts(
                    writer, gt_sentiments, pred_sentiments, summary_current_row
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse unique labels from prediction results JSON file"
    )
    parser.add_argument(
        "--json_file_path",
        type=str,
        required=True,
        help="Path to the JSON file containing labels"
    )
    parser.add_argument(
        "--enable_only_feedback_label_format",
        action="store_true",
        help="Enable only feedback label format (returns tuples instead of grouped by exercise)"
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
        help="Output Excel file path (default: input filename with _parsed.xlsx suffix)"
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
        result = parse_unique_labels(data, args.enable_only_feedback_label_format)
        # print(result)
    except KeyError as e:
        print(f"Error: Missing required key in data: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing labels: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Create default output filename based on input
        output_path = input_path.parent / f"{input_path.stem}_parsed.xlsx"
    
    # Ensure output has .xlsx extension
    if output_path.suffix.lower() != '.xlsx':
        output_path = output_path.with_suffix('.xlsx')
    
    # Save to Excel
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_to_excel(
            result,
            args.enable_only_feedback_label_format,
            output_path,
            args.enable_feedback_sentiment_analysis,
        )
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
