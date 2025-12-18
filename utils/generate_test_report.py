#!/usr/bin/env python3
"""
Generate Test Evaluation Report with Cosine Similarity

This script processes test inference results and generates an Excel report
with cosine similarity scores between predictions and ground truth.

Usage:
    python utils/generate_test_report.py --predictions test_predictions.json
    python utils/generate_test_report.py --predictions test_predictions.json --output test_report.xlsx
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.legend import Legend
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from sentence_transformers import SentenceTransformer


# ROUGE-L Score thresholds (0-1 scale, higher is better)
ROUGE_GREEN_THRESHOLD = 0.5   # >= this value is green (good)
ROUGE_YELLOW_THRESHOLD = 0.2  # >= this value is yellow (moderate), below is red

# BERT Similarity thresholds (0-1 scale, higher is better)
BERT_GREEN_THRESHOLD = 0.7   # >= this value is green (good)
BERT_YELLOW_THRESHOLD = 0.4  # >= this value is yellow (moderate), below is red

# METEOR Score thresholds (0-1 scale, higher is better)
METEOR_GREEN_THRESHOLD = 0.5   # >= this value is green (good)
METEOR_YELLOW_THRESHOLD = 0.2  # >= this value is yellow (moderate), below is red


def compute_meteor_score(reference: str, hypothesis: str, metric) -> float:
    """Compute METEOR score."""
    if not reference or not hypothesis or metric is None:
        return 0.0

    try:
        return metric.compute(predictions=[hypothesis], references=[reference])['meteor']
    except:
        return 0.0



def compute_rouge_score(reference: str, hypothesis: str, metric) -> float:
    """Compute ROUGE-L score."""
    if not reference or not hypothesis or metric is None:
        return 0.0

    try:
        result = metric.compute(predictions=[hypothesis], references=[reference])
        return result['rougeL']
    except:
        return 0.0


def extract_exercise_name(text: str) -> str:
    """Extract exercise name from text (before the dash)."""
    if not text:
        return ""

    # Find first dash and get text before it
    if '-' in text:
        return text.split('-')[0].strip().lower()
    return text.strip().lower()


def check_exercise_match(ground_truth: str, prediction: str) -> bool:
    """Check if exercise names match between ground truth and prediction."""
    gt_exercise = extract_exercise_name(ground_truth)
    pred_exercise = extract_exercise_name(prediction)

    if not gt_exercise or not pred_exercise:
        return False

    return gt_exercise == pred_exercise



def compute_rouge_score(reference: str, hypothesis: str, metric) -> float:
    """Compute ROUGE-L score."""
    if not reference or not hypothesis or metric is None:
        return 0.0

    try:
        result = metric.compute(predictions=[hypothesis], references=[reference])
        return result['rougeL']
    except:
        return 0.0


def extract_exercise_name(text: str) -> str:
    """Extract exercise name from text (before the dash)."""
    if not text:
        return ""

    # Find first dash and get text before it
    if '-' in text:
        return text.split('-')[0].strip().lower()
    return text.strip().lower()


def check_exercise_match(ground_truth: str, prediction: str) -> bool:
    """Check if exercise names match between ground truth and prediction."""
    gt_exercise = extract_exercise_name(ground_truth)
    pred_exercise = extract_exercise_name(prediction)

    if not gt_exercise or not pred_exercise:
        return False

    return gt_exercise == pred_exercise


def compute_cosine_similarity_bert(text1: str, text2: str, model) -> float:
    """Compute cosine similarity using BERT embeddings."""
    if not text1 or not text2:
        return 0.0

    try:
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0


def create_excel_report(results: List[Dict], output_path: str, use_bert: bool = True):
    """Create an Excel report with formatted results and similarity scores."""

    # Load BERT model if requested
    bert_model = None
    if use_bert:
        print("Loading BERT model for semantic similarity...")
        try:
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ BERT model loaded")
        except Exception as e:
            print(f"⚠ Failed to load BERT model: {e}")
            print("  Falling back to TF-IDF similarity")
            use_bert = False

    # Create workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Evaluation Results"

    # Define styles
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    cell_alignment = Alignment(vertical="top", wrap_text=True)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Headers
    headers = [
        "ID",
        "Video Path",
        "Ground Truth",
        "Model Prediction",
    ]

    if use_bert:
        headers.append("BERT Similarity")

    headers.extend(["METEOR Score", "ROUGE-L Score", "Exercise Identified", "Status", "Error"])

    # Write headers
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = border

    # Set column widths
    ws.column_dimensions['A'].width = 8   # ID
    ws.column_dimensions['B'].width = 40  # Video Path
    ws.column_dimensions['C'].width = 50  # Ground Truth
    ws.column_dimensions['D'].width = 50  # Prediction

    if use_bert:
        ws.column_dimensions['E'].width = 18  # BERT Similarity
        ws.column_dimensions['F'].width = 18  # METEOR Score
        ws.column_dimensions['G'].width = 18  # ROUGE-L Score
        ws.column_dimensions['H'].width = 18  # Exercise Identified
        ws.column_dimensions['I'].width = 12  # Status
        ws.column_dimensions['J'].width = 40  # Error
    else:
        ws.column_dimensions['E'].width = 18  # METEOR Score
        ws.column_dimensions['F'].width = 18  # ROUGE-L Score
        ws.column_dimensions['G'].width = 18  # Exercise Identified
        ws.column_dimensions['H'].width = 12  # Status
        ws.column_dimensions['I'].width = 40  # Error

    # Freeze header row
    ws.freeze_panes = "A2"

    # Process results
    print(f"\nProcessing {len(results)} results...")

    # Load METEOR metric
    meteor_metric = None
    try:
        meteor_metric = evaluate.load('meteor')
        print("✓ METEOR metric loaded")
    except Exception as e:
        print(f"⚠ Failed to load METEOR metric: {e}")

    # Load ROUGE metric
    rouge_metric = None
    try:
        rouge_metric = evaluate.load('rouge')
        print("✓ ROUGE metric loaded")
    except Exception as e:
        print(f"⚠ Failed to load ROUGE metric: {e}")

    bert_scores = []
    meteor_scores = []
    rouge_scores = []
    exercise_matches = []

    for idx, result in enumerate(results, start=1):
        row = idx + 1

        video_path = result.get('video_path', '')
        ground_truth = result.get('ground_truth', '')
        prediction = result.get('prediction', '')
        status = result.get('status', 'unknown')
        error = result.get('error', '')

        # Compute similarities
        bert_sim = None
        if use_bert and bert_model:
            bert_sim = compute_cosine_similarity_bert(ground_truth, prediction, bert_model)
            bert_scores.append(bert_sim)

        meteor_sim = compute_meteor_score(ground_truth, prediction, meteor_metric)
        meteor_scores.append(meteor_sim)

        rouge_sim = compute_rouge_score(ground_truth, prediction, rouge_metric)
        rouge_scores.append(rouge_sim)

        exercise_match = check_exercise_match(ground_truth, prediction)
        exercise_matches.append(exercise_match)

        # Write data
        col = 1
        ws.cell(row=row, column=col).value = idx
        col += 1
        ws.cell(row=row, column=col).value = video_path
        col += 1
        ws.cell(row=row, column=col).value = ground_truth
        col += 1
        ws.cell(row=row, column=col).value = prediction
        col += 1

        # Track column indices for color coding
        bert_col_idx = None
        if use_bert and bert_sim is not None:
            ws.cell(row=row, column=col).value = round(bert_sim, 4)
            bert_col_idx = col
            col += 1

        meteor_col_idx = col
        ws.cell(row=row, column=col).value = round(meteor_sim, 4)
        col += 1

        rouge_col_idx = col
        ws.cell(row=row, column=col).value = round(rouge_sim, 4)
        col += 1

        exercise_col_idx = col
        ws.cell(row=row, column=col).value = "TRUE" if exercise_match else "FALSE"
        col += 1

        ws.cell(row=row, column=col).value = status
        col += 1
        ws.cell(row=row, column=col).value = error

        # Apply formatting
        for c in range(1, col + 1):
            cell = ws.cell(row=row, column=c)
            cell.alignment = cell_alignment
            cell.border = border

            # Color code similarity scores
            if bert_col_idx and c == bert_col_idx:  # BERT column
                score = bert_sim if bert_sim is not None else 0
                if score >= BERT_GREEN_THRESHOLD:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= BERT_YELLOW_THRESHOLD:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

            if c == meteor_col_idx:  # METEOR column
                score = meteor_sim
                if score >= METEOR_GREEN_THRESHOLD:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= METEOR_YELLOW_THRESHOLD:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

            if c == rouge_col_idx:  # ROUGE-L column
                score = rouge_sim
                if score >= ROUGE_GREEN_THRESHOLD:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= ROUGE_YELLOW_THRESHOLD:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

            if c == exercise_col_idx:  # Exercise Identified column
                if exercise_match:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    # Add summary sheet
    summary_ws = wb.create_sheet("Summary")

    summary_data = [
        ["Metric", "Value"],
        ["Total Samples", len(results)],
        ["Successful", sum(1 for r in results if r.get('status') == 'success')],
        ["Failed", sum(1 for r in results if r.get('status') == 'error')],
        ["", ""],
    ]

    # Track chart data positions
    bert_chart_start_row = None
    meteor_chart_start_row = None

    if use_bert and bert_scores:
        bert_green = sum(1 for s in bert_scores if s >= BERT_GREEN_THRESHOLD)
        bert_yellow = sum(1 for s in bert_scores if BERT_YELLOW_THRESHOLD <= s < BERT_GREEN_THRESHOLD)
        bert_red = sum(1 for s in bert_scores if s < BERT_YELLOW_THRESHOLD)
        bert_chart_start_row = len(summary_data) + 7  # Row where Green count will be
        summary_data.extend([
            ["BERT Similarity", ""],
            ["Mean", round(np.mean(bert_scores), 4)],
            ["Median", round(np.median(bert_scores), 4)],
            ["Std Dev", round(np.std(bert_scores), 4)],
            ["Min", round(np.min(bert_scores), 4)],
            ["Max", round(np.max(bert_scores), 4)],
            [f"Green (≥{BERT_GREEN_THRESHOLD})", bert_green],
            [f"Yellow ({BERT_YELLOW_THRESHOLD}-{BERT_GREEN_THRESHOLD})", bert_yellow],
            [f"Red (<{BERT_YELLOW_THRESHOLD})", bert_red],
            ["", ""],
        ])

    if meteor_scores:
        meteor_green = sum(1 for s in meteor_scores if s >= METEOR_GREEN_THRESHOLD)
        meteor_yellow = sum(1 for s in meteor_scores if METEOR_YELLOW_THRESHOLD <= s < METEOR_GREEN_THRESHOLD)
        meteor_red = sum(1 for s in meteor_scores if s < METEOR_YELLOW_THRESHOLD)
        meteor_chart_start_row = len(summary_data) + 7  # Row where Green count will be
        summary_data.extend([
            ["METEOR Score", ""],
            ["Mean", round(np.mean(meteor_scores), 4)],
            ["Median", round(np.median(meteor_scores), 4)],
            ["Std Dev", round(np.std(meteor_scores), 4)],
            ["Min", round(np.min(meteor_scores), 4)],
            ["Max", round(np.max(meteor_scores), 4)],
            [f"Green (≥{METEOR_GREEN_THRESHOLD})", meteor_green],
            [f"Yellow ({METEOR_YELLOW_THRESHOLD}-{METEOR_GREEN_THRESHOLD})", meteor_yellow],
            [f"Red (<{METEOR_YELLOW_THRESHOLD})", meteor_red],
            ["", ""],
        ])

    rouge_chart_start_row = None
    if rouge_scores:
        rouge_green = sum(1 for s in rouge_scores if s >= ROUGE_GREEN_THRESHOLD)
        rouge_yellow = sum(1 for s in rouge_scores if ROUGE_YELLOW_THRESHOLD <= s < ROUGE_GREEN_THRESHOLD)
        rouge_red = sum(1 for s in rouge_scores if s < ROUGE_YELLOW_THRESHOLD)
        rouge_chart_start_row = len(summary_data) + 7
        summary_data.extend([
            ["ROUGE-L Score", ""],
            ["Mean", round(np.mean(rouge_scores), 4)],
            ["Median", round(np.median(rouge_scores), 4)],
            ["Std Dev", round(np.std(rouge_scores), 4)],
            ["Min", round(np.min(rouge_scores), 4)],
            ["Max", round(np.max(rouge_scores), 4)],
            [f"Green (≥{ROUGE_GREEN_THRESHOLD})", rouge_green],
            [f"Yellow ({ROUGE_YELLOW_THRESHOLD}-{ROUGE_GREEN_THRESHOLD})", rouge_yellow],
            [f"Red (<{ROUGE_YELLOW_THRESHOLD})", rouge_red],
            ["", ""],
        ])

    if exercise_matches:
        exercise_correct = sum(1 for match in exercise_matches if match)
        exercise_incorrect = sum(1 for match in exercise_matches if not match)
        exercise_accuracy = (exercise_correct / len(exercise_matches) * 100) if exercise_matches else 0
        summary_data.extend([
            ["Exercise Identification", ""],
            ["Correct", exercise_correct],
            ["Incorrect", exercise_incorrect],
            ["Accuracy (%)", round(exercise_accuracy, 2)],
            ["", ""],
        ])

    for row_idx, row_data in enumerate(summary_data, start=1):
        for col_idx, value in enumerate(row_data, start=1):
            cell = summary_ws.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.border = border

            if row_idx == 1 or (isinstance(value, str) and value and row_idx > 1 and col_idx == 1):
                cell.font = Font(bold=True)

    summary_ws.column_dimensions['A'].width = 20
    summary_ws.column_dimensions['B'].width = 15

    if use_bert and bert_scores and bert_chart_start_row:
        # BERT Similarity Chart
        bert_chart = BarChart()
        bert_chart.type = "col"
        bert_chart.style = 10
        bert_chart.title = "BERT Similarity Distribution"
        bert_chart.y_axis.title = "Count"
        bert_chart.x_axis.title = "Category"

        # Show primary axes
        bert_chart.x_axis.delete = False
        bert_chart.y_axis.delete = False
        bert_chart.x_axis.majorTickMark = "out"
        bert_chart.y_axis.majorTickMark = "out"

        # Legend position below x-axis
        bert_chart.legend = Legend()
        bert_chart.legend.position = "b"

        # Data reference (values)
        bert_data = Reference(summary_ws, min_col=2, min_row=bert_chart_start_row,
                              max_row=bert_chart_start_row + 2)
        # Categories reference (labels)
        bert_cats = Reference(summary_ws, min_col=1, min_row=bert_chart_start_row,
                              max_row=bert_chart_start_row + 2)

        bert_chart.add_data(bert_data, titles_from_data=False)
        bert_chart.set_categories(bert_cats)
        bert_chart.shape = 4
        bert_chart.width = 12
        bert_chart.height = 8

        # Color the bars: green, yellow, red
        from openpyxl.chart.series import DataPoint
        from openpyxl.drawing.fill import PatternFillProperties, ColorChoice
        from openpyxl.chart.shapes import GraphicalProperties

        series = bert_chart.series[0]
        # Green bar
        pt_green = DataPoint(idx=0)
        pt_green.graphicalProperties = GraphicalProperties()
        pt_green.graphicalProperties.solidFill = "00B050"
        series.data_points.append(pt_green)
        # Yellow bar
        pt_yellow = DataPoint(idx=1)
        pt_yellow.graphicalProperties = GraphicalProperties()
        pt_yellow.graphicalProperties.solidFill = "FFC000"
        series.data_points.append(pt_yellow)
        # Red bar
        pt_red = DataPoint(idx=2)
        pt_red.graphicalProperties = GraphicalProperties()
        pt_red.graphicalProperties.solidFill = "FF0000"
        series.data_points.append(pt_red)

        summary_ws.add_chart(bert_chart, "D2")

    if meteor_scores and meteor_chart_start_row:
        # METEOR Score Chart
        meteor_chart = BarChart()
        meteor_chart.type = "col"
        meteor_chart.style = 10
        meteor_chart.title = "METEOR Score Distribution"
        meteor_chart.y_axis.title = "Count"
        meteor_chart.x_axis.title = "Category"

        # Show primary axes
        meteor_chart.x_axis.delete = False
        meteor_chart.y_axis.delete = False
        meteor_chart.x_axis.majorTickMark = "out"
        meteor_chart.y_axis.majorTickMark = "out"

        # Legend position below x-axis
        meteor_chart.legend = Legend()
        meteor_chart.legend.position = "b"

        # Data reference (values)
        meteor_data = Reference(summary_ws, min_col=2, min_row=meteor_chart_start_row,
                                max_row=meteor_chart_start_row + 2)
        # Categories reference (labels)
        meteor_cats = Reference(summary_ws, min_col=1, min_row=meteor_chart_start_row,
                                max_row=meteor_chart_start_row + 2)

        meteor_chart.add_data(meteor_data, titles_from_data=False)
        meteor_chart.set_categories(meteor_cats)
        meteor_chart.shape = 4
        meteor_chart.width = 12
        meteor_chart.height = 8

        # Color the bars: green, yellow, red
        from openpyxl.chart.series import DataPoint
        from openpyxl.chart.shapes import GraphicalProperties

        series = meteor_chart.series[0]
        # Green bar
        pt_green = DataPoint(idx=0)
        pt_green.graphicalProperties = GraphicalProperties()
        pt_green.graphicalProperties.solidFill = "00B050"
        series.data_points.append(pt_green)
        # Yellow bar
        pt_yellow = DataPoint(idx=1)
        pt_yellow.graphicalProperties = GraphicalProperties()
        pt_yellow.graphicalProperties.solidFill = "FFC000"
        series.data_points.append(pt_yellow)
        # Red bar
        pt_red = DataPoint(idx=2)
        pt_red.graphicalProperties = GraphicalProperties()
        pt_red.graphicalProperties.solidFill = "FF0000"
        series.data_points.append(pt_red)

        # Position second chart below the first
        chart_position = "D17" if use_bert else "D2"
        summary_ws.add_chart(meteor_chart, chart_position)

    if rouge_scores and rouge_chart_start_row:
        # ROUGE-L Score Chart
        rouge_chart = BarChart()
        rouge_chart.type = "col"
        rouge_chart.style = 10
        rouge_chart.title = "ROUGE-L Score Distribution"
        rouge_chart.y_axis.title = "Count"
        rouge_chart.x_axis.title = "Category"

        # Show primary axes
        rouge_chart.x_axis.delete = False
        rouge_chart.y_axis.delete = False
        rouge_chart.x_axis.majorTickMark = "out"
        rouge_chart.y_axis.majorTickMark = "out"

        # Legend position below x-axis
        rouge_chart.legend = Legend()
        rouge_chart.legend.position = "b"

        # Data reference (values)
        rouge_data = Reference(summary_ws, min_col=2, min_row=rouge_chart_start_row,
                                max_row=rouge_chart_start_row + 2)
        # Categories reference (labels)
        rouge_cats = Reference(summary_ws, min_col=1, min_row=rouge_chart_start_row,
                                max_row=rouge_chart_start_row + 2)

        rouge_chart.add_data(rouge_data, titles_from_data=False)
        rouge_chart.set_categories(rouge_cats)
        rouge_chart.shape = 4
        rouge_chart.width = 12
        rouge_chart.height = 8

        # Color the bars: green, yellow, red
        from openpyxl.chart.series import DataPoint
        from openpyxl.chart.shapes import GraphicalProperties

        series = rouge_chart.series[0]
        # Green bar
        pt_green = DataPoint(idx=0)
        pt_green.graphicalProperties = GraphicalProperties()
        pt_green.graphicalProperties.solidFill = "00B050"
        series.data_points.append(pt_green)
        # Yellow bar
        pt_yellow = DataPoint(idx=1)
        pt_yellow.graphicalProperties = GraphicalProperties()
        pt_yellow.graphicalProperties.solidFill = "FFC000"
        series.data_points.append(pt_yellow)
        # Red bar
        pt_red = DataPoint(idx=2)
        pt_red.graphicalProperties = GraphicalProperties()
        pt_red.graphicalProperties.solidFill = "FF0000"
        series.data_points.append(pt_red)

        # Position third chart
        if use_bert and meteor_scores:
            chart_position = "D32"  # Below METEOR chart
        elif meteor_scores:
            chart_position = "D17"  # Below METEOR chart
        else:
            chart_position = "D2"   # First chart
        summary_ws.add_chart(rouge_chart, chart_position)

    # Save workbook
    wb.save(output_path)
    print(f"✓ Excel report saved to: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.get('status') == 'error')}")

    if use_bert and bert_scores:
        print(f"\nBERT Similarity:")
        print(f"  Mean: {np.mean(bert_scores):.4f}")
        print(f"  Median: {np.median(bert_scores):.4f}")
        print(f"  Std Dev: {np.std(bert_scores):.4f}")

    if meteor_scores:
        print(f"\nMETEOR Score:")
        print(f"  Mean: {np.mean(meteor_scores):.4f}")
        print(f"  Median: {np.median(meteor_scores):.4f}")
        print(f"  Std Dev: {np.std(meteor_scores):.4f}")

    if rouge_scores:
        print(f"\nROUGE-L Score:")
        print(f"  Mean: {np.mean(rouge_scores):.4f}")
        print(f"  Median: {np.median(rouge_scores):.4f}")
        print(f"  Std Dev: {np.std(rouge_scores):.4f}")

    if exercise_matches:
        exercise_correct = sum(1 for match in exercise_matches if match)
        exercise_accuracy = (exercise_correct / len(exercise_matches) * 100) if exercise_matches else 0
        print(f"\nExercise Identification:")
        print(f"  Correct: {exercise_correct}/{len(exercise_matches)}")
        print(f"  Accuracy: {exercise_accuracy:.2f}%")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate test evaluation report with similarity scores")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON from test_inference.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output Excel file path (default: same directory as predictions)")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BERT similarity (faster, uses only TF-IDF)")

    args = parser.parse_args()

    # Set default output path to same directory as predictions if not provided
    if args.output is None:
        pred_path = Path(args.predictions)
        args.output = str(pred_path.parent / "test_evaluation_report.xlsx")
        print(f"Output will be saved to: {args.output}")

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} predictions")

    # Generate report
    create_excel_report(results, args.output, use_bert=not args.no_bert)


if __name__ == "__main__":
    main()
