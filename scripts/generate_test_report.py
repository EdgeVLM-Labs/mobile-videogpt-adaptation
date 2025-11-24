#!/usr/bin/env python3
"""
Generate Test Evaluation Report with Cosine Similarity

This script processes test inference results and generates an Excel report
with cosine similarity scores between predictions and ground truth.

Usage:
    python scripts/generate_test_report.py --predictions test_predictions.json
    python scripts/generate_test_report.py --predictions test_predictions.json --output test_report.xlsx
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def compute_cosine_similarity_tfidf(text1: str, text2: str) -> float:
    """Compute cosine similarity using TF-IDF vectors."""
    if not text1 or not text2:
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


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
        "TF-IDF Similarity",
    ]

    if use_bert:
        headers.append("BERT Similarity")

    headers.extend(["Status", "Error"])

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
    ws.column_dimensions['E'].width = 18  # TF-IDF Similarity
    if use_bert:
        ws.column_dimensions['F'].width = 18  # BERT Similarity
        ws.column_dimensions['G'].width = 12  # Status
        ws.column_dimensions['H'].width = 40  # Error
    else:
        ws.column_dimensions['F'].width = 12  # Status
        ws.column_dimensions['G'].width = 40  # Error

    # Freeze header row
    ws.freeze_panes = "A2"

    # Process results
    print(f"\nProcessing {len(results)} results...")

    tfidf_scores = []
    bert_scores = []

    for idx, result in enumerate(results, start=1):
        row = idx + 1

        video_path = result.get('video_path', '')
        ground_truth = result.get('ground_truth', '')
        prediction = result.get('prediction', '')
        status = result.get('status', 'unknown')
        error = result.get('error', '')

        # Compute similarities
        tfidf_sim = compute_cosine_similarity_tfidf(ground_truth, prediction)
        tfidf_scores.append(tfidf_sim)

        bert_sim = None
        if use_bert and bert_model:
            bert_sim = compute_cosine_similarity_bert(ground_truth, prediction, bert_model)
            bert_scores.append(bert_sim)

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
        ws.cell(row=row, column=col).value = round(tfidf_sim, 4)
        col += 1

        if use_bert and bert_sim is not None:
            ws.cell(row=row, column=col).value = round(bert_sim, 4)
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
            if c == 5:  # TF-IDF column
                score = tfidf_sim
                if score >= 0.7:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= 0.4:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

            if use_bert and c == 6:  # BERT column
                score = bert_sim if bert_sim is not None else 0
                if score >= 0.7:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif score >= 0.4:
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
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
        ["TF-IDF Similarity", ""],
        ["Mean", round(np.mean(tfidf_scores), 4) if tfidf_scores else 0],
        ["Median", round(np.median(tfidf_scores), 4) if tfidf_scores else 0],
        ["Std Dev", round(np.std(tfidf_scores), 4) if tfidf_scores else 0],
        ["Min", round(np.min(tfidf_scores), 4) if tfidf_scores else 0],
        ["Max", round(np.max(tfidf_scores), 4) if tfidf_scores else 0],
    ]

    if use_bert and bert_scores:
        summary_data.extend([
            ["", ""],
            ["BERT Similarity", ""],
            ["Mean", round(np.mean(bert_scores), 4)],
            ["Median", round(np.median(bert_scores), 4)],
            ["Std Dev", round(np.std(bert_scores), 4)],
            ["Min", round(np.min(bert_scores), 4)],
            ["Max", round(np.max(bert_scores), 4)],
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
    print(f"\nTF-IDF Similarity:")
    print(f"  Mean: {np.mean(tfidf_scores):.4f}")
    print(f"  Median: {np.median(tfidf_scores):.4f}")
    print(f"  Std Dev: {np.std(tfidf_scores):.4f}")

    if use_bert and bert_scores:
        print(f"\nBERT Similarity:")
        print(f"  Mean: {np.mean(bert_scores):.4f}")
        print(f"  Median: {np.median(bert_scores):.4f}")
        print(f"  Std Dev: {np.std(bert_scores):.4f}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate test evaluation report with similarity scores")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON from test_inference.py")
    parser.add_argument("--output", type=str, default="test_evaluation_report.xlsx",
                        help="Output Excel file path")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BERT similarity (faster, uses only TF-IDF)")

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} predictions")

    # Generate report
    create_excel_report(results, args.output, use_bert=not args.no_bert)


if __name__ == "__main__":
    main()
