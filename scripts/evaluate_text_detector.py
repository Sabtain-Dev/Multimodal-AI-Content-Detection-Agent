import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector


def run_evaluation_and_save():
    project_root = Path(__file__).resolve().parents[1]

    base_dir = project_root / "data" / "text" / "evaluation"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    human_dir = base_dir / "human"
    ai_dir = base_dir / "ai"

    # --------------------------------------------------
    # Check Evaluation Directories
    # --------------------------------------------------
    if not human_dir.exists() or not ai_dir.exists():
        print(f"Error: Evaluation directories not found at {base_dir}")
        print("Please create:")
        print("  data/text/evaluation/human/")
        print("  data/text/evaluation/ai/")
        return

    detector = TextAIDetector()

    records = []

    # Confusion matrix counters
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # --------------------------------------------------
    # Process Evaluation Files
    # --------------------------------------------------
    def process_files(files_dir, actual_label):
        nonlocal tp, fp, tn, fn

        files = sorted(files_dir.glob("*.csv"))

        if not files:
            print(f"Warning: No CSV files found in {files_dir}")
            return

        for file_path in files:
            df = pd.read_csv(file_path)

            if "text" not in df.columns:
                print(
                    f"Warning: 'text' column not found in "
                    f"{file_path.name}. Skipping file."
                )
                continue

            for index, row in df.iterrows():
                text = str(row["text"]).strip()

                if not text:
                    continue

                result = detector.detect(text)

                model_prediction = (
                    "AI"
                    if result.prediction == "likely_ai_generated"
                    else "HUMAN"
                )

                # ------------------------------------------
                # Update Confusion Matrix
                # ------------------------------------------
                if actual_label == "AI":
                    if model_prediction == "AI":
                        tp += 1
                    else:
                        fn += 1
                else:
                    if model_prediction == "AI":
                        fp += 1
                    else:
                        tn += 1

                # ------------------------------------------
                # Calculate Text Statistics
                # ------------------------------------------
                word_count = len(text.split())

                token_count = sum(
                    chunk.token_count
                    for chunk in result.chunk_results
                )

                # ------------------------------------------
                # Save Detailed Sample Result
                # ------------------------------------------
                records.append(
                    {
                        "file_name": file_path.name,
                        "row_index": index + 1,
                        "actual_label": actual_label,
                        "prediction": model_prediction,
                        "ai_score": result.ai_score,
                        "human_score": result.human_score,
                        "chunks_analyzed": result.chunks_analyzed,
                        "word_count": word_count,
                        "token_count": token_count,
                        "text_snippet": (
                            text[:80] + "..."
                            if len(text) > 80
                            else text
                        ),
                    }
                )

    # --------------------------------------------------
    # Evaluate Human Samples
    # --------------------------------------------------
    print("\nProcessing HUMAN samples...")
    process_files(human_dir, "HUMAN")

    # --------------------------------------------------
    # Evaluate AI Samples
    # --------------------------------------------------
    print("Processing AI samples...")
    process_files(ai_dir, "AI")

    # --------------------------------------------------
    # Check Results
    # --------------------------------------------------
    total = tp + fp + tn + fn

    if total == 0:
        print("\nNo valid evaluation samples found.")
        return

    # --------------------------------------------------
    # Calculate Performance Metrics
    # --------------------------------------------------
    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --------------------------------------------------
    # Save Detailed Results
    # --------------------------------------------------
    results_df = pd.DataFrame(records)

    detailed_results_path = results_dir / "tmr_results.csv"
    results_df.to_csv(
        detailed_results_path,
        index=False
    )

    # --------------------------------------------------
    # Save Evaluation Summary
    # --------------------------------------------------
    summary_df = pd.DataFrame(
        [
            {
                "total_samples": total,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        ]
    )

    summary_path = results_dir / "evaluation_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # Display Final Results
    # --------------------------------------------------
    print("\n====================================================")
    print("                  EVALUATION RESULTS")
    print("====================================================")

    print(f"Total Samples   : {total}")
    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"True Negatives  : {tn}")
    print(f"False Negatives : {fn}")

    print("\n---------------- Performance -----------------------")

    print(
        f"Accuracy        : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision       : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall          : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score        : {f1_score:.4f} "
        f"({f1_score * 100:.2f}%)"
    )

    print("\n---------------- Interpretation --------------------")

    print(f"Human Samples   : {tn + fp}")
    print(f"  Correct Human : {tn}")
    print(f"  Wrongly AI    : {fp}")

    print(f"\nAI Samples      : {tp + fn}")
    print(f"  Correct AI    : {tp}")
    print(f"  Wrongly Human : {fn}")

    print("\n---------------- Output Files ----------------------")

    print(f"Detailed Results : {detailed_results_path}")
    print(f"Evaluation Summary: {summary_path}")

    print("====================================================")


if __name__ == "__main__":
    run_evaluation_and_save()