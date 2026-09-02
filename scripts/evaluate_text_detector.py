import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector


def run_evaluation():
    base_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "text"
        / "evaluation"
    )

    human_dir = base_dir / "human"
    ai_dir = base_dir / "ai"

    if not human_dir.exists() or not ai_dir.exists():
        print(f"Error: Evaluation directories not found at {base_dir}")
        print("Please create:")
        print("  data/text/evaluation/human/")
        print("  data/text/evaluation/ai/")
        return

    detector = TextAIDetector()

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # --------------------------------------------------
    # Evaluate Human Samples
    # --------------------------------------------------
    print("\n====================================================")
    print("              HUMAN SAMPLE EVALUATION")
    print("====================================================")

    human_files = sorted(human_dir.glob("*.csv"))

    if not human_files:
        print("No human CSV files found.")

    for file_path in human_files:
        df = pd.read_csv(file_path)

        if "text" not in df.columns:
            print(f"Error: 'text' column not found in {file_path.name}")
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

            actual_label = "HUMAN"

            if model_prediction == "AI":
                fp += 1
                status = "FP"
            else:
                tn += 1
                status = "TN"

            print(
                f"\n[{status}] {file_path.name} | Row {index + 1}\n"
                f"     Actual    : {actual_label}\n"
                f"     Model     : {model_prediction}\n"
                f"     AI Score  : {result.ai_score:.4f}"
            )

    # --------------------------------------------------
    # Evaluate AI Samples
    # --------------------------------------------------
    print("\n====================================================")
    print("                AI SAMPLE EVALUATION")
    print("====================================================")

    ai_files = sorted(ai_dir.glob("*.csv"))

    if not ai_files:
        print("No AI CSV files found.")

    for file_path in ai_files:
        df = pd.read_csv(file_path)

        if "text" not in df.columns:
            print(f"Error: 'text' column not found in {file_path.name}")
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

            actual_label = "AI"

            if model_prediction == "AI":
                tp += 1
                status = "TP"
            else:
                fn += 1
                status = "FN"

            print(
                f"\n[{status}] {file_path.name} | Row {index + 1}\n"
                f"     Actual    : {actual_label}\n"
                f"     Model     : {model_prediction}\n"
                f"     AI Score  : {result.ai_score:.4f}"
            )

    # --------------------------------------------------
    # Calculate Metrics
    # --------------------------------------------------
    total = tp + fp + tn + fn

    if total == 0:
        print("\nNo evaluation samples found.")
        return

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

    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --------------------------------------------------
    # Final Results
    # --------------------------------------------------
    print("\n\n====================================================")
    print("                 EVALUATION RESULTS")
    print("====================================================")

    print(f"Total Samples   : {total}")
    print(f"True Positives  : {tp}")
    print(f"False Positives : {fp}")
    print(f"True Negatives  : {tn}")
    print(f"False Negatives : {fn}")

    print("\n---------------- Performance -----------------------")

    print(f"Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision       : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall          : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score        : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\n---------------- Interpretation --------------------")

    print(f"Human Samples   : {tn + fp}")
    print(f"  Correct Human : {tn}")
    print(f"  Wrongly AI    : {fp}")

    print(f"\nAI Samples      : {tp + fn}")
    print(f"  Correct AI    : {tp}")
    print(f"  Wrongly Human : {fn}")

    print("====================================================")


if __name__ == "__main__":
    run_evaluation()