import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector

MODEL_NAME = "Oxidane/tmr-ai-text-detector"
EXPECTED_SAMPLE_COUNT = 101


def run_tmr_evaluation():
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "data" / "text" / "evaluation"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = TextAIDetector(model_name=MODEL_NAME)
    records = []

    def process_csv(file_path: Path, actual_label: str):
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist.")
            return

        dataframe = pd.read_csv(file_path)
        if "text" not in dataframe.columns:
            print(f"Error: 'text' column missing in {file_path.name}")
            return

        for row_number in range(len(dataframe)):
            row = dataframe.iloc[row_number]
            text = str(row["text"]).strip()
            if not text:
                continue

            result = detector.detect(text)
            records.append({
                "row_id": row_number + 1,
                "actual_label": actual_label,
                "category": row.get("category", "unspecified"),
                "source": row.get("source", "unspecified"),
                "ai_score": result.ai_score,
                "human_score": result.human_score,
                "prediction": result.prediction,
                "chunks": result.chunks_analyzed,
                "word_count": len(text.split()),
                "text_snippet": text[:70] + "...",
            })

    process_csv(base_dir / "human" / "human.csv", "HUMAN")
    process_csv(base_dir / "ai" / "ai.csv", "AI")

    result_columns = [
        "row_id",
        "actual_label",
        "category",
        "source",
        "ai_score",
        "human_score",
        "prediction",
        "chunks",
        "word_count",
        "text_snippet",
    ]
    results = pd.DataFrame(records, columns=result_columns)
    output_path = results_dir / "extended_tmr_results.csv"
    results.to_csv(output_path, index=False)

    if len(results) != EXPECTED_SAMPLE_COUNT:
        print(
            f"Warning: Evaluated {len(results)} samples; "
            f"expected {EXPECTED_SAMPLE_COUNT}."
        )

    true_positive = len(
        results[
            (results["actual_label"] == "AI")
            & (results["prediction"] == "likely_ai_generated")
        ]
    )
    false_positive = len(
        results[
            (results["actual_label"] == "HUMAN")
            & (results["prediction"] == "likely_ai_generated")
        ]
    )
    true_negative = len(
        results[
            (results["actual_label"] == "HUMAN")
            & (results["prediction"] == "likely_human")
        ]
    )
    false_negative = len(
        results[
            (results["actual_label"] == "AI")
            & (results["prediction"] == "likely_human")
        ]
    )
    uncertain = len(results[results["prediction"] == "uncertain"])

    total = len(results)
    accuracy = (true_positive + true_negative) / total if total else 0
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"--> Model: {MODEL_NAME}")
    print(f"--> Saved {total} evaluation rows to: {output_path}")
    print("\n================ TMR Evaluation Results ================")
    print(f"Total samples evaluated : {total}")
    print(f"True positives (TP)    : {true_positive}")
    print(f"False positives (FP)   : {false_positive}")
    print(f"True negatives (TN)    : {true_negative}")
    print(f"False negatives (FN)   : {false_negative}")
    print(f"Uncertain predictions  : {uncertain}")
    print(f"Accuracy               : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision              : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall                 : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 score               : {f1:.4f} ({f1 * 100:.2f}%)")
    print("========================================================")


if __name__ == "__main__":
    run_tmr_evaluation()
