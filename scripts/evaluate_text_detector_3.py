import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector


MODEL_NAME = "ShantanuT01/gradient-ai-text-detector"
AI_THRESHOLD = 0.50


def run_evaluation():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / "text" / "evaluation"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = TextAIDetector(model_name=MODEL_NAME, threshold=AI_THRESHOLD)
    records = []

    for actual_label, folder_name in (("HUMAN", "human"), ("AI", "ai")):
        file_path = dataset_dir / folder_name / f"{folder_name}.csv"
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist.")
            continue

        dataframe = pd.read_csv(file_path)
        if "text" not in dataframe.columns:
            print(f"Error: 'text' column missing in {file_path.name}")
            continue

        for row_index, row in dataframe.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue

            result = detector.detect(text)
            records.append(
                {
                    "row_id": row_index + 1,
                    "actual_label": actual_label,
                    "category": row.get("category", "unspecified"),
                    "source": row.get("source", "unspecified"),
                    "ai_score": result.ai_score,
                    "human_score": result.human_score,
                    "prediction": result.prediction,
                    "chunks": result.chunks_analyzed,
                    "word_count": len(text.split()),
                    "text_snippet": text[:70] + "...",
                }
            )

    results = pd.DataFrame(records)
    output_path = results_dir / "extended_gradient_results.csv"
    results.to_csv(output_path, index=False)
    print(f"--> Saved evaluation raw data ({len(results)} rows) to: {output_path}")

    true_positive = len(results[(results["actual_label"] == "AI") & (results["prediction"] == "likely_ai_generated")])
    false_positive = len(results[(results["actual_label"] == "HUMAN") & (results["prediction"] == "likely_ai_generated")])
    true_negative = len(results[(results["actual_label"] == "HUMAN") & (results["prediction"] == "likely_human")])
    false_negative = len(results[(results["actual_label"] == "AI") & (results["prediction"] == "likely_human")])

    total = len(results)
    accuracy = (true_positive + true_negative) / total if total else 0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n================ Gradient Detector Evaluation ================")
    print(f"Total samples evaluated : {total}")
    print(f"True positives (TP)     : {true_positive}")
    print(f"False positives (FP)    : {false_positive}")
    print(f"True negatives (TN)     : {true_negative}")
    print(f"False negatives (FN)    : {false_negative}")
    print(f"Accuracy                : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision               : {precision:.4f} ({precision:.2%})")
    print(f"Recall                  : {recall:.4f} ({recall:.2%})")
    print(f"F1 score                : {f1:.4f} ({f1:.2%})")
    print("==============================================================")


if __name__ == "__main__":
    run_evaluation()