import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detectors.text.detector import TextAIDetector

def run_extended_evaluation():
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "data" / "text" / "evaluation"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = TextAIDetector(model_name="mujian2026/multilingual-ai-text-detector")

    records = []

    def process_csv(file_path, actual_label):
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist.")
            return

        df = pd.read_csv(file_path)
        if "text" not in df.columns:
            print(f"Error: 'text' column missing in {file_path.name}")
            return

        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue

            category = row.get("category", "unspecified")
            source = row.get("source", "unspecified")

            result = detector.detect(text)

            records.append({
                "row_id": idx + 1,
                "actual_label": actual_label,
                "category": category,
                "source": source,
                "ai_score": result.ai_score,
                "human_score": result.human_score,
                "prediction": result.prediction,
                "chunks": result.chunks_analyzed,
                "word_count": len(text.split()),
                "text_snippet": text[:70] + "..."
            })

    process_csv(base_dir / "human" / "human.csv", "HUMAN")
    process_csv(base_dir / "ai" / "ai.csv", "AI")

    df_results = pd.DataFrame(records, columns=[
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
    ])
    output_path = results_dir / "extended_multilingual_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"--> Saved evaluation raw data ({len(df_results)} rows) to: {output_path}")

    # Metrics Computation
    tp = len(df_results[(df_results["actual_label"] == "AI") & (df_results["prediction"] == "likely_ai_generated")])
    fp = len(df_results[(df_results["actual_label"] == "HUMAN") & (df_results["prediction"] == "likely_ai_generated")])
    tn = len(df_results[(df_results["actual_label"] == "HUMAN") & (df_results["prediction"] == "likely_human")])
    fn = len(df_results[(df_results["actual_label"] == "AI") & (df_results["prediction"] == "likely_human")])
    unc = len(df_results[df_results["prediction"] == "uncertain"])

    total = len(df_results)
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n================ Extended Dataset Evaluation Results ================")
    print(f"Total Samples Evaluated : {total}")
    print(f"True Positives (TP)      : {tp} | False Positives (FP) : {fp}")
    print(f"True Negatives (TN)      : {tn} | False Negatives (FN) : {fn}")
    print(f"Uncertain Predictions   : {unc}")
    print(f"Accuracy                 : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision                : {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall                   : {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1 Score                 : {f1:.4f} ({f1*100:.2f}%)")
    print("=======================================================================")

    # Performance Breakdown by Category
    if "category" in df_results.columns:
        print("\n---------------- Accuracy Breakdown by Category ----------------")
        for cat, grp in df_results.groupby("category"):
            correct = len(grp[
                ((grp["actual_label"] == "AI") & (grp["prediction"] == "likely_ai_generated")) |
                ((grp["actual_label"] == "HUMAN") & (grp["prediction"] == "likely_human"))
            ])
            cat_total = len(grp)
            cat_acc = correct / cat_total if cat_total > 0 else 0
            print(f"Category: {cat:<20} | Samples: {cat_total:<3} | Accuracy: {cat_acc:.2%}")
        print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_extended_evaluation()