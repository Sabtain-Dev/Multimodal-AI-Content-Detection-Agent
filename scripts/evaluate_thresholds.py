import sys
from pathlib import Path
import pandas as pd

def run_threshold_sweep():
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "results" / "tmr_results.csv"

    if not input_path.exists():
        print(f"Error: Raw results file not found at {input_path}. Run evaluate_text_detector.py first.")
        return

    df = pd.read_csv(input_path)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    sweep_records = []

    for t in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0

        for _, row in df.iterrows():
            pred = "AI" if row["ai_score"] >= t else "HUMAN"
            actual = row["actual_label"]

            if actual == "AI" and pred == "AI":
                tp += 1
            elif actual == "HUMAN" and pred == "AI":
                fp += 1
            elif actual == "HUMAN" and pred == "HUMAN":
                tn += 1
            elif actual == "AI" and pred == "HUMAN":
                fn += 1

        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        sweep_records.append({
            "Threshold": t,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn
        })

    results_df = pd.DataFrame(sweep_records)
    print("\n================ THRESHOLD SWEEP RESULTS ================")
    print(results_df.to_string(index=False))
    print("=========================================================")

    output_path = project_root / "results" / "threshold_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved threshold comparison matrix to: {output_path}")

if __name__ == "__main__":
    run_threshold_sweep()