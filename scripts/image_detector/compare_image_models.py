import sys
from pathlib import Path
import pandas as pd

CSV_MODEL_1 = Path("results/image_detector_1_results.csv")
CSV_MODEL_2 = Path("results/image_detector_2_results.csv")
OUTPUT_ENSEMBLE_CSV = Path("results/image_detector_ensemble_results.csv")
OUTPUT_DISAGREEMENTS_CSV = Path("results/image_detector_disagreements.csv")

M1_THRESHOLD = 0.70
M2_THRESHOLD = 0.50


def choose_final_label(model_1_label: str, model_2_label: str) -> str:
    if model_1_label == model_2_label:
        return model_1_label
    return model_2_label


def derive_confidence_indicator(primary_score: float, models_agree: bool) -> str:
    if not models_agree:
        return "Moderate"
    if primary_score >= 0.90:
        return "High"
    elif primary_score >= 0.70:
        return "Moderate"
    else:
        return "Low"


def calculate_binary_metrics(df, pred_col):
    actuals = df["actual_label"]
    preds = df[pred_col]

    tp = len(df[(actuals == "AI") & (preds == "AI")])
    tn = len(df[(actuals == "HUMAN") & (preds == "HUMAN")])
    fp = len(df[(actuals == "HUMAN") & (preds == "AI")])
    fn = len(df[(actuals == "AI") & (preds == "HUMAN")])

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def run_ensemble_comparison():
    print("=" * 60)
    print("RUNNING PRODUCTION IMAGE ENSEMBLE EVALUATION")
    print(f"Active Thresholds -> Model 1: {M1_THRESHOLD:.2f} | Model 2: {M2_THRESHOLD:.2f}")
    print("=" * 60)

    if not CSV_MODEL_1.exists() or not CSV_MODEL_2.exists():
        print("Error: Base evaluation CSV files missing.")
        sys.exit(1)

    df1 = pd.read_csv(CSV_MODEL_1)
    df2 = pd.read_csv(CSV_MODEL_2)

    merged = pd.merge(
        df1[["image_path", "actual_label", "ai_score"]],
        df2[["image_path", "ai_score"]],
        on="image_path",
        suffixes=("_m1", "_m2")
    )

    if len(merged) == 0:
        print("Error: No matching image entries found.")
        sys.exit(1)

    ensemble_records = []
    for _, row in merged.iterrows():
        s1 = row["ai_score_m1"]
        s2 = row["ai_score_m2"]

        p1 = "AI" if s1 >= M1_THRESHOLD else "HUMAN"
        p2 = "AI" if s2 >= M2_THRESHOLD else "HUMAN"

        models_agree = (p1 == p2)
        final_label = choose_final_label(p1, p2)
        decision_mode = "consensus" if models_agree else "model_2_primary_fallback"

        model_score = s2 if final_label == "AI" else (1.0 - s2)
        confidence_indicator = derive_confidence_indicator(model_score, models_agree)

        ensemble_records.append({
            "image_path": row["image_path"],
            "actual_label": row["actual_label"],
            "final_label": final_label,
            "model_score": round(model_score, 4),
            "confidence_indicator": confidence_indicator,
            "models_agree": models_agree,
            "decision_mode": decision_mode,
            "primary_model_label": p2,
            "primary_model_score": s2,
            "secondary_model_label": p1,
            "secondary_model_score": s1
        })

    df_out = pd.DataFrame(ensemble_records)
    OUTPUT_ENSEMBLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_ENSEMBLE_CSV, index=False)

    df_disagreements = df_out[~df_out["models_agree"]].copy()
    df_disagreements.to_csv(OUTPUT_DISAGREEMENTS_CSV, index=False)

    m = calculate_binary_metrics(df_out, "final_label")

    print(f"Saved complete ensemble results to: {OUTPUT_ENSEMBLE_CSV}")
    print(f"Saved {len(df_disagreements)} disagreement metadata cases to: {OUTPUT_DISAGREEMENTS_CSV}\n")

    print("--- PRODUCTION STRICT MODE EVALUATION METRICS ---")
    print(f"Total Evaluated Samples       : {len(df_out)}")
    print(f"True Positives (AI -> AI)     : {m['tp']}")
    print(f"True Negatives (HUMAN -> HUMAN): {m['tn']}")
    print(f"False Positives (HUMAN -> AI) : {m['fp']}")
    print(f"False Negatives (AI -> HUMAN) : {m['fn']}")
    print("-" * 42)
    print(f"Accuracy                      : {m['accuracy'] * 100:.2f}%")
    print(f"Precision                     : {m['precision'] * 100:.2f}%")
    print(f"Recall                        : {m['recall'] * 100:.2f}%")
    print(f"F1-Score                      : {m['f1'] * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ensemble_comparison()