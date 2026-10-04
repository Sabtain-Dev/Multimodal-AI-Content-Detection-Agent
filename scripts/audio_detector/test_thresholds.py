import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path("results")
M1_CSV = RESULTS_DIR / "audio_detector_model_1_results.csv"
M2_CSV = RESULTS_DIR / "audio_detector_model_2_results.csv"

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

MODEL_1_NAME = "Sayantan090/audio-fake-detector"
MODEL_2_NAME = "Mahmoud59/wav2vec2-fake-audio-detector"


def compute_metrics_for_threshold(df: pd.DataFrame, threshold: float, is_model_1: bool) -> dict:
    """
    Evaluates metrics when classifying as AI only if the raw score associated 
    with the 'AI' prediction meets or exceeds the specified threshold.
    """
    adjusted_preds = []

    for _, row in df.iterrows():
        raw_pred = row["predicted_label"]
        raw_score = float(row["score"])

        # Determine raw AI score
        if raw_pred == "AI":
            ai_score = raw_score
        else:
            ai_score = 1.0 - raw_score

        # Apply threshold decision rule
        if ai_score >= threshold:
            adjusted_preds.append("AI")
        else:
            adjusted_preds.append("HUMAN")

    actuals = df["actual_label"]
    preds = pd.Series(adjusted_preds)

    tp = len(df[(actuals == "AI") & (preds == "AI")])
    tn = len(df[(actuals == "HUMAN") & (preds == "HUMAN")])
    fp = len(df[(actuals == "HUMAN") & (preds == "AI")])
    fn = len(df[(actuals == "AI") & (preds == "HUMAN")])

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Threshold": threshold,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "Specificity": round(specificity * 100, 2),
        "F1": round(f1 * 100, 2),
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn
    }


def main():
    print("=" * 60)
    print("AUDIO DETECTOR THRESHOLD SWEEP ANALYSIS")
    print("=" * 60)

    if not M1_CSV.exists() or not M2_CSV.exists():
        print(f"Error: Could not locate result CSV files in {RESULTS_DIR}")
        sys.exit(1)

    df_m1 = pd.read_csv(M1_CSV)
    df_m2 = pd.read_csv(M2_CSV)

    m1_sweep = [compute_metrics_for_threshold(df_m1, t, is_model_1=True) for t in THRESHOLDS]
    m2_sweep = [compute_metrics_for_threshold(df_m2, t, is_model_1=False) for t in THRESHOLDS]

    df_m1_sweep = pd.DataFrame(m1_sweep)
    df_m2_sweep = pd.DataFrame(m2_sweep)

    print(f"\nMODEL 1 THRESHOLD SWEEP ({MODEL_1_NAME}):")
    print(df_m1_sweep[["Threshold", "Accuracy", "Precision", "Recall", "Specificity", "F1", "FP", "FN"]].to_string(index=False))

    print(f"\nMODEL 2 THRESHOLD SWEEP ({MODEL_2_NAME}):")
    print(df_m2_sweep[["Threshold", "Accuracy", "Precision", "Recall", "Specificity", "F1", "FP", "FN"]].to_string(index=False))

    # Save outputs to CSV
    m1_out = RESULTS_DIR / "audio_detector_model_1_thresholds.csv"
    m2_out = RESULTS_DIR / "audio_detector_model_2_thresholds.csv"
    
    df_m1_sweep.to_csv(m1_out, index=False)
    df_m2_sweep.to_csv(m2_out, index=False)

    print(f"\nSaved threshold sweeps to:")
    print(f" - {m1_out}")
    print(f" - {m2_out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()