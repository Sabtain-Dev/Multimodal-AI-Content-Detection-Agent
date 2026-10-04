import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path("results")
M1_CSV = RESULTS_DIR / "audio_detector_model_1_results.csv"
M2_CSV = RESULTS_DIR / "audio_detector_model_2_results.csv"

# Threshold sweep space for both models
THRESHOLDS = [0.70, 0.80, 0.90, 0.95]

MODEL_1_NAME = "Sayantan090/audio-fake-detector"
MODEL_2_NAME = "Mahmoud59/wav2vec2-fake-audio-detector"


def extract_raw_ai_score(row: pd.Series) -> float:
    """
    Extracts the score associated with AI prediction.
    """
    pred_label = row["predicted_label"]
    score = float(row["score"])
    return score if pred_label == "AI" else (1.0 - score)


def evaluate_ensemble_grid(df_m1: pd.DataFrame, df_m2: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        df_m1[["file", "actual_label", "predicted_label", "score"]],
        df_m2[["file", "actual_label", "predicted_label", "score"]],
        on=["file", "actual_label"],
        suffixes=("_m1", "_m2")
    )

    ensemble_results = []

    for t1 in THRESHOLDS:
        for t2 in THRESHOLDS:
            preds = []
            agreements = []

            for _, row in merged.iterrows():
                m1_ai_score = extract_raw_ai_score(row.iloc[:4] if "predicted_label_m1" not in row else pd.Series({
                    "predicted_label": row["predicted_label_m1"],
                    "score": row["score_m1"]
                }))
                
                m2_ai_score = extract_raw_ai_score(row.iloc[3:] if "predicted_label_m2" not in row else pd.Series({
                    "predicted_label": row["predicted_label_m2"],
                    "score": row["score_m2"]
                }))

                m1_decision = "AI" if m1_ai_score >= t1 else "HUMAN"
                m2_decision = "AI" if m2_ai_score >= t2 else "HUMAN"

                is_agree = (m1_decision == m2_decision)
                agreements.append(is_agree)

                if is_agree:
                    preds.append(m1_decision)
                else:
                    preds.append("UNCERTAIN")

            merged_eval = merged.copy()
            merged_eval["ensemble_pred"] = preds
            merged_eval["agreed"] = agreements

            total_samples = len(merged_eval)
            agreed_df = merged_eval[merged_eval["agreed"]]
            uncertain_df = merged_eval[~merged_eval["agreed"]]

            agreed_count = len(agreed_df)
            uncertain_count = len(uncertain_df)
            coverage = agreed_count / total_samples if total_samples > 0 else 0.0
            uncertainty_rate = uncertain_count / total_samples if total_samples > 0 else 0.0

            # Compute classification metrics over all samples
            actuals = merged_eval["actual_label"]
            
            tp = len(merged_eval[(actuals == "AI") & (merged_eval["ensemble_pred"] == "AI")])
            tn = len(merged_eval[(actuals == "HUMAN") & (merged_eval["ensemble_pred"] == "HUMAN")])
            fp = len(merged_eval[(actuals == "HUMAN") & (merged_eval["ensemble_pred"] == "AI")])
            fn = len(merged_eval[(actuals == "AI") & (merged_eval["ensemble_pred"] == "HUMAN")])

            overall_accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Consensus accuracy: accuracy on agreed samples only
            consensus_correct = len(agreed_df[agreed_df["actual_label"] == agreed_df["ensemble_pred"]])
            consensus_accuracy = consensus_correct / agreed_count if agreed_count > 0 else 0.0

            ensemble_results.append({
                "T1_M1": t1,
                "T2_M2": t2,
                "Accuracy": round(overall_accuracy * 100, 2),
                "Precision": round(precision * 100, 2),
                "Recall": round(recall * 100, 2),
                "Specificity": round(specificity * 100, 2),
                "F1": round(f1 * 100, 2),
                "Coverage": round(coverage * 100, 2),
                "Uncertainty_Rate": round(uncertainty_rate * 100, 2),
                "Consensus_Accuracy": round(consensus_accuracy * 100, 2),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Uncertain_Count": uncertain_count
            })

    return pd.DataFrame(ensemble_results)


def main():
    print("=" * 60)
    print("AUDIO DETECTOR ENSEMBLE GRID EVALUATION")
    print("=" * 60)

    if not M1_CSV.exists() or not M2_CSV.exists():
        print(f"Error: Required CSV results not found in {RESULTS_DIR}")
        sys.exit(1)

    df_m1 = pd.read_csv(M1_CSV)
    df_m2 = pd.read_csv(M2_CSV)

    ensemble_df = evaluate_ensemble_grid(df_m1, df_m2)

    # Sort grid by overall Accuracy then F1 score
    sorted_df = ensemble_df.sort_values(by=["Accuracy", "F1", "Consensus_Accuracy"], ascending=False)

    print("\nTOP ENSEMBLE THRESHOLD CONFIGURATIONS:")
    print(sorted_df.head(10)[
        ["T1_M1", "T2_M2", "Accuracy", "Precision", "Recall", "Specificity", "F1", "Coverage", "Uncertainty_Rate", "Consensus_Accuracy"]
    ].to_string(index=False))

    output_path = RESULTS_DIR / "audio_detector_ensemble_evaluation.csv"
    ensemble_df.to_csv(output_path, index=False)

    print(f"\nSaved complete ensemble grid evaluation to {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()