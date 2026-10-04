import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path("results")
M1_CSV = RESULTS_DIR / "audio_detector_model_1_results.csv"
M2_CSV = RESULTS_DIR / "audio_detector_model_2_results.csv"


def main():
    print("=" * 60)
    print("AUDIO MODELS DISAGREEMENT & ERROR ANALYSIS")
    print("=" * 60)

    if not M1_CSV.exists() or not M2_CSV.exists():
        print(f"Error: Required CSV files not found in {RESULTS_DIR}")
        print("Please ensure evaluate_audio_models.py has been executed.")
        sys.exit(1)

    df_m1 = pd.read_csv(M1_CSV)
    df_m2 = pd.read_csv(M2_CSV)

    # Merge results on file name and actual label
    merged = pd.merge(
        df_m1,
        df_m2,
        on=["file", "actual_label"],
        suffixes=("_m1", "_m2")
    )

    total_samples = len(merged)
    print(f"Total Merged Evaluation Samples: {total_samples}")

    # Agreement categories
    both_correct = merged[merged["correct_m1"] & merged["correct_m2"]]
    both_wrong = merged[(~merged["correct_m1"]) & (~merged["correct_m2"])]
    m1_correct_m2_wrong = merged[merged["correct_m1"] & (~merged["correct_m2"])]
    m2_correct_m1_wrong = merged[(~merged["correct_m1"]) & merged["correct_m2"]]

    disagreements = merged[merged["predicted_label_m1"] != merged["predicted_label_m2"]]

    print("\n--- PERFORMANCE BREAKDOWN ---")
    print(f"Both Models Correct        : {len(both_correct)} ({len(both_correct)/total_samples*100:.2f}%)")
    print(f"Both Models Wrong          : {len(both_wrong)} ({len(both_wrong)/total_samples*100:.2f}%)")
    print(f"Model 1 Correct / Model 2 Wrong : {len(m1_correct_m2_wrong)} ({len(m1_correct_m2_wrong)/total_samples*100:.2f}%)")
    print(f"Model 2 Correct / Model 1 Wrong : {len(m2_correct_m1_wrong)} ({len(m2_correct_m1_wrong)/total_samples*100:.2f}%)")
    print(f"Total Disagreements        : {len(disagreements)} ({len(disagreements)/total_samples*100:.2f}%)")

    # Detailed view of disagreements
    if not disagreements.empty:
        print("\n--- SAMPLE DISAGREEMENTS DETAIL ---")
        disp_cols = [
            "file", "actual_label", 
            "predicted_label_m1", "score_m1", 
            "predicted_label_m2", "score_m2"
        ]
        print(disagreements[disp_cols].to_string(index=False))

    # Detailed view of cases where Model 2 recovers correct answers missed by Model 1
    if not m2_correct_m1_wrong.empty:
        print("\n--- SAMPLES RECOVERED BY MODEL 2 (Model 1 Failed, Model 2 Correct) ---")
        disp_cols_m2 = [
            "file", "actual_label", 
            "predicted_label_m1", "score_m1", 
            "predicted_label_m2", "score_m2"
        ]
        print(m2_correct_m1_wrong[disp_cols_m2].to_string(index=False))

    # Save detailed analysis output
    output_path = RESULTS_DIR / "audio_detector_disagreement_analysis.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved detailed disagreement analysis to {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()