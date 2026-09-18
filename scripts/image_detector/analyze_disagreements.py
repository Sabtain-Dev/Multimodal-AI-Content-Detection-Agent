import sys
from pathlib import Path
import pandas as pd

DISAGREEMENT_CSV = Path("results/image_detector_disagreements.csv")


def analyze_disagreements():
    print("=" * 60)
    print("ANALYZING INTERNAL DISAGREEMENT METADATA")
    print("=" * 60)

    if not DISAGREEMENT_CSV.exists():
        print("Error: Disagreement metadata file missing. Run compare_image_models.py first.")
        sys.exit(1)

    df = pd.read_csv(DISAGREEMENT_CSV)
    total_disagreements = len(df)

    if total_disagreements == 0:
        print("No disagreement cases found under active thresholds.")
        return

    m2_correct = df[df["primary_model_label"] == df["actual_label"]]
    m1_correct = df[df["secondary_model_label"] == df["actual_label"]]
    neither_correct = df[(df["primary_model_label"] != df["actual_label"]) & (df["secondary_model_label"] != df["actual_label"])]

    print(f"Total Disagreements Analyzed      : {total_disagreements}")
    print(f"Primary Winner (Model 2 Correct) : {len(m2_correct)} ({len(m2_correct)/total_disagreements*100:.2f}%)")
    print(f"Secondary Winner (Model 1 Correct): {len(m1_correct)} ({len(m1_correct)/total_disagreements*100:.2f}%)")
    print(f"Neither Model Correct             : {len(neither_correct)}")
    print("-" * 60)

    print("\nSAMPLE DISAGREEMENT METADATA RECORDS:")
    for idx, row in df.head(10).iterrows():
        winner = "Primary (M2)" if row["primary_model_label"] == row["actual_label"] else "Secondary (M1)"
        print(f"- {row['image_path']} | Actual: {row['actual_label']} | Final Output: {row['final_label']} | Winner: {winner}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyze_disagreements()