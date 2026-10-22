import pandas as pd

CSV_PATH = "results/model2_evaluation.csv"
THRESHOLD = 0.025

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[
        (df["status"] == "SUCCESS")
        & df["fake_score"].notna()
    ].copy()
    
    df["true_label"] = df["true_label"].str.upper()
    df["predicted_at_threshold"] = df["fake_score"].apply(
        lambda score: "AI" if score >= THRESHOLD else "HUMAN"
    )
    
    errors = df[df["true_label"] != df["predicted_at_threshold"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: (
            "False Positive"
            if row["true_label"] == "HUMAN"
            else "False Negative"
        ),
        axis=1,
    )
    
    print(f"Threshold: {THRESHOLD:.3f}")
    print(f"Correct predictions: {len(df) - len(errors)}/{len(df)}")
    print(f"Misclassified videos: {len(errors)}\n")
    
    if not errors.empty:
        columns = ["file_path", "true_label", "predicted_at_threshold", "fake_score", "error_type"]
        print(errors[columns].to_string(index=False))
    else:
        print("No errors found at this threshold.")

if __name__ == "__main__":
    main()