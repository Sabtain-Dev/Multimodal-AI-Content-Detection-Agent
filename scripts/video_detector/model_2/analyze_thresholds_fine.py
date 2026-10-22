import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CSV_PATH = "results/model2_evaluation.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[
        (df["status"] == "SUCCESS")
        & df["fake_score"].notna()
    ].copy()
    
    y_true = (df["true_label"].str.upper() == "AI").astype(int)
    scores = df["fake_score"].astype(float)
    
    results = []
    # Fine-grained sweep from 0.005 to 0.050.
    for threshold in [i / 1000 for i in range(5, 51)]:
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        
        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })
        
    result_df = pd.DataFrame(results)
    result_df.to_csv("results/model2_threshold_sweep.csv", index=False)
    
    print("\nTop 10 thresholds by F1:")
    print(
        result_df.sort_values(
            ["f1", "accuracy"], ascending=False
        ).head(10).to_string(index=False, float_format="%.3f")
    )
    
    print("\nTop thresholds with zero false positives:")
    zero_fp = result_df[result_df["FP"] == 0]
    if zero_fp.empty:
        print("None in the tested range.")
    else:
        print(zero_fp.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    main()