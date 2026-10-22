import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

CSV_PATH = Path("results/model2_evaluation.csv")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Evaluation report not found at {CSV_PATH}. Please run model evaluation first.")

    with CSV_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    
    rows = [
        row for row in rows
        if row.get("status") == "SUCCESS"
        and row.get("fake_score", "").strip()
    ]
    
    if not rows:
        raise RuntimeError(f"No successful predictions found in {CSV_PATH}")

    # Dataset labels: AI = 1, HUMAN = 0
    y_true = np.array([
        1 if row["true_label"].upper() == "AI" else 0
        for row in rows
    ])
    scores = np.array([float(row["fake_score"]) for row in rows])

    print(f"Successful videos analyzed: {len(rows)}")
    print("Threshold | Accuracy | Precision | Recall | F1 | TN FP FN TP")
    print("-" * 78)
    
    for threshold in (0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        
        print(
            f"{threshold:9.3f} | "
            f"{accuracy_score(y_true, y_pred):8.3f} | "
            f"{precision_score(y_true, y_pred, zero_division=0):9.3f} | "
            f"{recall_score(y_true, y_pred, zero_division=0):6.3f} | "
            f"{f1_score(y_true, y_pred, zero_division=0):.3f} | "
            f"{tn:2d} {fp:2d} {fn:2d} {tp:2d}"
        )

if __name__ == "__main__":
    main()