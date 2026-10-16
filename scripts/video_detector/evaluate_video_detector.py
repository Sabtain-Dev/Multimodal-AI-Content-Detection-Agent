import csv
import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.video import VideoDetectorModel

AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")
REPORT_PATH = Path("results/video_evaluation_report.csv")


def main():
    print("=" * 60)
    print("VIDEO DETECTOR EVALUATION PIPELINE")
    print("=" * 60)

    if not AI_DIR.exists() or not HUMAN_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directories not found. Ensure '{AI_DIR}' and '{HUMAN_DIR}' exist."
        )

    # Collect dataset items
    dataset = []
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

    for ext in video_extensions:
        for p in AI_DIR.glob(ext):
            dataset.append((p, "AI"))
        for p in HUMAN_DIR.glob(ext):
            dataset.append((p, "HUMAN"))

    print(f"Total labeled videos discovered: {len(dataset)}")
    if len(dataset) == 0:
        print("No videos found in data/ai/ or data/human/.")
        return

    print("\nInitializing Video Detector Model...")
    detector = VideoDetectorModel()

    results = []
    y_true = []
    y_pred = []

    print("\nRunning inference on evaluation dataset...")
    for idx, (video_path, true_label) in enumerate(dataset, start=1):
        print(f"[{idx}/{len(dataset)}] Evaluating: {video_path} (True: {true_label})")
        try:
            pred_res = detector.predict(str(video_path))
            pred_label = pred_res["label"]
            model_score = pred_res["model_score"]
            status = "SUCCESS"
        except Exception as e:
            print(f"  -> ERROR during inference: {e}")
            pred_label = "ERROR"
            model_score = 0.0
            status = f"FAILED: {str(e)}"

        y_true.append(true_label)
        y_pred.append(pred_label if status == "SUCCESS" else "UNKNOWN")

        results.append({
            "file_path": str(video_path),
            "true_label": true_label,
            "predicted_label": pred_label,
            "model_score": model_score,
            "status": status,
        })

    # Ensure results directory exists
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV report
    with open(REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_path", "true_label", "predicted_label", "model_score", "status"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed prediction report saved to: {REPORT_PATH}")

    # Calculate metrics for successfully processed samples
    valid_indices = [i for i, s in enumerate(results) if s["status"] == "SUCCESS"]
    if not valid_indices:
        print("\nNo samples successfully processed.")
        return

    filtered_true = [y_true[i] for i in valid_indices]
    filtered_pred = [y_pred[i] for i in valid_indices]

    acc = accuracy_score(filtered_true, filtered_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_true, filtered_pred, average="binary", pos_label="AI", zero_division=0
    )
    cm = confusion_matrix(filtered_true, filtered_pred, labels=["HUMAN", "AI"])

    print("\n" + "=" * 30 + " EVALUATION METRICS " + "=" * 30)
    print(f"Total Evaluated Successfully : {len(valid_indices)}/{len(dataset)}")
    print(f"Accuracy                     : {acc:.4f}")
    print(f"Precision (AI)               : {precision:.4f}")
    print(f"Recall (AI)                  : {recall:.4f}")
    print(f"F1 Score (AI)                : {f1:.4f}")
    print("\nConfusion Matrix (Rows: True [HUMAN, AI], Cols: Pred [HUMAN, AI]):")
    print(cm)
    print("=" * 79)


if __name__ == "__main__":
    main()