import os
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.detectors.kaggle_dataset import resolve_image_dataset_dir

MODEL_NAME = "haywoodsloan/ai-image-detector-deploy"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = resolve_image_dataset_dir(PROJECT_ROOT)
OUTPUT_CSV = PROJECT_ROOT / "results" / "image_detector_2_results.csv"

LABEL_MAPPING = {
    "ai": "AI",
    "hum": "HUMAN",
    "human": "HUMAN",
    "artificial": "AI",
    "real": "HUMAN"
}


def load_dataset_file_paths():
    records = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    for label in ["ai", "human"]:
        folder = DATASET_DIR / label
        if not folder.exists():
            continue
        for file_path in folder.glob("*"):
            if file_path.suffix.lower() in valid_extensions:
                relative_image_path = f"{label}/{file_path.name}"
                records.append({
                    "file_path": str(file_path),
                    "image_path": relative_image_path,
                    "file_name": file_path.name,
                    "actual_label": "AI" if label == "ai" else "HUMAN"
                })
    return records


def get_ai_score_and_prediction(model, outputs):
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    
    ai_score = 0.0
    for idx, native_label in model.config.id2label.items():
        mapped_label = LABEL_MAPPING.get(native_label.lower(), native_label)
        if mapped_label == "AI":
            ai_score = float(probs[int(idx)].item())
            break

    prediction = "AI" if ai_score >= 0.5 else "HUMAN"
    human_score = 1.0 - ai_score
    
    top_class_idx = int(torch.argmax(probs).item())
    native_label = model.config.id2label[top_class_idx]
    
    return native_label, prediction, ai_score, human_score


def calculate_metrics(df):
    tp = len(df[(df["actual_label"] == "AI") & (df["prediction"] == "AI")])
    tn = len(df[(df["actual_label"] == "HUMAN") & (df["prediction"] == "HUMAN")])
    fp = len(df[(df["actual_label"] == "HUMAN") & (df["prediction"] == "AI")])
    fn = len(df[(df["actual_label"] == "AI") & (df["prediction"] == "HUMAN")])

    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score
    }


def evaluate_model_2():
    print("=" * 50)
    print(f"EVALUATING MODEL 2: {MODEL_NAME}")
    print("=" * 50)

    records = load_dataset_file_paths()
    if not records:
        print(f"Error: No images found in {DATASET_DIR}.")
        sys.exit(1)

    print(f"Loaded {len(records)} image samples for evaluation.")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    results = []
    for idx, item in enumerate(records, 1):
        img_path = item["file_path"]
        actual_label = item["actual_label"]

        start_time = time.time()
        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            file_size_kb = os.path.getsize(img_path) / 1024.0

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            native_label, pred, ai_score, human_score = get_ai_score_and_prediction(model, outputs)
            inference_time = time.time() - start_time

            results.append({
                "sample_id": idx,
                "image_path": item["image_path"],
                "file_name": item["file_name"],
                "actual_label": actual_label,
                "width": width,
                "height": height,
                "file_size_kb": round(file_size_kb, 2),
                "native_label": native_label,
                "prediction": pred,
                "ai_score": ai_score,
                "human_score": human_score,
                "correct": (pred == actual_label),
                "inference_time_sec": round(inference_time, 4)
            })
            print(f"[{idx}/{len(records)}] {item['image_path']} -> Pred: {pred} | Actual: {actual_label} | AI Score: {ai_score:.4f} ({inference_time:.2f}s)")
        except Exception as e:
            print(f"[{idx}/{len(records)}] Error processing {img_path}: {e}")

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved evaluation results to {OUTPUT_CSV}")

    m = calculate_metrics(df)

    print("\n--- MODEL 2 EVALUATION METRICS SUMMARY ---")
    print(f"Total Images Evaluated : {m['total']}")
    print(f"True Positives (AI)    : {m['tp']}")
    print(f"True Negatives (HUMAN) : {m['tn']}")
    print(f"False Positives        : {m['fp']}")
    print(f"False Negatives        : {m['fn']}")
    print("-" * 42)
    print(f"Accuracy               : {m['accuracy'] * 100:.2f}%")
    print(f"Precision (AI)         : {m['precision'] * 100:.2f}%")
    print(f"Recall (AI Sensitivity): {m['recall'] * 100:.2f}%")
    print(f"Specificity (HUMAN)    : {m['specificity'] * 100:.2f}%")
    print(f"F1-Score               : {m['f1_score'] * 100:.2f}%")
    print(f"Average Inference Time : {df['inference_time_sec'].mean():.4f} seconds/image")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    evaluate_model_2()