import csv
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
FRAMES_PER_VIDEO = 8
OUTPUT_PATH = Path("results/model2_aggregation_comparison.csv")

def extract_frame_scores(video_path, processor, model):
    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) == 0:
        return []
    indices = np.linspace(0, len(reader) - 1, min(FRAMES_PER_VIDEO, len(reader))).astype(int)
    frames = reader.get_batch(indices).asnumpy()
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        
    if logits.shape[-1] == 1:
        scores = torch.sigmoid(logits).flatten().cpu().numpy()
    else:
        probs = torch.softmax(logits, dim=-1)
        labels = model.config.id2label
        fake_ids = [int(i) for i, label in labels.items() if "fake" in str(label).lower()]
        if fake_ids:
            scores = probs[:, fake_ids[0]].cpu().numpy()
        else:
            scores = probs[:, -1].cpu().numpy()
    return scores

def main():
    print("Loading processor and model for frame aggregation analysis...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    videos = [
        (path, 1) for path in sorted(Path("data/ai").glob("*.mp4"))
    ] + [
        (path, 0) for path in sorted(Path("data/human").glob("*.mp4"))
    ]
    
    records = []
    for path, true_label in videos:
        try:
            scores = extract_frame_scores(path, processor, model)
            if len(scores) == 0:
                continue
            records.append({
                "file_path": str(path),
                "true_label": "AI" if true_label == 1 else "HUMAN",
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "p90_score": float(np.percentile(scores, 90)),
            })
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    df_agg = pd.DataFrame(records) if 'pd' in globals() else None
    # Fallback standard CSV writing if pandas isn't explicitly imported for dataframes here
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "true_label", "mean_score", "median_score", "p90_score"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Aggregation scores saved to {OUTPUT_PATH}")
    
    # Quick sweep for comparison across aggregation strategies at threshold 0.025
    y_true = np.array([1 if r["true_label"] == "AI" else 0 for r in records])
    threshold = 0.025
    
    print("\n--- Aggregation Strategy Comparison (Threshold = 0.025) ---")
    for method in ["mean_score", "median_score", "p90_score"]:
        scores = np.array([r[method] for r in records])
        y_pred = (scores >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Method: {method:12s} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    main()