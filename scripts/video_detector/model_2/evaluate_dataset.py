import csv
import time
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
FRAMES_PER_VIDEO = 8
OUTPUT = Path("results/model2_evaluation.csv")

def get_frames(path, count=FRAMES_PER_VIDEO):
    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError("Video contains no readable frames")
    indices = np.linspace(0, len(reader) - 1, min(count, len(reader)))
    frames = reader.get_batch(indices.astype(int)).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]

def main():
    start = time.time()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    videos = [
        (path, 1) for path in sorted(Path("data/ai").glob("*.mp4"))
    ] + [
        (path, 0) for path in sorted(Path("data/human").glob("*.mp4"))
    ]
    
    print(f"Discovered {len(videos)} videos for evaluation.")
    
    results = []
    y_true = []
    y_pred = []
    
    for idx, (path, true_label) in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Evaluating: {path} (True: {'AI' if true_label==1 else 'HUMAN'})")
        try:
            frames = get_frames(path)
            inputs = processor(images=frames, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
            
            if logits.shape[-1] == 1:
                score = torch.sigmoid(logits).mean().item()
            else:
                probs = torch.softmax(logits, dim=-1)
                labels = model.config.id2label
                fake_ids = [int(i) for i, label in labels.items() if "fake" in str(label).lower()]
                if fake_ids:
                    score = probs[:, fake_ids[0]].mean().item()
                else:
                    score = probs[:, -1].mean().item()
                    
            pred_label = 1 if score >= 0.5 else 0
            status = "SUCCESS"
        except Exception as e:
            print(f"  -> Error: {e}")
            score = 0.0
            pred_label = 0
            status = f"FAILED: {e}"
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        results.append({
            "file_path": str(path),
            "true_label": "AI" if true_label == 1 else "HUMAN",
            "predicted_label": "AI" if pred_label == 1 else "HUMAN",
            "fake_score": score,
            "status": status
        })
        
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "true_label", "predicted_label", "fake_score", "status"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nEvaluation results saved to {OUTPUT}")
    print(f"Total time: {time.time() - start:.1f}s")
    
    valid_indices = [i for i, s in enumerate(results) if s["status"] == "SUCCESS"]
    if valid_indices:
        vt = [y_true[i] for i in valid_indices]
        vp = [y_pred[i] for i in valid_indices]
        acc = accuracy_score(vt, vp)
        precision, recall, f1, _ = precision_recall_fscore_support(vt, vp, average="binary", zero_division=0)
        cm = confusion_matrix(vt, vp)
        print("\n--- Provisional Metrics ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()