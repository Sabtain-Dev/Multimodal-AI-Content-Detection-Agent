import csv
import time
from pathlib import Path
import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModelForVideoClassification

MODEL_ID = "mannicui0/Rethinking-the-readout"
AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")
OUTPUT_FILE = Path("results/model3_evaluation.csv")
NUM_FRAMES = 8
DEVICE = torch.device("cpu")

def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract evenly spaced RGB frames using Decord."""
    reader = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        raise ValueError("Video contains no readable frames.")
    indices = np.linspace(
        0,
        total_frames - 1,
        min(num_frames, total_frames),
        dtype=int,
    )
    frames = reader.get_batch(indices).asnumpy()
    # Return list of numpy arrays or PIL images depending on processor expectation
    return [frames[i] for i in range(len(frames))]

def main():
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    # Determine AI label index from id2label
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError("Model config does not define id2label mapping.")
    
    ai_idx = None
    for idx, label_name in id2label.items():
        if "ai" in str(label_name).lower() or "fake" in str(label_name).lower():
            ai_idx = int(idx)
            break
    if ai_idx is None:
        raise ValueError(f"Could not unambiguously identify AI/fake class in id2label: {id2label}")

    videos = [(p, "AI") for p in sorted(AI_DIR.glob("*.mp4"))] + \
             [(p, "HUMAN") for p in sorted(HUMAN_DIR.glob("*.mp4"))]

    print(f"Discovered {len(videos)} videos for evaluation.")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()

    for idx, (video_path, true_label) in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] Evaluating: {video_path}")
        try:
            frames = extract_frames(video_path)
            # Standard video classification input formatting
            inputs = processor(frames, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]

            ai_score = float(probs[ai_idx].item())
            pred_idx = int(torch.argmax(probs).item())
            pred_label = "AI" if pred_idx == ai_idx else "HUMAN"
            status = "success"
        except Exception as e:
            print(f"  -> Error: {e}")
            ai_score = 0.0
            pred_label = "ERROR"
            status = f"error: {e}"

        results.append({
            "file_path": str(video_path),
            "true_label": true_label,
            "predicted_label": pred_label,
            "ai_score": ai_score,
            "status": status
        })

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "true_label", "predicted_label", "ai_score", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation completed in {time.time() - start_time:.1f}s. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()