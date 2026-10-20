from pathlib import Path
import time
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
VIDEO_PATH = Path("data/ai/1.mp4")
NUM_FRAMES = 8

def extract_frames(video_path: Path, num_frames: int = NUM_FRAMES):
    """Extract uniformly spaced RGB frames using the existing Decord dependency."""
    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"No frames found in {video_path}")
    indices = np.linspace(
        0, len(reader) - 1, min(num_frames, len(reader))
    ).astype(int)
    frames = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]

def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    start = time.time()
    
    print(f"Loading model: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    print(f"Model loaded in {time.time() - start:.1f} seconds")

    print(f"Extracting frames from: {VIDEO_PATH}")
    frames = extract_frames(VIDEO_PATH)
    print(f"Extracted {len(frames)} frames.")

    # Process frames and run inference
    inputs = processor(images=frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Handle single-logit sigmoid output or multi-class softmax outputs gracefully
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits)
            mean_prob = probs.mean().item()
            print(f"\nSingle-logit output detected.")
            print(f"Fake Probability (Mean across frames): {mean_prob:.4f}")
            print(f"Verdict: {'AI / Fake' if mean_prob > 0.5 else 'Human / Real'}")
        else:
            probs = torch.softmax(logits, dim=-1)
            mean_probs = probs.mean(dim=0)
            id2label = getattr(model.config, "id2label", {0: "Real/Human", 1: "AI/Fake"})
            print("\nPredictions (Averaged across frames):")
            for idx, prob in enumerate(mean_probs):
                label = id2label.get(idx, str(idx))
                print(f"  {label}: {prob.item():.4f}")
            pred_idx = torch.argmax(mean_probs).item()
            print(f"\nFinal Predicted Label: {id2label.get(pred_idx, pred_idx)} (Confidence: {mean_probs[pred_idx].item():.4f})")

    print(f"Total execution time: {time.time() - start:.1f} seconds")

if __name__ == "__main__":
    main()