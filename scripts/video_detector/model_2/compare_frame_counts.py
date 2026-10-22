import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

def score_video(path, count, processor, model):
    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        return 0.0
    indices = np.linspace(0, len(reader) - 1, min(count, len(reader))).astype(int)
    frames = reader.get_batch(indices).asnumpy()
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits).mean().item()
    labels = model.config.id2label
    fake_ids = [
        int(i) for i, label in labels.items()
        if "fake" in str(label).lower()
    ]
    if not fake_ids:
        raise ValueError(f"Cannot identify fake class from labels: {labels}")
    return torch.softmax(logits, dim=-1).mean(dim=0)[fake_ids[0]].item()

def main():
    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    test_videos = ["data/ai/1.mp4", "data/human/1.mp4"]
    for v_path in test_videos:
        print(f"\nTesting video: {v_path}")
        score_4 = score_video(v_path, 4, processor, model)
        score_8 = score_video(v_path, 8, processor, model)
        print(f"  4 frames score: {score_4:.4f}")
        print(f"  8 frames score: {score_8:.4f}")
        print(f"  Absolute difference: {abs(score_4 - score_8):.4f}")

if __name__ == "__main__":
    main()