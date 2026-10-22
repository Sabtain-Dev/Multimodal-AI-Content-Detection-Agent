import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

def main():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    
    for path in ("data/ai/1.mp4", "data/human/1.mp4"):
        reader = VideoReader(path, ctx=cpu(0))
        indices = np.linspace(0, len(reader) - 1, min(8, len(reader))).astype(int)
        frames = reader.get_batch(indices).asnumpy()
        images = [Image.fromarray(frame).convert("RGB") for frame in frames]
        inputs = processor(images=images, return_tensors="pt")
        
        with torch.inference_mode():
            logits = model(**inputs).logits
            
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits).flatten().tolist()
            score_type = "sigmoid output (provisionally fake score)"
        else:
            labels = model.config.id2label
            fake_ids = [
                int(i) for i, label in labels.items()
                if "fake" in str(label).lower()
            ]
            if not fake_ids:
                print(f"\n{path}: Cannot identify fake class: {labels}")
                continue
            scores = torch.softmax(logits, dim=-1)[:, fake_ids[0]].tolist()
            score_type = "fake-class probability"
            
        print(f"\nVideo: {path}")
        print("Score type:", score_type)
        for frame_no, (idx, score) in enumerate(zip(indices, scores), 1):
            print(f"  Sample {frame_no} (frame {idx}): {score:.4f}")
        print(f"Mean: {np.mean(scores):.4f}")
        print(f"Standard deviation: {np.std(scores):.4f}")
        print(f"Range: {min(scores):.4f} - {max(scores):.4f}")

if __name__ == "__main__":
    main()