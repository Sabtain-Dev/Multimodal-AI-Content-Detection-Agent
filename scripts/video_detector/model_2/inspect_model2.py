from pathlib import Path
import torch
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

def main():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    print("Model:", MODEL_ID)
    print("Model class:", type(model).__name__)
    print("Number of labels:", model.config.num_labels)
    print("id2label:", model.config.id2label)
    print("label2id:", model.config.label2id)
    
    for path in (
        Path("data/ai/1.mp4"),
        Path("data/human/1.mp4"),
    ):
        if not path.exists():
            print(f"\nSkipping missing video: {path}")
            continue
        reader = VideoReader(str(path), ctx=cpu(0))
        if len(reader) == 0:
            print(f"\nVideo contains no frames: {path}")
            continue
        indices = [0, len(reader) // 2, len(reader) - 1]
        frames = reader.get_batch(indices).asnumpy()
        images = [Image.fromarray(frame).convert("RGB") for frame in frames]
        inputs = processor(images=images, return_tensors="pt")
        
        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
        
        print(f"\nVideo: {path}")
        print("Logits shape:", logits.shape)
        print("Logits:", logits)
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits)
            print("Sigmoid probabilities (per frame):", probs.flatten().tolist())
            print("Mean probability:", probs.mean().item())
        else:
            probs = torch.softmax(logits, dim=-1)
            print("Softmax probabilities (per frame):", probs.tolist())
            print("Mean probabilities:", probs.mean(dim=0).tolist())

if __name__ == "__main__":
    main()