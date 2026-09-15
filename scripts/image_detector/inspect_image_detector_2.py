import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.detectors.image.label_normalization import build_normalized_prediction

MODEL_NAME = "haywoodsloan/ai-image-detector-deploy"


def inspect_model_2():
    print(f"==================================================")
    print(f"INSPECTING MODEL 2: {MODEL_NAME}")
    print(f"==================================================")

    start_time = time.time()

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Model Architecture  : {model.config.architectures}")
    print(f"id2label Config     : {model.config.id2label}")
    print(f"label2id Config     : {model.config.label2id}")
    print(f"Processor Details   : {processor}")

    # Identify AI label index dynamically
    ai_label_idx = None
    for idx, label in model.config.id2label.items():
        if label.lower() in ["ai", "artificial", "generated", "synthetic"]:
            ai_label_idx = int(idx)
            break

    if ai_label_idx is None:
        raise ValueError("Could not dynamically resolve AI label index from id2label mapping.")

    # Generate test image
    dummy_img = Image.new("RGB", (224, 224), color=(200, 100, 80))

    inputs = processor(images=dummy_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    ai_score = float(probs[ai_label_idx].item())
    top_class_idx = int(torch.argmax(probs).item())
    native_label = model.config.id2label[top_class_idx]

    normalized = build_normalized_prediction(
        model_name=MODEL_NAME,
        native_label=native_label,
        ai_score=ai_score,
        threshold=0.5,
    )

    elapsed = time.time() - start_time

    print("\n--- INFERENCE RESULTS ---")
    print(f"Native Model Label  : {native_label}")
    print(f"Normalized Label    : {normalized['prediction']}")
    print(f"AI Score            : {normalized['ai_score']:.6f}")
    print(f"Human Score         : {normalized['human_score']:.6f}")
    print(f"Inference Time      : {elapsed:.4f}s")
    print(f"Full Normalized Output:\n{normalized}")
    print(f"==================================================\n")


if __name__ == "__main__":
    inspect_model_2()