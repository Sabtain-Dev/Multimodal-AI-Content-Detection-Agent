from pathlib import Path
import csv
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
THRESHOLD = 0.025
FRAMES_PER_VIDEO = 8

ERROR_VIDEOS = [
    ("data/ai/2.mp4", "AI"),
    ("data/ai/7.mp4", "AI"),
    ("data/ai/13.mp4", "AI"),
    ("data/ai/17.mp4", "AI"),
    ("data/human/9.mp4", "HUMAN"),
]

OUTPUT_DIR = Path("results/model2_error_frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

def inspect_error_video(video_path_str, true_label):
    path = Path(video_path_str)
    if not path.exists():
        print(f"[SKIP] File not found: {path}")
        return
    
    print(f"\nInspecting error video: {path} (True: {true_label})")
    reader = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(reader)
    if total_frames == 0:
        print("  -> Video contains no readable frames.")
        return

    indices = np.linspace(0, total_frames - 1, min(FRAMES_PER_VIDEO, total_frames)).astype(int)
    frames_np = reader.get_batch(indices).asnumpy()
    images = [Image.fromarray(f).convert("RGB") for f in frames_np]

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        logits = model(**inputs).logits
        if logits.shape[-1] == 1:
            scores = torch.sigmoid(logits).flatten().cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1)
            labels = model.config.id2label
            fake_ids = [int(i) for i, label in labels.items() if "fake" in str(label).lower()]
            scores = probs[:, fake_ids[0]].cpu().numpy() if fake_ids else probs[:, -1].cpu().numpy()

    mean_score = float(np.mean(scores))
    pred_label = "AI" if mean_score >= THRESHOLD else "HUMAN"
    print(f"  Mean Fake Score: {mean_score:.4f} | Predicted: {pred_label} | True: {true_label}")

    # Build contact sheet
    columns = 4
    thumb_w, thumb_h, caption_h = 320, 180, 30
    sheet_w = columns * thumb_w
    rows = (len(images) + columns - 1) // columns
    sheet_h = rows * (thumb_h + caption_h)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, (frame_idx, img, score) in enumerate(zip(indices, images, scores)):
        thumb = img.copy()
        thumb.thumbnail((thumb_w, thumb_h))
        x = (idx % columns) * thumb_w
        y = (idx // columns) * (thumb_h + caption_h)
        
        img_x = x + (thumb_w - thumb.width) // 2
        img_y = y + (thumb_h - thumb.height) // 2
        sheet.paste(thumb, (img_x, img_y))
        
        draw.text(
            (x + 5, y + thumb_h + 5),
            f"F:{frame_idx} | S:{score:.3f}",
            fill="black"
        )

    out_file = OUTPUT_DIR / f"{path.stem}_{true_label.lower()}_contact_sheet.jpg"
    sheet.save(out_file, quality=90)
    print(f"  Saved contact sheet: {out_file}")

def main():
    for v_path, label in ERROR_VIDEOS:
        inspect_error_video(v_path, label)

if __name__ == "__main__":
    main()