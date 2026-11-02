import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from PIL import Image
from decord import VideoReader, cpu

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from scripts.video_detector.model_3.backbones import BACKBONE_REGISTRY
from scripts.video_detector.model_3.pvp_head import PVPHead


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "videomae_k400"

BACKBONE_PATH = Path("models/model3_videomae_k400")
HEAD_PATH = Path("models/model3_vpvp.pt")

AI_DIR = Path("data/ai")
HUMAN_DIR = Path("data/human")

OUTPUT_CSV = Path("results/model3_evaluation.csv")
OUTPUT_JSON = Path("results/model3_metrics.json")

NUM_FRAMES = 16
IMAGE_SIZE = 224
THRESHOLD = 0.5

DEVICE = torch.device("cpu")

# Dataset normalization used by the authors' evaluation code.
DATASET_MEAN = (0.45, 0.45, 0.45)
DATASET_STD = (0.225, 0.225, 0.225)


# =========================================================
# Video preprocessing
# =========================================================

def load_video_frames(video_path):
    """
    Load 16 uniformly sampled frames.

    Returns:
        Tensor with shape [1, 3, 16, 224, 224],
        normalized using the authors' dataset statistics.
    """

    reader = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(reader)

    if total_frames == 0:
        raise ValueError("Video contains no readable frames.")

    # Always sample exactly NUM_FRAMES.
    # Repeated indices are allowed for short videos.
    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES,
        dtype=int,
    )

    frames = reader.get_batch(indices).asnumpy()

    processed_frames = []

    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")

        # Center crop to square, matching the authors' preprocessing.
        width, height = image.size
        side = min(width, height)

        left = (width - side) // 2
        top = (height - side) // 2

        image = image.crop((
            left,
            top,
            left + side,
            top + side,
        ))

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BICUBIC,
        )

        # Convert to [C, H, W], range [0, 1].
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)

        mean = torch.tensor(DATASET_MEAN).view(3, 1, 1)
        std = torch.tensor(DATASET_STD).view(3, 1, 1)

        tensor = (tensor - mean) / std
        processed_frames.append(tensor)

    # [T, C, H, W] -> [1, C, T, H, W]
    clip = torch.stack(processed_frames, dim=0)
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)

    return clip


def normalize_for_backbone(clip, backbone):
    """
    Undo dataset normalization and apply VideoMAE's ImageNet
    normalization, matching the official evaluation procedure.
    """

    ds_mean = torch.tensor(
        DATASET_MEAN, device=clip.device
    ).view(1, 3, 1, 1, 1)

    ds_std = torch.tensor(
        DATASET_STD, device=clip.device
    ).view(1, 3, 1, 1, 1)

    bb_mean = torch.tensor(
        backbone.input_mean, device=clip.device
    ).view(1, 3, 1, 1, 1)

    bb_std = torch.tensor(
        backbone.input_std, device=clip.device
    ).view(1, 3, 1, 1, 1)

    # Undo dataset normalization.
    clip = clip * ds_std + ds_mean

    # Apply backbone normalization.
    clip = (clip - bb_mean) / bb_std

    return clip


# =========================================================
# Model loading
# =========================================================

def load_model():
    print("\nLoading VideoMAE backbone...")

    if not (BACKBONE_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"Backbone configuration not found: {BACKBONE_PATH}"
        )

    backbone_cls = BACKBONE_REGISTRY[MODEL_NAME]

    backbone = backbone_cls(
        pretrained_model=str(BACKBONE_PATH),
        num_frames=NUM_FRAMES,
        img_size=IMAGE_SIZE,
    )

    backbone = backbone.to(DEVICE)
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    print("VideoMAE backbone loaded.")

    print("\nLoading V-PVP head...")

    if not HEAD_PATH.exists():
        raise FileNotFoundError(
            f"V-PVP checkpoint not found: {HEAD_PATH}\n"
            "Run prepare_model3.py first."
        )

    head = PVPHead(
        "velocity_gated",
        D=backbone.embed_dim,
    )

    checkpoint = torch.load(
        HEAD_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # The released checkpoint is a state_dict.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    head.load_state_dict(checkpoint)
    head = head.to(DEVICE)
    head.eval()

    print("V-PVP head loaded.")

    return backbone, head


# =========================================================
# Single-video prediction
# =========================================================

@torch.inference_mode()
def predict_video(video_path, backbone, head):
    clip = load_video_frames(video_path)
    clip = clip.to(DEVICE)

    clip = normalize_for_backbone(clip, backbone)

    # Extract patch-level temporal features.
    # Expected shape: [B, P, T', D]
    features = backbone.extract_patch_features(clip)

    if features.ndim != 4:
        raise RuntimeError(
            f"Unexpected feature shape: {tuple(features.shape)}"
        )

    # Calculate temporal velocity.
    # Expected shape: [B, P, T'-1, D]
    velocity = (
        features[:, :, 1:, :]
        - features[:, :, :-1, :]
    )

    if velocity.shape[2] < 1:
        raise RuntimeError("Insufficient temporal features.")

    # V-PVP requires both features and velocity.
    logits = head(features, velocity)

    ai_score = torch.sigmoid(logits).item()

    predicted_label = (
        "AI" if ai_score >= THRESHOLD else "HUMAN"
    )

    return ai_score, predicted_label


# =========================================================
# Evaluation
# =========================================================

def main():
    start_time = time.time()

    print("=" * 60)
    print("MODEL 3 EVALUATION")
    print("VideoMAE-K400 + V-PVP")
    print("=" * 60)

    backbone, head = load_model()

    videos = (
        [(p, "AI") for p in sorted(AI_DIR.glob("*.mp4"))]
        +
        [(p, "HUMAN") for p in sorted(HUMAN_DIR.glob("*.mp4"))]
    )

    print(f"\nAI videos: {len(list(AI_DIR.glob('*.mp4')))}")
    print(f"Human videos: {len(list(HUMAN_DIR.glob('*.mp4')))}")
    print(f"Total videos: {len(videos)}")

    if not videos:
        raise RuntimeError("No MP4 videos found in the dataset folders.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for index, (video_path, true_label) in enumerate(videos, 1):

        print(
            f"\n[{index}/{len(videos)}] "
            f"Evaluating {video_path}"
        )

        try:
            ai_score, predicted_label = predict_video(
                video_path,
                backbone,
                head,
            )

            status = "success"
            error_message = ""

            print(
                f"  True: {true_label} | "
                f"Predicted: {predicted_label} | "
                f"AI score: {ai_score:.6f}"
            )

        except Exception as error:
            ai_score = None
            predicted_label = "ERROR"
            status = "error"
            error_message = str(error)

            print(f"  ERROR: {error}")

        results.append({
            "file_path": str(video_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "ai_score": ai_score,
            "status": status,
            "error": error_message,
        })

    # -----------------------------------------------------
    # Save individual predictions
    # -----------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "file_path",
            "true_label",
            "predicted_label",
            "ai_score",
            "status",
            "error",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Calculate metrics using successful predictions only
    # -----------------------------------------------------

    successful = [
        result for result in results
        if result["status"] == "success"
    ]

    y_true = [r["true_label"] for r in successful]
    y_pred = [r["predicted_label"] for r in successful]

    y_true_binary = [
        1 if r["true_label"] == "AI" else 0
        for r in successful
    ]

    y_scores = [r["ai_score"] for r in successful]

    if successful:
        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            pos_label="AI",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            pos_label="AI",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            pos_label="AI",
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=["HUMAN", "AI"],
        ).tolist()

        if len(set(y_true_binary)) == 2:
            auc = roc_auc_score(y_true_binary, y_scores)
        else:
            auc = None

    else:
        accuracy = precision = recall = f1 = None
        auc = None
        cm = [[0, 0], [0, 0]]

    elapsed_time = time.time() - start_time

    metrics = {
        "model": "VideoMAE-K400 + V-PVP",
        "backbone": str(BACKBONE_PATH),
        "checkpoint": str(HEAD_PATH),
        "threshold": THRESHOLD,
        "num_frames": NUM_FRAMES,
        "device": str(DEVICE),
        "total_videos": len(videos),
        "successful_evaluations": len(successful),
        "failed_evaluations": len(videos) - len(successful),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix_human_ai": cm,
        "elapsed_time_seconds": round(elapsed_time, 2),
    }

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # -----------------------------------------------------
    # Display summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL 3 EVALUATION SUMMARY")
    print("=" * 60)

    print(json.dumps(metrics, indent=4))

    print("\nConfusion matrix:")
    print("Rows: Actual [HUMAN, AI]")
    print("Columns: Predicted [HUMAN, AI]")
    print(np.array(cm))

    print("\nSaved files:")
    print(f"CSV:  {OUTPUT_CSV}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()