# Model 4 — AEGIS Video Detector

## Overview

AEGIS is a multimodal video detector designed to distinguish fully AI-generated videos from real camera footage.

It is not primarily a face-swap or face-manipulation detector.

## Architecture

The model contains three branches:

1. Pixel Branch — DINOv2, LNP, and FFT features.
2. Motion Branch — LightFlowNet and a Temporal Transformer.
3. Consistency Branch — DINOv2 embeddings and temporal consistency features.

The three branches are combined using TripleFusion.

The model produces:
- Main AI probability.
- Three individual branch scores.
- Three pairwise branch scores.

## Upstream

Repository: https://github.com/MusapYildiz/ai_video_detection_benchmark

Hugging Face: https://huggingface.co/MusapYildiz/aegis-video-detector

License: MIT.

See `vendor/aegis/UPSTREAM_SOURCE.md` for source attribution.

## Checkpoint

The recommended checkpoint is:

`checkpoint_best.pt`

It belongs to Phase 2, Epoch 7.

The checkpoint will be downloaded and configured in a later implementation task.

## Implementation Policy

- Preserve Models 1, 2, and 3.
- Keep the original AEGIS implementation isolated under `vendor/aegis/`.
- Implement project-specific compatibility changes separately.
- Use deterministic video preprocessing during evaluation.
- Do not tune thresholds on the final evaluation set.

## Current Status

Source setup is in progress.

Dependency verification, DINOv2 loading, checkpoint loading, inference integration, and evaluation will be implemented in subsequent tasks.