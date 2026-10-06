# Project Theory

## Project Title

Multimodal AI Content Detection Agent for Text, Image, Audio, and Video

## Problem Statement

The rapid development of generative AI has made it increasingly
difficult to distinguish between human-created and AI-generated
content across text, images, audio, and video.

## Proposed Solution

The current implementation uses pretrained detectors for text, image, and
audio, with a file-based orchestration layer that routes supported inputs to
the corresponding detection logic. The image route currently combines
caller-supplied model scores; image-model inference is not yet integrated into
the multimodal agent. Video detection and the full web/API reporting flow are
future work.

## Modalities

- **Text:** Three-model detector ensemble with majority voting
- **Image:** Two-model decision ensemble; score inference is supplied externally
- **Audio:** Two-model detector with a primary-decision policy
- **Video:** Planned; not currently supported

## Model Strategy

The initial version will use existing pretrained models.
Fine-tuning and training models from scratch are outside the
scope of V1.

## Intended Result

The system will provide a probabilistic assessment of whether
submitted content is likely AI-generated, together with model
information, confidence/evidence, and limitations.