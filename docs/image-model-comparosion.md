# Image AI Detection Model Comparison

## Overview

This document evaluates the current image-detection candidates for the Multimodal AI Content Detection Agent. The goal is to select a lightweight, reliable image-classification strategy that can operate in a V1 baseline while keeping the output standardized across the project.

The image branch follows the same design principle as the text branch: use pretrained models, normalize the output, and combine evidence into a binary AI vs. HUMAN label.

---

## Project Constraints

- Use pretrained models only in V1.
- Keep the output format consistent with the rest of the project.
- Favor low-overhead decisions suitable for local testing and cloud demos.
- Minimize false positives on real human-generated images.
- Return both a user-facing label and model telemetry for debugging.

---

## Candidate 1: Ateeqq AI vs Human Image Detector

* **Repository:** `Ateeqq/ai-vs-human-image-detector`
* **Task:** Binary image authenticity classification
* **Role in the system:** Supporting model
* **Default threshold:** `0.70`

### Strengths
- Strong baseline for general AI-vs-human image discrimination.
- Produces a conventional probability-like score that fits the project schema.
- Good as a secondary signal in an ensemble for added coverage.

### Weaknesses
- Model-card and benchmark coverage are limited in this project context.
- More likely to disagree with a deployment-oriented detector on ambiguous or stylized examples.
- Should be treated as a validator rather than the sole decision-maker.

---

## Candidate 2: Haywoodsloan AI Image Detector Deploy

* **Repository:** `haywoodsloan/ai-image-detector-deploy`
* **Task:** Binary AI-vs-human image classification
* **Role in the system:** Primary decision model
* **Default threshold:** `0.50`

### Strengths
- Better suited as the primary decision signal in the current pipeline.
- Directly aligns with the project’s deployment-oriented detection workflow.
- Produces a clean binary decision that is easy to combine with the supporting model.

### Weaknesses
- Its output is sensitive to threshold choice and image preprocessing assumptions.
- Can be overconfident when the input is visually ambiguous or low-quality.
- Must be paired with the secondary model to reduce single-model risk.

---

## Direct Comparison Matrix

| Criterion | Ateeqq Model | Haywoodsloan Model |
| --- | --- | --- |
| Role | Supporting validator | Primary decision model |
| Typical threshold | 0.70 | 0.50 |
| Output style | Binary probability-style score | Binary decision-oriented score |
| Ensemble fit | Good supporting signal | Strong primary signal |
| Risk profile | Lower confidence on edge cases | Higher sensitivity to ambiguous inputs |
| Best use | Cross-checking the final label | Driving final decision |

---

## Final Architecture Decision

The V1 image pipeline uses a two-model ensemble instead of a larger multi-model system.

### Decision policy

1. Convert both model scores into binary labels:
   - `AI` if score >= threshold
   - `HUMAN` otherwise
2. If the models agree, keep the agreed label.
3. If the models disagree, prefer the second model’s label.
4. Cap confidence to `Moderate` when there is disagreement.
5. Return normalized telemetry alongside the final answer.

This keeps the design simple while preserving explainability and enough disagreement information for debugging.

---

## Why this is the current V1 choice

- It is lightweight and easy to reason about.
- It minimizes implementation risk compared with a complex multi-model ensemble.
- It matches the repository’s modular design: each modality has a focused detector and a normalization layer.
- It allows the agent to expose a clean object with both `label` and `models_agree` metadata.

---

## Output Contract

The image detector returns a dictionary similar to the following:

```python
{
    "modality": "image",
    "label": "HUMAN",
    "model_score": 0.8,
    "confidence_indicator": "Moderate",
    "model_1": {
        "name": "Ateeqq/ai-vs-human-image-detector",
        "label": "AI",
        "score": 0.8
    },
    "model_2": {
        "name": "haywoodsloan/ai-image-detector-deploy",
        "label": "HUMAN",
        "score": 0.2
    },
    "models_agree": False,
    "decision_mode": "model_2_primary_fallback"
}
```

This output is designed to be easy to consume in downstream orchestration and easy to inspect in evaluation logs.

---

## Selection Recommendation

Use:

- `Ateeqq/ai-vs-human-image-detector` as the supporting validator
- `haywoodsloan/ai-image-detector-deploy` as the primary model

This provides a balanced V1 image detector with a clear decision policy and stable output contract while keeping the architecture simple enough to expand later.
