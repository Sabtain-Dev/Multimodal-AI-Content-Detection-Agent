from __future__ import annotations

import re
from typing import Any
from typing import Union

AI_LABELS = {
	"ai",
	"artificial",
	"artificial intelligence",
	"generated",
	"synthetic",
	"fake",
}
HUMAN_LABELS = {
	"human",
	"real",
	"authentic",
	"natural",
}


"""
Label normalization for image detection models.
Standardizes heterogeneous raw model outputs into canonical binary values ("AI" | "HUMAN").
"""


def normalize_image_label(raw_label: str) -> str:
    """
    Normalizes raw model label strings into standard binary format.

    Supported variants:
    - AI: 'ai', 'artificial', 'fake', 'generated', 'label_1', '1'
    - HUMAN: 'human', 'real', 'authentic', 'label_0', '0'
    """
    if not raw_label:
        raise ValueError("Cannot normalize an empty or None label string.")

    cleaned = str(raw_label).strip().lower()

    ai_indicators = {"ai", "artificial", "fake", "generated", "synthetic", "label_1", "1"}
    human_indicators = {"human", "real", "authentic", "label_0", "0"}

    if cleaned in ai_indicators:
        return "AI"
    elif cleaned in human_indicators:
        return "HUMAN"
    else:
        raise ValueError(f"Unrecognized image detection label string: '{raw_label}'")


def build_normalized_prediction(
	model_name: str,
	native_label: str,
	ai_score: float,
	threshold: float = 0.5,
) -> dict[str, Any]:
	"""Build a consistent prediction result from a model's native output."""
	if not 0.0 <= ai_score <= 1.0:
		raise ValueError("ai_score must be between 0.0 and 1.0")
	if not 0.0 <= threshold <= 1.0:
		raise ValueError("threshold must be between 0.0 and 1.0")

	normalize_image_label(native_label)
	prediction = "AI" if ai_score >= threshold else "HUMAN"

	return {
		"model_name": model_name,
		"native_label": native_label,
		"prediction": prediction,
		"ai_score": round(ai_score, 6),
		"human_score": round(1.0 - ai_score, 6),
		"threshold_used": threshold,
	}
