from __future__ import annotations

import re
from typing import Any


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


def normalize_image_label(native_label: str) -> str:
	"""Map a model-specific image label to the shared prediction vocabulary."""
	normalized_label = re.sub(r"[_-]+", " ", native_label.strip().lower())

	if normalized_label in AI_LABELS or any(
		label in normalized_label for label in ("ai", "artificial", "generated", "synthetic")
	):
		return "AI"
	if normalized_label in HUMAN_LABELS or any(
		label in normalized_label for label in ("human", "real", "authentic", "natural")
	):
		return "HUMAN"

	raise ValueError(f"Unsupported image classification label: {native_label!r}")


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
