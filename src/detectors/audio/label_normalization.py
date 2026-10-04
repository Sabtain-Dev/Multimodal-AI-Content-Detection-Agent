from typing import Any, Dict

MODEL_1 = "Sayantan090/audio-fake-detector"
MODEL_2 = "Mahmoud59/wav2vec2-fake-audio-detector"

MODEL_LABEL_MAPS = {
    MODEL_1: {
        "FAKE": "AI",
        "REAL": "HUMAN",
    },
    MODEL_2: {
        "LABEL_0": "AI",
        "LABEL_1": "HUMAN",
    },
}


def normalize_audio_label(model_name: str, raw_label: str) -> str:
    """
    Normalize a model's native output label to the canonical AI/HUMAN labels.
    """
    try:
        label_map = MODEL_LABEL_MAPS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported audio model: '{model_name}'") from exc

    clean_label = str(raw_label).strip().upper()
    try:
        return label_map[clean_label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label '{raw_label}' for model '{model_name}'"
        ) from exc


def build_normalized_prediction(
    model_name: str,
    native_label: str,
    score: float,
) -> Dict[str, Any]:
    """Build the normalized representation of one model prediction."""
    return {
        "model": model_name,
        "native_label": native_label,
        "normalized_label": normalize_audio_label(model_name, native_label),
        "score": float(score),
    }


def compute_confidence_indicator(score: float) -> str:
    """
    Maps raw probability score (0.0 to 1.0) to a qualitative confidence indicator.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0.0 and 1.0")

    if score >= 0.90 or score <= 0.10:
        return "High"
    elif score >= 0.75 or score <= 0.25:
        return "Moderate"
    else:
        return "Low"