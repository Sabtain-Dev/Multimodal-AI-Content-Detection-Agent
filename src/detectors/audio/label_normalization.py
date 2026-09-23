"""
Label normalization layer for audio detection models.
Maps raw output labels to unified canonical values: AI or HUMAN.
"""
from typing import Dict, Any

MODEL_1 = "Sayantan090/audio-fake-detector"
MODEL_2 = "Mahmoud59/wav2vec2-fake-audio-detector"

MODEL_1_LABEL_MAP = {
    "FAKE": "AI",
    "REAL": "HUMAN",
}

MODEL_2_LABEL_MAP = {
    "LABEL_0": "AI",
    "LABEL_1": "HUMAN",
}


def normalize_audio_label(model_name: str, label: str) -> str:
    """
    Normalizes native audio detector model labels to canonical binary labels:
    AI or HUMAN
    """
    if not label:
        raise ValueError("Cannot normalize an empty or None label string.")

    cleaned_label = str(label).strip().upper()

    if model_name == MODEL_1:
        mapping = MODEL_1_LABEL_MAP
    elif model_name == MODEL_2:
        mapping = MODEL_2_LABEL_MAP
    else:
        raise ValueError(f"Unsupported audio model: '{model_name}'")

    if cleaned_label not in mapping:
        raise ValueError(f"Unknown label '{label}' for model '{model_name}'")

    return mapping[cleaned_label]


def build_normalized_prediction(
    model_name: str,
    native_label: str,
    score: float
) -> Dict[str, Any]:
    """
    Constructs a normalized prediction dictionary for individual model outputs.
    """
    return {
        "model": model_name,
        "native_label": native_label,
        "normalized_label": normalize_audio_label(model_name, native_label),
        "score": float(score),
    }