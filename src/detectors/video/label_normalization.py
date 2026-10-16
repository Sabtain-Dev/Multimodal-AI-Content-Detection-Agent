def normalize_video_prediction(native_label: str, score: float) -> dict:
    """Normalizes native video model labels into the project's unified AI/HUMAN schema.

    Args:
        native_label: The model's native label ('real' or 'fake').
        score: The model's prediction confidence score.

    Returns:
        Standardized dictionary containing unified label and confidence assessment.
    """
    clean_label = native_label.strip().lower()

    label_map = {
        "real": "HUMAN",
        "fake": "AI",
    }

    if clean_label not in label_map:
        raise ValueError(f"Unsupported native video label: '{native_label}'. Expected 'real' or 'fake'.")

    unified_label = label_map[clean_label]

    # Determine confidence indicator based on score thresholds
    if score >= 0.85:
        confidence = "High"
    elif score >= 0.60:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "label": unified_label,
        "score": round(float(score), 4),
        "confidence_indicator": confidence,
        "native_label": clean_label,
    }