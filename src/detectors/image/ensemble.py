"""
Decision policy and ensemble logic for the image detection branch.
Combines primary and secondary image classifiers into a unified decision payload.
"""
from typing import Dict, Any, Optional

M1_THRESHOLD = 0.70
M2_THRESHOLD = 0.50

MODEL_1_NAME = "Ateeqq/ai-vs-human-image-detector"
MODEL_2_NAME = "haywoodsloan/ai-image-detector-deploy"


def choose_final_label(model_1_label: str, model_2_label: str) -> str:
    """
    Returns AI or HUMAN.
    Model 2 acts as primary decision maker; Model 1 acts as supporting validator.
    """
    if model_1_label == model_2_label:
        return model_1_label
    return model_2_label


def derive_confidence_indicator(primary_score: float, models_agree: bool) -> str:
    """
    Derives qualitative confidence indicators.
    Caps confidence at 'Moderate' if models disagree.
    """
    if not models_agree:
        return "Moderate"

    if primary_score >= 0.90:
        return "High"
    elif primary_score >= 0.70:
        return "Moderate"
    else:
        return "Low"


def evaluate_image_ensemble(
    m1_score: float,
    m2_score: float,
    m1_raw_label: Optional[str] = None,
    m2_raw_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes ensemble voting and returns complete output payload with telemetry.
    """
    p1 = "AI" if m1_score >= M1_THRESHOLD else "HUMAN"
    p2 = "AI" if m2_score >= M2_THRESHOLD else "HUMAN"

    models_agree = (p1 == p2)
    final_label = choose_final_label(p1, p2)
    decision_mode = "consensus" if models_agree else "model_2_primary_fallback"

    raw_model_score = m2_score if final_label == "AI" else (1.0 - m2_score)
    confidence_indicator = derive_confidence_indicator(
        primary_score=raw_model_score,
        models_agree=models_agree
    )

    return {
        "modality": "image",
        "label": final_label,
        "model_score": round(raw_model_score, 4),
        "confidence_indicator": confidence_indicator,
        "model_1": {
            "name": MODEL_1_NAME,
            "label": p1,
            "score": round(m1_score, 4)
        },
        "model_2": {
            "name": MODEL_2_NAME,
            "label": p2,
            "score": round(m2_score, 4)
        },
        "models_agree": models_agree,
        "decision_mode": decision_mode
    }