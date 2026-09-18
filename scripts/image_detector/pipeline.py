import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

M1_THRESHOLD = 0.70
M2_THRESHOLD = 0.50


def choose_final_label(model_1_label: str, model_2_label: str) -> str:
    """
    Always returns AI or HUMAN.
    Model 2 acts as primary classifier; Model 1 acts as supporting validator.
    """
    if model_1_label == model_2_label:
        return model_1_label
    return model_2_label


def derive_confidence_indicator(primary_score: float, models_agree: bool) -> str:
    """
    Maps uncalibrated model scores to qualitative confidence tiers.
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


def process_image_detection(
    image_path: str,
    m1_score: float,
    m2_score: float
) -> Dict[str, Any]:
    """
    Executes the dual-model image decision policy.
    Returns user-facing result alongside internal telemetry metadata.
    """
    p1 = "AI" if m1_score >= M1_THRESHOLD else "HUMAN"
    p2 = "AI" if m2_score >= M2_THRESHOLD else "HUMAN"

    models_agree = (p1 == p2)
    final_label = choose_final_label(p1, p2)
    decision_mode = "consensus" if models_agree else "model_2_primary_fallback"

    # Compute score for the winning prediction
    raw_model_score = m2_score if final_label == "AI" else (1.0 - m2_score)
    confidence_indicator = derive_confidence_indicator(
        primary_score=raw_model_score,
        models_agree=models_agree
    )

    # User-facing output (Strict AI/HUMAN binary response)
    user_payload = {
        "modality": "image",
        "label": final_label,
        "model_score": round(raw_model_score, 4),
        "confidence_indicator": confidence_indicator
    }

    # Internal telemetry payload
    internal_metadata = {
        "final_label": final_label,
        "primary_model": "haywoodsloan/ai-image-detector-deploy",
        "primary_model_label": p2,
        "primary_model_score": round(m2_score, 4),
        "secondary_model": "Ateeqq/ai-vs-human-image-detector",
        "secondary_model_label": p1,
        "secondary_model_score": round(m1_score, 4),
        "models_agree": models_agree,
        "decision_mode": decision_mode
    }

    return {
        "user_result": user_payload,
        "internal_telemetry": internal_metadata
    }


if __name__ == "__main__":
    # Internal execution test case
    sample_result = process_image_detection(
        image_path="samples/test_01.jpg",
        m1_score=0.82,  # Model 1 -> AI
        m2_score=0.15   # Model 2 -> HUMAN
    )
    
    print("--- USER-FACING PAYLOAD ---")
    print(sample_result["user_result"])
    print("\n--- INTERNAL TELEMETRY METADATA ---")
    print(sample_result["internal_telemetry"])