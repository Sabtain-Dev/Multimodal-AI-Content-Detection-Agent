from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.detectors.audio.audio_detectors import AudioDetectorModel
from src.detectors.audio.label_normalization import compute_confidence_indicator

MODEL_1_NAME = "Sayantan090/audio-fake-detector"
MODEL_2_NAME = "Mahmoud59/wav2vec2-fake-audio-detector"
PRIMARY_MODEL_THRESHOLD = 0.90


class PrimaryDecisionAudioEnsemble:
    """
    Primary/Fallback Audio Ensemble Engine.
    Uses Model 2 (Mahmoud59) with threshold 0.90 as the primary decision maker,
    while capturing Model 1 as secondary diagnostic telemetry.
    """
    def __init__(self, device: Optional[Union[str, int]] = None):
        self.model_1 = AudioDetectorModel(MODEL_1_NAME, device=device)
        self.model_2 = AudioDetectorModel(MODEL_2_NAME, device=device)
        self.primary_threshold = PRIMARY_MODEL_THRESHOLD

    def predict(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        # Execute predictions on both models
        res_m1 = self.model_1.predict(audio_path)
        res_m2 = self.model_2.predict(audio_path)

        # Primary decision based on Model 2 threshold
        m2_ai_score = res_m2["ai_score"]
        if m2_ai_score >= self.primary_threshold:
            final_label = "AI"
            final_score = m2_ai_score
        else:
            final_label = "HUMAN"
            final_score = 1.0 - m2_ai_score

        # Distinct agreement indicators
        models_agree = (res_m1["normalized_label"] == res_m2["normalized_label"])
        agrees_with_final = (res_m2["normalized_label"] == final_label)

        return {
            "modality": "audio",
            "label": final_label,
            "model_score": round(final_score, 4),
            "confidence_indicator": compute_confidence_indicator(final_score),
            "model_1": {
                "name": res_m1["name"],
                "raw_label": res_m1["raw_label"],
                "normalized_label": res_m1["normalized_label"],
                "score": res_m1["score"],
                "ai_score": res_m1["ai_score"]
            },
            "model_2": {
                "name": res_m2["name"],
                "raw_label": res_m2["raw_label"],
                "normalized_label": res_m2["normalized_label"],
                "score": res_m2["score"],
                "ai_score": res_m2["ai_score"]
            },
            "models_agree": models_agree,
            "primary_agrees_with_final": agrees_with_final,
            "decision_model": MODEL_2_NAME,
            "decision_threshold": self.primary_threshold
        }