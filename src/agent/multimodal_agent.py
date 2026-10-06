from pathlib import Path
from typing import Any, Dict, Optional

from src.detectors.audio import PrimaryDecisionAudioEnsemble
from src.detectors.image import ImageDetector
from src.detectors.text import TextDetectorEnsemble

from .modality_router import detect_modality


class MultimodalAgent:
    """High-level orchestration layer for text, image, and audio AI content detection."""

    def __init__(self):
        self.text_detector = TextDetectorEnsemble()
        self.image_detector = ImageDetector()
        self.audio_detector = PrimaryDecisionAudioEnsemble()

    def detect(
        self,
        file_path: str,
        text_content: Optional[str] = None,
        image_m1_score: Optional[float] = None,
        image_m2_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Route input file to the appropriate detector and return a unified output structure.

        Args:
            file_path: Path to the target file.
            text_content: Optional raw text string override for text files.
            image_m1_score: Optional Model 1 score for pre-computed image inference.
            image_m2_score: Optional Model 2 score for pre-computed image inference.

        Returns:
            Standardized dictionary containing modality, final label, score/confidence, and details.
        """
        modality = detect_modality(file_path)
        path = Path(file_path)

        if modality == "text":
            content = text_content if text_content is not None else path.read_text(encoding="utf-8")
            raw_result = self.text_detector.detect(content)

            # Extract fields from EnsembleTextResult
            final_pred = getattr(raw_result, "final_prediction", "likely_human")
            normalized_label = "AI" if final_pred == "likely_ai_generated" else "HUMAN"
            
            # Compute average AI score across models for reporting score
            tmr_score = getattr(getattr(raw_result, "tmr", None), "ai_score", 0.0)
            multi_score = getattr(getattr(raw_result, "multilingual", None), "ai_score", 0.0)
            grad_score = getattr(getattr(raw_result, "gradient", None), "ai_score", 0.0)
            avg_score = (tmr_score + multi_score + grad_score) / 3.0

            return {
                "modality": "text",
                "label": normalized_label,
                "score": round(avg_score, 4),
                "confidence_indicator": getattr(raw_result, "agreement_strength", "Moderate"),
                "details": {
                    "raw_prediction": final_pred,
                    "votes_ai": getattr(raw_result, "votes_ai", 0),
                    "votes_human": getattr(raw_result, "votes_human", 0),
                    "warning": getattr(raw_result, "warning", None),
                },
                "file_name": path.name,
            }

        elif modality == "image":
            m1 = image_m1_score if image_m1_score is not None else 0.5
            m2 = image_m2_score if image_m2_score is not None else 0.5

            raw_result = self.image_detector.predict(
                image_path=str(path),
                m1_score=m1,
                m2_score=m2,
            )

            raw_label = raw_result.get("label", "HUMAN")
            normalized_label = "AI" if str(raw_label).upper() in ["AI", "FAKE"] else "HUMAN"

            return {
                "modality": "image",
                "label": normalized_label,
                "score": float(raw_result.get("score", 0.0)),
                "confidence_indicator": raw_result.get("confidence_indicator", "Moderate"),
                "details": {
                    "models_agree": raw_result.get("models_agree", False),
                    "model_1": raw_result.get("model_1", {}),
                    "model_2": raw_result.get("model_2", {}),
                },
                "file_name": path.name,
            }

        elif modality == "audio":
            raw_result = self.audio_detector.predict(str(path))

            return {
                "modality": "audio",
                "label": raw_result.get("label", "HUMAN"),
                "score": float(raw_result.get("model_score", 0.0)),
                "confidence_indicator": raw_result.get("confidence_indicator", "Moderate"),
                "details": {
                    "decision_model": raw_result.get("decision_model", ""),
                    "decision_threshold": raw_result.get("decision_threshold", 0.90),
                    "models_agree": raw_result.get("models_agree", False),
                    "primary_agrees_with_final": raw_result.get("primary_agrees_with_final", False),
                },
                "file_name": path.name,
            }

        else:
            raise ValueError(f"Unsupported modality: {modality}")