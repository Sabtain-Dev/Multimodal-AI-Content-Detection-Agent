"""
Main ImageDetector implementation class for production inference integration.
"""
import logging
from typing import Dict, Any
from .ensemble import evaluate_image_ensemble

logger = logging.getLogger(__name__)


class ImageDetector:
    """
    Detector class managing dual-model image analysis pipeline.
    """

    def __init__(self, m1_threshold: float = 0.70, m2_threshold: float = 0.50):
        self.m1_threshold = m1_threshold
        self.m2_threshold = m2_threshold

    def predict(self, image_path: str, m1_score: float, m2_score: float) -> Dict[str, Any]:
        """
        Processes image inputs through active ensemble decision pipeline.
        
        Note: Model score inputs accept raw inference floats from underlying transformers.
        """
        logger.info(f"Running ImageDetector on: {image_path}")
        return evaluate_image_ensemble(m1_score=m1_score, m2_score=m2_score)