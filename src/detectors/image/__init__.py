"""Image detection package exports."""

from .image_detectors import ImageDetector
from .label_normalization import normalize_image_label, build_normalized_prediction
from .ensemble import evaluate_image_ensemble

__all__ = [
    "ImageDetector",
    "normalize_image_label",
    "build_normalized_prediction",
    "evaluate_image_ensemble",
]