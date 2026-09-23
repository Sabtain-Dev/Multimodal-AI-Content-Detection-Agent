"""
Audio Detection Package Initializer.
"""
from .label_normalization import (
    normalize_audio_label,
    build_normalized_prediction,
)

__all__ = [
    "normalize_audio_label",
    "build_normalized_prediction",
]