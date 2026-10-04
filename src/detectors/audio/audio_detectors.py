import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from transformers import pipeline

from src.detectors.audio.label_normalization import normalize_audio_label


class AudioDetectorModel:
    """
    Wrapper for single Audio Classification models.
    """
    def __init__(self, model_name: str, device: Optional[Union[str, int]] = None):
        self.model_name = model_name
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        self.classifier = pipeline(
            "audio-classification",
            model=self.model_name,
            device=self.device
        )

    def predict(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        path_str = str(path)
        start_time = time.time()
        
        # HuggingFace pipeline infer
        predictions = self.classifier(path_str)
        if not predictions:
            raise ValueError(f"Audio model returned no predictions for: {path}")
        elapsed_sec = round(time.time() - start_time, 4)

        # Top prediction
        top_pred = predictions[0]
        raw_label = top_pred["label"]
        score = float(top_pred["score"])
        norm_label = normalize_audio_label(self.model_name, raw_label)

        # Calculate explicit AI probability score
        if norm_label == "AI":
            ai_score = score
        elif norm_label == "HUMAN":
            ai_score = 1.0 - score
        return {
            "name": self.model_name,
            "raw_label": raw_label,
            "normalized_label": norm_label,
            "score": round(score, 4),
            "ai_score": round(ai_score, 4),
            "inference_time_sec": elapsed_sec
        }