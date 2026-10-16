from pathlib import Path
from typing import Dict, Union
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from src.detectors.video.video_processor import VideoProcessor

MODEL_NAME = "eftt/VideoMae-ffc23-deepfake-detector"


class VideoDetectorModel:
    """Wrapper for the VideoMAE Deepfake Detector model."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.num_frames = int(getattr(self.model.config, "num_frames", 16))
        self.processor_engine = VideoProcessor(num_frames=self.num_frames)

    def predict(self, video_path: Union[str, Path]) -> Dict:
        """Extracts frames from video and returns classification results with explicit AI probability scoring."""
        frames = self.processor_engine.extract_uniform_frames(video_path)
        frames_array = np.asarray(frames)

        inputs = self.processor(list(frames_array), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        # Map model labels (id2label usually {0: 'real', 1: 'fake'})
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        
        # Extract class probabilities mapped to project terminology
        class_scores = {}
        ai_probability = 0.0
        
        for idx, prob in enumerate(probabilities):
            native_lbl = id2label.get(idx, "unknown")
            if native_lbl == "real":
                project_lbl = "HUMAN"
            elif native_lbl == "fake":
                project_lbl = "AI"
                ai_probability = float(prob.item())
            else:
                project_lbl = native_lbl.upper()
            
            class_scores[project_lbl] = float(prob.item())

        predicted_id = int(torch.argmax(probabilities).item())
        native_predicted_label = id2label.get(predicted_id, "real")
        
        final_label = "AI" if native_predicted_label == "fake" else "HUMAN"

        return {
            "modality": "video",
            "label": final_label,
            "model_score": ai_probability,  # Explicitly store AI probability for consistent thresholding
            "class_scores": class_scores,
            "native_label": native_predicted_label,
            "model_name": self.model_name,
            "num_frames": self.num_frames,
            "device": str(self.device),
        }