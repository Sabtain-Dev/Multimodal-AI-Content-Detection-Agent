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
        """Extracts frames from video and returns classification results."""
        frames = self.processor_engine.extract_uniform_frames(video_path)

        # Convert frame list to numpy array
        frames_array = np.asarray(frames)

        # Prepare inputs using HF Image Processor
        inputs = self.processor(list(frames_array), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        native_label = self.model.config.id2label[predicted_id].lower()

        label_map = {
            "real": "HUMAN",
            "fake": "AI",
        }

        if native_label not in label_map:
            raise ValueError(f"Unsupported video model label: {native_label}")

        return {
            "modality": "video",
            "label": label_map[native_label],
            "model_score": float(probabilities[predicted_id].item()),
            "class_scores": {
                label_map[
                    self.model.config.id2label[index].lower()
                ]: float(probabilities[index].item())
                for index in range(len(probabilities))
            },
            "native_label": native_label,
            "model_name": self.model_name,
            "num_frames": self.num_frames,
            "device": str(self.device),
        }