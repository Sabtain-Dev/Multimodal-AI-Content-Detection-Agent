from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TextDetectionResult:
    modality: str
    prediction: str
    ai_score: float
    human_score: float
    model_name: str
    status: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "prediction": self.prediction,
            "ai_score": self.ai_score,
            "human_score": self.human_score,
            "model_name": self.model_name,
            "status": self.status,
            "metadata": self.metadata,
        }