from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ChunkResult:
    chunk_index: int
    ai_score: float
    human_score: float
    token_count: int

@dataclass
class TextDetectionResult:
    modality: str
    prediction: str
    ai_score: float
    human_score: float
    threshold_used: float
    model_name: str
    status: str
    chunks_analyzed: int
    ai_chunks: int
    aggregation_method: str
    chunk_results: List[ChunkResult] = field(default_factory=list)
    warning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "prediction": self.prediction,
            "ai_score": self.ai_score,
            "human_score": self.human_score,
            "threshold_used": self.threshold_used,
            "model_name": self.model_name,
            "status": self.status,
            "chunks_analyzed": self.chunks_analyzed,
            "ai_chunks": self.ai_chunks,
            "aggregation_method": self.aggregation_method,
            "chunk_results": [
                {
                    "chunk_index": c.chunk_index,
                    "ai_score": c.ai_score,
                    "human_score": c.human_score,
                    "token_count": c.token_count
                }
                for c in self.chunk_results
            ],
            "warning": self.warning,
            "metadata": self.metadata,
        }