import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, List

from .preprocessing import validate_and_clean_text, chunk_text_by_tokens
from .schemas import TextDetectionResult, ChunkResult

DEFAULT_MODEL_NAME = "Oxidane/tmr-ai-text-detector"
DEFAULT_THRESHOLD = 0.5
SHORT_TEXT_CHAR_THRESHOLD = 50

class TextAIDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_size: int = 480,
        overlap: int = 32
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _predict_chunk(self, chunk_text: str) -> tuple[float, float, int]:
        """Runs model inference on a single text chunk."""
        inputs = self.tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        token_count = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        # Label 0: Human, Label 1: AI (Standard TMR Mapping)
        human_score = float(probabilities[0].cpu().item())
        ai_score = float(probabilities[1].cpu().item())

        return ai_score, human_score, token_count

    def detect(self, text: str) -> TextDetectionResult:
        cleaned_text = validate_and_clean_text(text)
        
        # Determine short text warning
        warning = None
        if len(cleaned_text) < SHORT_TEXT_CHAR_THRESHOLD:
            warning = "Input text is very short. Detection scores may be less reliable."

        # Token-based chunking
        text_chunks = chunk_text_by_tokens(
            text=cleaned_text,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )

        chunk_results: List[ChunkResult] = []
        ai_chunk_count = 0

        for idx, chunk_str in enumerate(text_chunks):
            ai_score, human_score, token_count = self._predict_chunk(chunk_str)
            
            if ai_score >= self.threshold:
                ai_chunk_count += 1

            chunk_results.append(
                ChunkResult(
                    chunk_index=idx,
                    ai_score=round(ai_score, 4),
                    human_score=round(human_score, 4),
                    token_count=token_count
                )
            )

        # Aggregation: Mean AI probability across all analyzed chunks
        avg_ai_score = sum(c.ai_score for c in chunk_results) / len(chunk_results)
        avg_human_score = 1.0 - avg_ai_score

        prediction = "likely_ai_generated" if avg_ai_score >= self.threshold else "likely_human"

        return TextDetectionResult(
            modality="text",
            prediction=prediction,
            ai_score=round(avg_ai_score, 4),
            human_score=round(avg_human_score, 4),
            model_name=self.model_name,
            status="success",
            chunks_analyzed=len(chunk_results),
            ai_chunks=ai_chunk_count,
            aggregation_method="average",
            chunk_results=chunk_results,
            warning=warning,
            metadata={
                "device": self.device,
                "threshold": self.threshold,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            }
        )