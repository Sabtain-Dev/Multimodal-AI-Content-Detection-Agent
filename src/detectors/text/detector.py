import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

from .preprocessing import validate_and_clean_text
from .schemas import TextDetectionResult

DEFAULT_MODEL_NAME = "Oxidane/tmr-ai-text-detector"

class TextAIDetector:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()

        # Dynamic label mapping inspection
        self.id2label = self.model.config.id2label if hasattr(self.model.config, 'id2label') else {0: "human", 1: "ai"}

    def detect(self, text: str, threshold: float = 0.5) -> TextDetectionResult:
        cleaned_text = validate_and_clean_text(text)

        # Tokenize with truncation to model limits
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        # Map scores based on index inspection
        # Label 0: Human, Label 1: AI (Standard TMR Configuration)
        human_score = float(probabilities[0].cpu().item())
        ai_score = float(probabilities[1].cpu().item())

        prediction = "likely_ai_generated" if ai_score >= threshold else "likely_human"

        return TextDetectionResult(
            modality="text",
            prediction=prediction,
            ai_score=round(ai_score, 4),
            human_score=round(human_score, 4),
            model_name=self.model_name,
            status="success",
            metadata={
                "truncated": inputs["input_ids"].shape[1] >= 512,
                "input_token_count": inputs["input_ids"].shape[1],
                "device": self.device,
            }
        )