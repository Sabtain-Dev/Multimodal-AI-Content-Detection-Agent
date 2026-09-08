import importlib
from typing import Optional, List, Dict, Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .preprocessing import validate_and_clean_text, chunk_text_by_tokens
from .schemas import TextDetectionResult, ChunkResult

DEFAULT_MODEL_NAME = "mujian2026/multilingual-ai-text-detector"
DEFAULT_MODEL_SUBFOLDER = "fp32"
SHORT_TEXT_CHAR_THRESHOLD = 50

MODEL_DEFAULT_THRESHOLDS = {
    "Oxidane/tmr-ai-text-detector": 0.70,
    "mujian2026/multilingual-ai-text-detector": 0.50,
    "ShantanuT01/gradient-ai-text-detector": 0.50,
}

class TextAIDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        chunk_size: int = 480,
        overlap: int = 32
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold if threshold is not None else MODEL_DEFAULT_THRESHOLDS.get(model_name, 0.50)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.session = None
        if self.model_name == DEFAULT_MODEL_NAME:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, subfolder=DEFAULT_MODEL_SUBFOLDER
            )
            self.config = AutoConfig.from_pretrained(
                self.model_name, subfolder=DEFAULT_MODEL_SUBFOLDER
            )
            model_path = hf_hub_download(
                self.model_name,
                filename=f"{DEFAULT_MODEL_SUBFOLDER}/onnx/model.onnx",
            )
            ort = importlib.import_module("onnxruntime")
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self.model = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            self.config = self.model.config

        # Dynamically determine AI label index
        self.ai_label_index = 1
        for idx, label in self.config.id2label.items():
            if "ai" in str(label).lower() or "generated" in str(label).lower():
                self.ai_label_index = int(idx)
                break

    def _predict_chunk(self, chunk_text: str) -> tuple[float, float, int]:
        if self.session is not None:
            inputs = self.tokenizer(
                chunk_text,
                return_tensors="np",
                truncation=True,
                max_length=512,
            )
            logits = torch.from_numpy(self.session.run(["logits"], inputs)[0])
            if logits.shape[-1] == 1:
                probabilities = torch.sigmoid(logits).squeeze(0)
            else:
                probabilities = torch.softmax(logits, dim=-1).squeeze(0)
            token_count = inputs["input_ids"].shape[1]
        else:
            inputs = self.tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            token_count = inputs["input_ids"].shape[1]

            if self.model is None:
                raise RuntimeError("PyTorch model is unavailable for this detector")

            with torch.no_grad():
                outputs = self.model(**inputs)
                if outputs.logits.shape[-1] == 1:
                    probabilities = torch.sigmoid(outputs.logits).squeeze(0)
                else:
                    probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)

        if probabilities.shape[0] == 1:
            ai_score = float(probabilities[0].cpu().item())
        else:
            ai_score = float(probabilities[self.ai_label_index].cpu().item())
        human_score = 1.0 - ai_score

        return ai_score, human_score, token_count

    def detect(
        self,
        text: str,
        threshold: Optional[float] = None
    ) -> TextDetectionResult:
        active_threshold = threshold if threshold is not None else self.threshold

        cleaned_text = validate_and_clean_text(text)
        
        warning = None
        if len(cleaned_text) < SHORT_TEXT_CHAR_THRESHOLD:
            warning = "Input text is short. Detection confidence may be reduced."

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
            
            if ai_score >= active_threshold:
                ai_chunk_count += 1

            chunk_results.append(
                ChunkResult(
                    chunk_index=idx,
                    ai_score=round(ai_score, 4),
                    human_score=round(human_score, 4),
                    token_count=token_count
                )
            )

        avg_ai_score = sum(c.ai_score for c in chunk_results) / len(chunk_results)
        avg_human_score = 1.0 - avg_ai_score

        prediction = "likely_ai_generated" if avg_ai_score >= active_threshold else "likely_human"

        return TextDetectionResult(
            modality="text",
            prediction=prediction,
            ai_score=round(avg_ai_score, 4),
            human_score=round(avg_human_score, 4),
            threshold_used=active_threshold,
            model_name=self.model_name,
            status="success",
            chunks_analyzed=len(chunk_results),
            ai_chunks=ai_chunk_count,
            aggregation_method="average",
            chunk_results=chunk_results,
            warning=warning,
            metadata={
                "device": self.device,
                "threshold": active_threshold,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap
            }
        )