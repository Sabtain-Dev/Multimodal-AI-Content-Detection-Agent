import gc
from typing import Optional
import torch

from .detector import TextAIDetector
from .schemas import EnsembleTextResult, ModelResultSummary

TMR_MODEL_NAME = "Oxidane/tmr-ai-text-detector"
MULTILINGUAL_MODEL_NAME = "mujian2026/multilingual-ai-text-detector"
GRADIENT_MODEL_NAME = "ShantanuT01/gradient-ai-text-detector"

class TextDetectorEnsemble:
    def __init__(
        self,
        tmr_threshold: float = 0.70,
        multilingual_threshold: float = 0.50,
        gradient_threshold: float = 0.50,
        device: Optional[str] = None
    ):
        self.tmr_threshold = tmr_threshold
        self.multilingual_threshold = multilingual_threshold
        self.gradient_threshold = gradient_threshold
        self.device = device

    def _free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_single_detector(self, model_name: str, threshold: float, text: str):
        detector = TextAIDetector(model_name=model_name, threshold=threshold, device=self.device)
        result = detector.detect(text)
        del detector
        self._free_memory()
        return result

    def detect(self, text: str) -> EnsembleTextResult:
        # Sequential inference to optimize RAM usage
        tmr_res = self._run_single_detector(TMR_MODEL_NAME, self.tmr_threshold, text)
        multi_res = self._run_single_detector(MULTILINGUAL_MODEL_NAME, self.multilingual_threshold, text)
        grad_res = self._run_single_detector(GRADIENT_MODEL_NAME, self.gradient_threshold, text)

        tmr_summary = ModelResultSummary(
            model_name=tmr_res.model_name,
            ai_score=tmr_res.ai_score,
            human_score=tmr_res.human_score,
            prediction=tmr_res.prediction,
            threshold_used=tmr_res.threshold_used
        )

        multi_summary = ModelResultSummary(
            model_name=multi_res.model_name,
            ai_score=multi_res.ai_score,
            human_score=multi_res.human_score,
            prediction=multi_res.prediction,
            threshold_used=multi_res.threshold_used
        )

        grad_summary = ModelResultSummary(
            model_name=grad_res.model_name,
            ai_score=grad_res.ai_score,
            human_score=grad_res.human_score,
            prediction=grad_res.prediction,
            threshold_used=grad_res.threshold_used
        )

        # 3-Model Majority Voting Logic
        predictions = [tmr_res.prediction, multi_res.prediction, grad_res.prediction]
        ai_votes = predictions.count("likely_ai_generated")
        human_votes = predictions.count("likely_human")

        if ai_votes >= 2:
            final_prediction = "likely_ai_generated"
            agreement_strength = "Strong" if ai_votes == 3 else "Moderate"
        else:
            final_prediction = "likely_human"
            agreement_strength = "Strong" if human_votes == 3 else "Moderate"

        warning = tmr_res.warning or multi_res.warning or grad_res.warning

        return EnsembleTextResult(
            modality="text",
            final_prediction=final_prediction,
            votes_ai=ai_votes,
            votes_human=human_votes,
            agreement_strength=agreement_strength,
            tmr=tmr_summary,
            multilingual=multi_summary,
            gradient=grad_summary,
            warning=warning
        )