import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.text.preprocessing import validate_and_clean_text
from src.detectors.text.detector import TextAIDetector
from src.detectors.text.ensemble import TextDetectorEnsemble

def test_validate_text_errors():
    with pytest.raises(TypeError):
        validate_and_clean_text(12345)

    with pytest.raises(ValueError):
        validate_and_clean_text("   \n \t ")

@pytest.mark.slow
def test_multilingual_detector_inference():
    detector = TextAIDetector(model_name="mujian2026/multilingual-ai-text-detector")
    
    human_sample = (
        "Now i can say confidently that i know 50+% of the use of git and github. "
        "But the rest of 50% will be explore when i can work with more than 5 peoples on one same project."
    )
    result = detector.detect(human_sample)

    assert result.status == "success"
    assert result.prediction in ["likely_human", "likely_ai_generated"]
    assert 0.0 <= result.ai_score <= 1.0
    assert 0.0 <= result.human_score <= 1.0
    assert result.threshold_used == 0.50

@pytest.mark.slow
def test_short_text_warning():
    detector = TextAIDetector(model_name="mujian2026/multilingual-ai-text-detector")
    short_sample = "Fix code bug."
    result = detector.detect(short_sample)

    assert result.warning is not None
    assert "Input text is short" in result.warning

@pytest.mark.slow
def test_ensemble_agent_3model_execution():
    agent = TextDetectorEnsemble()
    sample_text = (
        "In this paper, we explore the application of neural networks in medical imaging "
        "and evaluate their diagnostics accuracy across standard benchmark datasets."
    )
    res = agent.detect(sample_text)

    assert res.modality == "text"
    assert res.final_prediction in ["likely_human", "likely_ai_generated"]
    assert res.votes_ai + res.votes_human == 3
    assert res.agreement_strength in ["Strong", "Moderate"]
    assert res.gradient.ai_score >= 0.0