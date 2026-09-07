import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.text.preprocessing import validate_and_clean_text, chunk_text_by_tokens
from src.detectors.text.detector import TextAIDetector

def test_validate_text_errors():
    with pytest.raises(TypeError):
        validate_and_clean_text(12345)

    with pytest.raises(ValueError):
        validate_and_clean_text("   \n \t ")

@pytest.mark.slow
def test_multilingual_detector_inference():
    detector = TextAIDetector(model_name="mujian2026/multilingual-ai-text-detector")
    
    human_sample = "Now i can say confidently that i know 50+% of the use of git and github. But the rest of 50% will be explore when i can work with more than 5 peoples on one same project."
    result = detector.detect(human_sample)

    assert result.status == "success"
    assert result.prediction in ["likely_human", "uncertain", "likely_ai_generated"]
    assert 0.0 <= result.ai_score <= 1.0
    assert 0.0 <= result.human_score <= 1.0
    assert result.threshold_used == 0.70

@pytest.mark.slow
def test_short_text_warning():
    detector = TextAIDetector(model_name="mujian2026/multilingual-ai-text-detector")
    short_sample = "Fix code bug."
    result = detector.detect(short_sample)

    assert result.warning is not None
    assert "Input text is short" in result.warning