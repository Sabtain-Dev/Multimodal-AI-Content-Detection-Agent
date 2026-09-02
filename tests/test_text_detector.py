import pytest
from src.detectors.text.preprocessing import validate_and_clean_text, chunk_text_by_tokens
from src.detectors.text.detector import TextAIDetector

# Preprocessing & Chunking Tests (Fast)
def test_validate_text_errors():
    with pytest.raises(TypeError):
        validate_and_clean_text(123)

    with pytest.raises(ValueError):
        validate_and_clean_text("   ")

def test_chunk_invalid_overlap():
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(100))

    with pytest.raises(ValueError, match="Overlap must be strictly smaller"):
        chunk_text_by_tokens("sample text", DummyTokenizer(), chunk_size=50, overlap=50)

# Integration / Model Inference Tests (Marked as slow)
@pytest.mark.slow
def test_detector_long_text_chunking():
    detector = TextAIDetector(chunk_size=100, overlap=20)
    
    # Generate long text (~300 words)
    long_text = "Artificial intelligence and machine learning transform industries. " * 50
    result = detector.detect(long_text)

    assert result.status == "success"
    assert result.chunks_analyzed > 1
    assert len(result.chunk_results) == result.chunks_analyzed
    assert 0.0 <= result.ai_score <= 1.0
    assert 0.0 <= result.human_score <= 1.0

@pytest.mark.slow
def test_detector_short_text_warning():
    detector = TextAIDetector()
    short_text = "Hello world."
    result = detector.detect(short_text)

    assert result.status == "success"
    assert result.chunks_analyzed == 1
    assert result.warning is not None