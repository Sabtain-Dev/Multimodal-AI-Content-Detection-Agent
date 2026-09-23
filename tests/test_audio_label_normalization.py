import pytest
from src.detectors.audio.label_normalization import (
    normalize_audio_label,
    build_normalized_prediction,
    MODEL_1,
    MODEL_2,
)


def test_model_1_labels():
    assert normalize_audio_label(MODEL_1, "FAKE") == "AI"
    assert normalize_audio_label(MODEL_1, "REAL") == "HUMAN"
    # Test case insensitivity
    assert normalize_audio_label(MODEL_1, "fake") == "AI"
    assert normalize_audio_label(MODEL_1, "real") == "HUMAN"


def test_model_2_labels():
    assert normalize_audio_label(MODEL_2, "LABEL_0") == "AI"
    assert normalize_audio_label(MODEL_2, "LABEL_1") == "HUMAN"
    # Test case insensitivity
    assert normalize_audio_label(MODEL_2, "label_0") == "AI"
    assert normalize_audio_label(MODEL_2, "label_1") == "HUMAN"


def test_normalized_prediction():
    result = build_normalized_prediction(
        MODEL_1,
        "FAKE",
        0.9561,
    )

    assert result["model"] == MODEL_1
    assert result["native_label"] == "FAKE"
    assert result["normalized_label"] == "AI"
    assert result["score"] == 0.9561


def test_unknown_label():
    with pytest.raises(ValueError, match="Unknown label 'UNKNOWN'"):
        normalize_audio_label(MODEL_1, "UNKNOWN")


def test_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported audio model"):
        normalize_audio_label("invalid/model-name", "FAKE")