import pytest
from src.detectors.audio import (
    normalize_audio_label,
    compute_confidence_indicator,
    PrimaryDecisionAudioEnsemble,
)
from src.detectors.audio import ensemble as audio_ensemble
from src.detectors.audio import audio_detectors


def test_model_specific_label_normalization():
    assert normalize_audio_label(audio_ensemble.MODEL_1_NAME, "FAKE") == "AI"
    assert normalize_audio_label(audio_ensemble.MODEL_1_NAME, "REAL") == "HUMAN"
    assert normalize_audio_label(audio_ensemble.MODEL_2_NAME, "LABEL_0") == "AI"
    assert normalize_audio_label(audio_ensemble.MODEL_2_NAME, "LABEL_1") == "HUMAN"


def test_confidence_indicator():
    assert compute_confidence_indicator(0.95) == "High"
    assert compute_confidence_indicator(0.05) == "High"
    assert compute_confidence_indicator(0.80) == "Moderate"
    assert compute_confidence_indicator(0.50) == "Low"


def test_primary_decision_ensemble_uses_model_2_threshold(monkeypatch):
    model_results = iter(
        [
            {
                "name": audio_ensemble.MODEL_1_NAME,
                "raw_label": "FAKE",
                "normalized_label": "AI",
                "score": 0.8,
                "ai_score": 0.8,
            },
            {
                "name": audio_ensemble.MODEL_2_NAME,
                "raw_label": "LABEL_0",
                "normalized_label": "AI",
                "score": 0.88,
                "ai_score": 0.88,
            },
        ]
    )

    class StubAudioDetectorModel:
        def __init__(self, model_name, device=None):
            self.model_name = model_name

        def predict(self, audio_path):
            return next(model_results)

    monkeypatch.setattr(audio_ensemble, "AudioDetectorModel", StubAudioDetectorModel)
    result = PrimaryDecisionAudioEnsemble().predict("sample.wav")

    assert result["modality"] == "audio"
    assert result["label"] == "HUMAN"
    assert result["model_score"] == pytest.approx(0.12)
    assert result["decision_threshold"] == 0.90
    assert "model_1" in result
    assert "model_2" in result
    assert result["models_agree"] is True
    assert result["primary_agrees_with_final"] is False


@pytest.mark.parametrize(
    ("native_label", "expected_label", "expected_ai_score"),
    [
        ("LABEL_0", "AI", 0.8),
        ("LABEL_1", "HUMAN", 0.2),
    ],
)
def test_audio_model_maps_model_2_scores(
    monkeypatch, tmp_path, native_label, expected_label, expected_ai_score
):
    monkeypatch.setattr(
        audio_detectors,
        "pipeline",
        lambda *args, **kwargs: lambda _path: [
            {"label": native_label, "score": 0.8}
        ],
    )

    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    result = audio_detectors.AudioDetectorModel(
        audio_ensemble.MODEL_2_NAME,
        device="cpu",
    ).predict(audio_path)

    assert result["normalized_label"] == expected_label
    assert result["ai_score"] == pytest.approx(expected_ai_score)