import struct
import wave
import pytest
from src.agent.modality_router import detect_modality
from src.agent.multimodal_agent import MultimodalAgent


def create_dummy_wav(file_path, duration_sec: float = 0.5, sample_rate: int = 16000):
    """Utility function to create a valid silent WAV file for testing."""
    num_samples = int(sample_rate * duration_sec)
    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = struct.pack(f"<{num_samples}h", *[0] * num_samples)
        wav_file.writeframes(frames)


# --- Router Unit Tests ---

def test_routing_text(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Sample text content", encoding="utf-8")
    assert detect_modality(str(f)) == "text"


def test_routing_markdown(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text("# Heading", encoding="utf-8")
    assert detect_modality(str(f)) == "text"


def test_routing_image(tmp_path):
    f = tmp_path / "sample.png"
    f.write_bytes(b"dummy image data")
    assert detect_modality(str(f)) == "image"


def test_routing_audio(tmp_path):
    f = tmp_path / "sample.mp3"
    f.write_bytes(b"dummy audio data")
    assert detect_modality(str(f)) == "audio"


def test_routing_case_insensitivity(tmp_path):
    f = tmp_path / "sample.WAV"
    f.write_bytes(b"dummy audio data")
    assert detect_modality(str(f)) == "audio"


def test_unsupported_modality_raises_error(tmp_path):
    f = tmp_path / "sample.mp4"
    f.write_bytes(b"dummy video data")
    with pytest.raises(ValueError, match="Unsupported modality"):
        detect_modality(str(f))


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        detect_modality("non_existent_file.txt")


# --- MultimodalAgent Integration Tests ---

def test_multimodal_agent_text_detection(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("This is an ordinary human sentence.", encoding="utf-8")
    agent = MultimodalAgent()
    res = agent.detect(str(f))

    assert res["modality"] == "text"
    assert res["label"] in ["AI", "HUMAN"]
    assert "score" in res
    assert "confidence_indicator" in res
    assert res["file_name"] == "sample.txt"


def test_multimodal_agent_image_detection(tmp_path):
    f = tmp_path / "sample.png"
    f.write_bytes(b"dummy image bytes")
    agent = MultimodalAgent()
    res = agent.detect(str(f), image_m1_score=0.85, image_m2_score=0.90)

    assert res["modality"] == "image"
    assert res["label"] in ["AI", "HUMAN"]
    assert "details" in res
    assert res["details"]["models_agree"] is True


def test_multimodal_agent_audio_detection(tmp_path):
    f = tmp_path / "sample.wav"
    create_dummy_wav(f)
    agent = MultimodalAgent()
    res = agent.detect(str(f))

    assert res["modality"] == "audio"
    assert res["label"] in ["AI", "HUMAN"]
    assert "score" in res
    assert "details" in res
    assert res["file_name"] == "sample.wav"