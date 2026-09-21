"""
Inspection script for Audio Detector Model 2: refikbklm/fake-audio-detector-model
Extracts architecture type, label mappings, and sampling rate settings.
"""
import sys
from transformers import AutoConfig

MODEL_ID = "refikbklm/fake-audio-detector-model"


def inspect_model_2():
    print("=" * 70)
    print("INSPECTING AUDIO DETECTOR 2")
    print("=" * 70)
    print(f"Model ID      : {MODEL_ID}")

    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
        print(f"Architecture  : {config.architectures}")
        print(f"Model type    : {config.model_type}")
        print(f"ID2LABEL      : {config.id2label}")
        print(f"LABEL2ID      : {config.label2id}")
        print(f"Sampling rate : {getattr(config, 'sampling_rate', 'not specified in config')}")
    except Exception as e:
        print(f"Error fetching configuration for {MODEL_ID}: {e}")
        sys.exit(1)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_model_2()