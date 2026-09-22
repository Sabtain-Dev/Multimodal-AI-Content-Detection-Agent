import sys
from transformers import AutoProcessor, AutoFeatureExtractor

MODEL_1_ID = "Sayantan090/audio-fake-detector"
MODEL_2_ID = "Mahmoud59/wav2vec2-fake-audio-detector"


def inspect_processor(model_id: str, label: str):
    print("=" * 60)
    print(f"{label} PROCESSOR / FEATURE EXTRACTOR")
    print(f"Model ID: {model_id}")
    print("=" * 60)
    try:
        try:
            proc = AutoProcessor.from_pretrained(model_id)
            print("Loaded via AutoProcessor:")
            print(proc)
            sampling_rate = getattr(proc, "sampling_rate", None) or getattr(getattr(proc, "feature_extractor", None), "sampling_rate", "not specified")
            print(f"Sampling rate: {sampling_rate}")
        except Exception:
            fe = AutoFeatureExtractor.from_pretrained(model_id)
            print("Loaded via AutoFeatureExtractor:")
            print(fe)
            print(f"Sampling rate: {getattr(fe, 'sampling_rate', 'not specified')}")
    except Exception as e:
        print(f"Failed to inspect {model_id}: {e}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    inspect_processor(MODEL_1_ID, "MODEL 1")
    inspect_processor(MODEL_2_ID, "MODEL 2")